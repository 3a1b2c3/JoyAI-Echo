import type {
  ConnectionStatus,
  InboundEvent,
  MemoryAssetUpload,
  ShotMemoryAssetCreate,
  MemorySlotReference,
  Outbound,
  OutboundMedia,
  StoryProfile,
  WorkplaceData,
} from "./types";

/** WebSocket readyState constants, referenced by value to stay portable
 * across runtimes that don't expose a global ``WebSocket`` (tests, SSR). */
const WS_OPEN = 1;
const WS_CLOSING = 2;
/** Workplace save actions can carry large payloads; allow ample server round-trip. */
const WORKPLACE_ACTION_TIMEOUT_MS = 60_000;
/** Cap offline delta frames per chat (Director can stream heavily). */
const MAX_OFFLINE_DELTAS = 500;

type Unsubscribe = () => void;
type EventHandler = (ev: InboundEvent) => void;
type StatusHandler = (status: ConnectionStatus) => void;
type PeHandler = (chatId: string, active: string) => void;

/** Structured connection-level errors surfaced to the UI.
 *
 * These are *not* InboundEvent errors from the server application layer —
 * those arrive as ``{event: "error"}`` messages via ``onChat``. These are
 * transport-level or protocol-level faults the UI should make visible so
 * the user understands *why* their action failed (as opposed to silently
 * reconnecting under the hood).
 */
export type StreamError =
  /** Server rejected the inbound frame as too large (WS close code 1009).
   * Typically means the user attached images whose base64 size exceeded
   * ``maxMessageBytes`` on the server. */
  | { kind: "message_too_big" };

type ErrorHandler = (error: StreamError) => void;

interface PendingNewChat {
  resolve: (chatId: string) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

type PendingWorkplaceAction = {
  resolve: (value: { work_id: string; workplace: WorkplaceData }) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

export interface NanobotClientOptions {
  url: string;
  reconnect?: boolean;
  /** Called when a connection drops so the app can refresh its token. */
  onReauth?: () => Promise<string | null>;
  /** Inject a custom WebSocket factory (used by unit tests). */
  socketFactory?: (url: string) => WebSocket;
  /** Delay-cap for reconnect backoff (ms). */
  maxBackoffMs?: number;
}

/**
 * Singleton WebSocket client that multiplexes chat streams.
 *
 * One socket carries many chat_ids: the server tags every outbound event with
 * ``chat_id``, and this class fans those events out to handlers registered
 * per chat. Reconnects are transparent and re-attach every known chat_id.
 */
export class NanobotClient {
  private socket: WebSocket | null = null;
  private statusHandlers = new Set<StatusHandler>();
  private errorHandlers = new Set<ErrorHandler>();
  /** Server-global PE-set change subscribers (event: "pe_updated"). */
  private peHandlers = new Set<PeHandler>();
  // chat_id -> handlers listening on it
  private chatHandlers = new Map<string, Set<EventHandler>>();
  // chat_ids we've attached to since connect; re-attached after reconnects
  private knownChats = new Set<string>();
  private pendingNewChat: PendingNewChat | null = null;
  /** request_id → Promise for workplace_save_* responses (workplace_action_ok/error). */
  private pendingWorkplaceActions = new Map<string, PendingWorkplaceAction>();
  // Frames queued while the socket is not yet OPEN
  private sendQueue: Outbound[] = [];
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly shouldReconnect: boolean;
  private readonly maxBackoffMs: number;
  private readonly socketFactory: (url: string) => WebSocket;
  private currentUrl: string;
  private status_: ConnectionStatus = "idle";
  private readyChatId: string | null = null;
  // Set by ``close()`` so the onclose handler knows the drop was intentional
  // and must not schedule a reconnect or flip status back to "reconnecting".
  private intentionallyClosed = false;
  /** Events received while no UI handler is subscribed (e.g. user switched chats). */
  private offlineBuffers = new Map<string, InboundEvent[]>();
  /** Coalesce multiple onChat calls in one React commit into a single replay. */
  private replayScheduled = new Set<string>();

  constructor(private options: NanobotClientOptions) {
    this.shouldReconnect = options.reconnect ?? true;
    this.maxBackoffMs = options.maxBackoffMs ?? 15_000;
    this.socketFactory =
      options.socketFactory ?? ((url) => new WebSocket(url));
    this.currentUrl = options.url;
  }

  get status(): ConnectionStatus {
    return this.status_;
  }

  get defaultChatId(): string | null {
    return this.readyChatId;
  }

  /** Swap the URL (e.g. after fetching a fresh token) then reconnect. */
  updateUrl(url: string): void {
    this.currentUrl = url;
  }

  onStatus(handler: StatusHandler): Unsubscribe {
    this.statusHandlers.add(handler);
    handler(this.status_);
    return () => {
      this.statusHandlers.delete(handler);
    };
  }

  /** Subscribe to transport-level faults (see :type:`StreamError`). */
  onError(handler: ErrorHandler): Unsubscribe {
    this.errorHandlers.add(handler);
    return () => {
      this.errorHandlers.delete(handler);
    };
  }

  /** Subscribe to per-chat PE-set changes (``pe_updated`` / ``attached.active_pe``). */
  onPe(handler: PeHandler): Unsubscribe {
    this.peHandlers.add(handler);
    return () => {
      this.peHandlers.delete(handler);
    };
  }

  /** Ask the server to bind the PE set for a given chat's session. */
  setPe(name: string, chatId: string): void {
    this.queueSend({ type: "set_pe", name, chat_id: chatId });
  }

  /** Whether events are queued for subscribe-time replay (user was away). */
  hasOfflineBuffer(chatId: string): boolean {
    const buf = this.offlineBuffers.get(chatId);
    return !!buf && buf.length > 0;
  }

  /** Replay buffered events synchronously (after ``onChat`` registers a handler). */
  replayOfflineNow(chatId: string): void {
    this.flushOfflineBuffer(chatId);
  }

  /** Subscribe to events for a given chat_id. Auto-attaches on the next open. */
  onChat(
    chatId: string,
    handler: EventHandler,
  ): Unsubscribe {
    let handlers = this.chatHandlers.get(chatId);
    if (!handlers) {
      handlers = new Set();
      this.chatHandlers.set(chatId, handlers);
    }
    handlers.add(handler);
    this.attach(chatId);
    const pending = this.offlineBuffers.get(chatId);
    if (pending && pending.length > 0) {
      this.scheduleOfflineReplay(chatId);
    }
    return () => {
      const current = this.chatHandlers.get(chatId);
      if (!current) return;
      current.delete(handler);
      if (current.size === 0) this.chatHandlers.delete(chatId);
    };
  }

  connect(): void {
    if (this.socket && this.socket.readyState < WS_CLOSING) return;
    this.intentionallyClosed = false;
    this.setStatus("connecting");
    const sock = this.socketFactory(this.currentUrl);
    this.socket = sock;
    sock.onopen = () => this.handleOpen();
    sock.onmessage = (ev) => this.handleMessage(ev);
    sock.onerror = () => this.setStatus("error");
    sock.onclose = (ev) => this.handleClose(ev);
  }

  close(): void {
    this.intentionallyClosed = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    const sock = this.socket;
    this.socket = null;
    try {
      sock?.close();
    } catch {
      // ignore
    }
    this.offlineBuffers.clear();
    this.replayScheduled.clear();
    this.setStatus("closed");
  }

  /** Ask the server to provision a new chat_id; resolves with the assigned id. */
  newChat(
    timeoutMs: number = 5_000,
    options?: { autoGenerate?: boolean },
  ): Promise<string> {
    if (this.pendingNewChat) {
      return Promise.reject(new Error("newChat already in flight"));
    }
    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingNewChat = null;
        reject(new Error("newChat timed out"));
      }, timeoutMs);
      this.pendingNewChat = {
        resolve,
        reject,
        timer,
      };
      this.queueSend({
        type: "new_chat",
        ...(options?.autoGenerate ? { autoGenerate: true } : {}),
      });
    });
  }

  attach(chatId: string): void {
    this.knownChats.add(chatId);
    if (this.socket?.readyState === WS_OPEN) {
      this.queueSend(this.buildAttachFrame(chatId));
    }
  }

  sendMessage(
    chatId: string,
    content: string,
    media?: OutboundMedia[],
    extras?: {
      temperature?: number;
      top_p?: number;
      top_k?: number;
      autoGenerate?: boolean;
      duration_sec?: number;
      reference_image_url?: string;
      reference_image_name?: string;
      reference_image_width?: number;
      reference_image_height?: number;
    },
  ): void {
    this.knownChats.add(chatId);
    const base =
      media && media.length > 0
        ? { type: "message" as const, chat_id: chatId, content, media }
        : { type: "message" as const, chat_id: chatId, content };
    const frame: Outbound = extras ? { ...base, ...extras } : base;
    this.queueSend(frame);
  }

  answerQuestion(
    chatId: string,
    questionBatchId: string,
    cardId: string,
    value: string,
  ): void {
    this.knownChats.add(chatId);
    this.queueSend({
      type: "answer_question",
      chat_id: chatId,
      question_batch_id: questionBatchId,
      card_id: cardId,
      value,
    });
  }

  /** Persist story markdown over WS (avoids HTTP header size limits on large scripts). */
  saveWorkplaceStory(
    chatId: string,
    storyMd: string,
  ): Promise<{ work_id: string; workplace: WorkplaceData }> {
    const trimmed = storyMd.trim();
    if (!trimmed) {
      return Promise.reject(new Error("story_md cannot be empty"));
    }
    const { requestId, promise } = this.registerWorkplaceAction();
    this.attach(chatId);
    this.queueSend({
      type: "workplace_save_story",
      chat_id: chatId,
      request_id: requestId,
      story_md: trimmed,
    });
    return promise;
  }

  /** Persist story profile (shot plan) over WS. */
  saveWorkplaceStoryProfile(
    chatId: string,
    storyProfile: StoryProfile,
  ): Promise<{ work_id: string; workplace: WorkplaceData }> {
    const { requestId, promise } = this.registerWorkplaceAction();
    this.attach(chatId);
    this.queueSend({
      type: "workplace_save_story_profile",
      chat_id: chatId,
      request_id: requestId,
      story_profile: storyProfile,
    });
    return promise;
  }

  /** Persist a first-frame image over WS so its data URL isn't put in HTTP headers. */
  saveWorkplaceReferenceImage(
    chatId: string,
    image: {
      url: string;
      name?: string;
      width?: number;
      height?: number;
    },
  ): Promise<{ work_id: string; workplace: WorkplaceData }> {
    const { requestId, promise } = this.registerWorkplaceAction();
    this.attach(chatId);
    this.queueSend({
      type: "workplace_save_reference_image",
      chat_id: chatId,
      request_id: requestId,
      image,
    });
    return promise;
  }

  /** Add or update a local Memory Workspace asset over WebSocket. */
  saveWorkplaceMemoryAsset(
    chatId: string,
    asset: MemoryAssetUpload,
  ): Promise<{ work_id: string; workplace: WorkplaceData }> {
    const { requestId, promise } = this.registerWorkplaceAction();
    this.attach(chatId);
    this.queueSend({
      type: "workplace_save_memory_asset",
      chat_id: chatId,
      request_id: requestId,
      asset,
    });
    return promise;
  }

  /** Extract a reusable frame/audio clip from one generated shot. */
  createWorkplaceShotMemoryAsset(
    chatId: string,
    shotId: number,
    asset: ShotMemoryAssetCreate,
  ): Promise<{ work_id: string; workplace: WorkplaceData }> {
    const { requestId, promise } = this.registerWorkplaceAction();
    this.attach(chatId);
    this.queueSend({
      type: "workplace_create_shot_memory_asset",
      chat_id: chatId,
      request_id: requestId,
      shot_id: shotId,
      asset,
    });
    return promise;
  }

  /** Remove a locally uploaded asset from the Memory Workspace. */
  deleteWorkplaceMemoryAsset(
    chatId: string,
    assetId: string,
  ): Promise<{ work_id: string; workplace: WorkplaceData }> {
    const { requestId, promise } = this.registerWorkplaceAction();
    this.attach(chatId);
    this.queueSend({
      type: "workplace_delete_memory_asset",
      chat_id: chatId,
      request_id: requestId,
      asset_id: assetId,
    });
    return promise;
  }

  /** Apply an ordered Memory Workspace selection to one shot. */
  saveWorkplaceShotMemorySlots(
    chatId: string,
    shotId: number,
    slots: MemorySlotReference[],
  ): Promise<{ work_id: string; workplace: WorkplaceData }> {
    const { requestId, promise } = this.registerWorkplaceAction();
    this.attach(chatId);
    this.queueSend({
      type: "workplace_save_shot_memory_slots",
      chat_id: chatId,
      request_id: requestId,
      shot_id: shotId,
      slots,
    });
    return promise;
  }

  // -- internals ---------------------------------------------------------

  private buildAttachFrame(chatId: string): Outbound {
    return { type: "attach", chat_id: chatId };
  }

  private setStatus(status: ConnectionStatus): void {
    if (this.status_ === status) return;
    this.status_ = status;
    for (const handler of this.statusHandlers) handler(status);
  }

  private handleOpen(): void {
    this.setStatus("open");
    this.reconnectAttempts = 0;
    // Re-attach every known chat_id so deliveries continue routing after a drop.
    for (const chatId of this.knownChats) {
      this.rawSend(this.buildAttachFrame(chatId));
    }
    // Flush anything queued during reconnect.
    const queued = this.sendQueue.splice(0);
    for (const frame of queued) this.rawSend(frame);
  }

  private handleMessage(ev: MessageEvent): void {
    let parsed: InboundEvent;
    try {
      parsed = JSON.parse(typeof ev.data === "string" ? ev.data : "") as InboundEvent;
    } catch {
      return;
    }

    if (parsed.event === "ready") {
      this.readyChatId = parsed.chat_id;
      this.knownChats.add(parsed.chat_id);
      return;
    }

    if (parsed.event === "attached") {
      this.knownChats.add(parsed.chat_id);
      if (this.pendingNewChat) {
        clearTimeout(this.pendingNewChat.timer);
        this.pendingNewChat.resolve(parsed.chat_id);
        this.pendingNewChat = null;
      }
      if (parsed.active_pe) {
        for (const handler of this.peHandlers) {
          handler(parsed.chat_id, parsed.active_pe);
        }
      }
      this.dispatch(parsed.chat_id, parsed);
      return;
    }

    if (parsed.event === "workplace_action_ok") {
      const pending = this.pendingWorkplaceActions.get(parsed.request_id);
      if (pending) {
        clearTimeout(pending.timer);
        this.pendingWorkplaceActions.delete(parsed.request_id);
        pending.resolve({
          work_id: parsed.work_id,
          workplace: parsed.workplace,
        });
      }
      return;
    }

    if (parsed.event === "workplace_action_error") {
      const pending = this.pendingWorkplaceActions.get(parsed.request_id);
      if (pending) {
        clearTimeout(pending.timer);
        this.pendingWorkplaceActions.delete(parsed.request_id);
        pending.reject(new Error(parsed.detail || "workplace action failed"));
      }
      return;
    }

    if (parsed.event === "pe_updated") {
      for (const handler of this.peHandlers) handler(parsed.chat_id, parsed.active);
      return;
    }

    const chatId = (parsed as { chat_id?: string }).chat_id;
    if (chatId) this.dispatch(chatId, parsed);
  }

  private dispatch(chatId: string, ev: InboundEvent): void {
    const handlers = this.chatHandlers.get(chatId);
    if (handlers && handlers.size > 0) {
      for (const h of handlers) h(ev);
      return;
    }
    if (!this.isReplayable(ev)) return;
    this.appendToOfflineBuffer(chatId, ev);
  }

  /** Whether an inbound frame should be retained for subscribe-time replay. */
  private isReplayable(ev: InboundEvent): boolean {
    if (ev.event === "delta" || ev.event === "stream_end") return true;
    if (ev.event === "question_answer_ok" || ev.event === "workplace_updated") {
      return true;
    }
    if (ev.event === "message") {
      return ev.kind !== "tool_hint" && ev.kind !== "progress";
    }
    return false;
  }

  private isTurnComplete(ev: InboundEvent): boolean {
    return (
      ev.event === "message" &&
      ev.kind !== "tool_hint" &&
      ev.kind !== "progress"
    );
  }

  private appendToOfflineBuffer(chatId: string, ev: InboundEvent): void {
    let buf = this.offlineBuffers.get(chatId);
    if (!buf) {
      buf = [];
      this.offlineBuffers.set(chatId, buf);
    }

    if (ev.event === "delta") {
      const last = buf.at(-1);
      if (last && this.isTurnComplete(last)) {
        // ask_user mid-turn: stream_end(resuming:true) → message(questions) → deltas.
        // Do not drop the prefix when the next segment is still the same agent turn.
        const prev = buf.at(-2);
        const midTurnInFlight =
          prev?.event === "stream_end" &&
          (prev as Extract<InboundEvent, { event: "stream_end" }>).resuming ===
            true;
        if (!midTurnInFlight) {
          buf.length = 0;
        }
      }
      const deltaCount = buf.filter((entry) => entry.event === "delta").length;
      if (deltaCount >= MAX_OFFLINE_DELTAS) {
        this.coalesceOfflineDeltas(chatId, buf);
      }
    }

    buf.push(ev);
  }

  /** Merge buffered deltas into one frame so long Director streams stay bounded. */
  private coalesceOfflineDeltas(chatId: string, buf: InboundEvent[]): void {
    let mergedText = "";
    let index = 0;
    while (index < buf.length && buf[index]?.event === "delta") {
      const entry = buf[index] as Extract<InboundEvent, { event: "delta" }>;
      mergedText += entry.text;
      index += 1;
    }
    const tail = buf.slice(index);
    buf.length = 0;
    if (mergedText) {
      buf.push({ event: "delta", chat_id: chatId, text: mergedText });
    }
    buf.push(...tail);
  }

  private scheduleOfflineReplay(chatId: string): void {
    if (this.replayScheduled.has(chatId)) return;
    this.replayScheduled.add(chatId);
    queueMicrotask(() => {
      this.replayScheduled.delete(chatId);
      this.flushOfflineBuffer(chatId);
    });
  }

  /** Deliver buffered frames to every current subscriber, then drop the buffer. */
  private flushOfflineBuffer(chatId: string): void {
    const buf = this.offlineBuffers.get(chatId);
    if (!buf || buf.length === 0) return;
    const handlers = this.chatHandlers.get(chatId);
    if (!handlers || handlers.size === 0) return;

    const events = buf.splice(0, buf.length);
    this.offlineBuffers.delete(chatId);
    for (const ev of events) {
      for (const h of handlers) {
        h(ev);
      }
    }
  }

  private handleClose(event?: { code?: number }): void {
    this.socket = null;
    if (this.pendingNewChat) {
      clearTimeout(this.pendingNewChat.timer);
      this.pendingNewChat.reject(new Error("socket closed"));
      this.pendingNewChat = null;
    }
    this.rejectAllPendingWorkplaceActions("socket closed");
    // Surface structured reasons *before* reconnect logic so the UI can
    // display the error even while the client transparently reconnects.
    // Browsers populate ``CloseEvent.code`` with the wire-level close code;
    // 1009 = Message Too Big (server's max frame guard).
    if (event?.code === 1009) {
      this.emitError({ kind: "message_too_big" });
    }
    if (this.intentionallyClosed || !this.shouldReconnect) {
      this.setStatus("closed");
      return;
    }
    this.scheduleReconnect();
  }

  private emitError(error: StreamError): void {
    // Isolate subscribers so a throwing handler cannot abort the surrounding
    // ``handleClose`` flow (which still owes us a reconnect decision + status
    // update). We deliberately swallow here: error reporting is best-effort
    // and must never be allowed to compound the failure it's reporting.
    for (const handler of this.errorHandlers) {
      try {
        handler(error);
      } catch {
        // best-effort: subscriber fault must not stall transport bookkeeping
      }
    }
  }

  private scheduleReconnect(): void {
    this.setStatus("reconnecting");
    const attempt = this.reconnectAttempts++;
    // Exponential backoff: 0.5s, 1s, 2s, 4s, capped.
    const delay = Math.min(500 * 2 ** attempt, this.maxBackoffMs);
    this.reconnectTimer = setTimeout(async () => {
      this.reconnectTimer = null;
      if (this.options.onReauth) {
        try {
          const refreshed = await this.options.onReauth();
          if (refreshed) this.currentUrl = refreshed;
        } catch {
          // fall through to retry with current URL
        }
      }
      this.connect();
    }, delay);
  }

  private registerWorkplaceAction(
    timeoutMs: number = WORKPLACE_ACTION_TIMEOUT_MS,
  ): {
    requestId: string;
    promise: Promise<{ work_id: string; workplace: WorkplaceData }>;
  } {
    const requestId = crypto.randomUUID();
    const promise = new Promise<{ work_id: string; workplace: WorkplaceData }>(
      (resolve, reject) => {
        const timer = setTimeout(() => {
          this.pendingWorkplaceActions.delete(requestId);
          reject(new Error("workplace action timed out"));
        }, timeoutMs);
        this.pendingWorkplaceActions.set(requestId, { resolve, reject, timer });
      },
    );
    return { requestId, promise };
  }

  private rejectAllPendingWorkplaceActions(reason: string): void {
    for (const [, pending] of this.pendingWorkplaceActions) {
      clearTimeout(pending.timer);
      pending.reject(new Error(reason));
    }
    this.pendingWorkplaceActions.clear();
  }

  private queueSend(frame: Outbound): void {
    if (this.socket?.readyState === WS_OPEN) {
      this.rawSend(frame);
    } else {
      this.sendQueue.push(frame);
    }
  }

  private rawSend(frame: Outbound): void {
    if (!this.socket) return;
    try {
      this.socket.send(JSON.stringify(frame));
    } catch {
      // Send failure will materialize as a close; queue the frame for retry.
      this.sendQueue.push(frame);
    }
  }
}
