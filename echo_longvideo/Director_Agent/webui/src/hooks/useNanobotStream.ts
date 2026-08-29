import { useCallback, useEffect, useRef, useState } from "react";

import { splitMediaByKind, wireMediaRefs } from "@/lib/media";
import {
  stripAskUserFollowupBlurbs,
  wireQuestionCards,
} from "@/lib/questions";
import { useClient } from "@/providers/ClientProvider";
import type { StreamError } from "@/lib/nanobot-client";
import { randomId } from "@/lib/utils";
import type {
  InboundEvent,
  OutboundMedia,
  UIImage,
  UIMessage,
} from "@/lib/types";

interface StreamBuffer {
  /** ID of the assistant message currently receiving deltas. */
  messageId: string;
  /** Sequence of deltas accumulated in order. */
  parts: string[];
}

/** Locate the empty top-of-turn wait bubble, if any. */
function findTurnAnchorId(list: UIMessage[]): string | null {
  const anchor = list.find(
    (m) =>
      m.turnWaiting &&
      m.isStreaming &&
      !m.content.trim() &&
      !m.questions?.length,
  );
  return anchor?.id ?? null;
}

/**
 * Subscribe to a chat by ID. Returns the in-memory message list for the chat,
 * a streaming flag, and a ``send`` function. Initial history must be seeded
 * separately (e.g. via ``fetchSessionMessages``) since the server only replays
 * live events.
 */
/** Payload passed to ``send`` when the user attaches one or more images.
 *
 * ``media`` is handed to the wire client verbatim; ``preview`` powers the
 * optimistic user bubble (blob URLs so the preview appears before the server
 * acks the frame). Keeping the two separate lets the bubble re-use the local
 * blob URL even after the server persists the file under a different name. */
export interface SendImage {
  /** Omit for a UI-only preview whose payload is carried in message extras. */
  media?: OutboundMedia;
  preview: UIImage;
}

export function useNanobotStream(
  chatId: string | null,
  initialMessages: UIMessage[] = [],
  options?: { onReplyEnd?: () => void },
): {
  messages: UIMessage[];
  isStreaming: boolean;
  /** ``true`` after ``stream_end`` with ``resuming: false`` (or a non-resuming reply). */
  turnComplete: boolean;
  send: (
    content: string,
    images?: SendImage[],
    extras?: {
      autoGenerate?: boolean;
      duration_sec?: number;
      reference_image_url?: string;
      reference_image_name?: string;
      reference_image_width?: number;
      reference_image_height?: number;
    },
  ) => void;
  answerQuestion: (messageId: string, cardId: string, value: string) => void;
  setMessages: React.Dispatch<React.SetStateAction<UIMessage[]>>;
  /** Latest transport-level fault raised since the last ``dismissStreamError``.
   * ``null`` when there is nothing to show. */
  streamError: StreamError | null;
  /** Clear the current ``streamError`` (e.g. after the user dismisses the
   * notification or starts a fresh action). */
  dismissStreamError: () => void;
} {
  const { client } = useClient();
  const messagesRef = useRef<UIMessage[]>(initialMessages);
  const [messages, setMessages] = useState<UIMessage[]>(initialMessages);
  messagesRef.current = messages;
  const [isStreaming, setIsStreaming] = useState(false);
  const [turnComplete, setTurnComplete] = useState(true);
  const turnCompleteRef = useRef(true);
  turnCompleteRef.current = turnComplete;
  /** Set when an intermediate ``stream_end`` arrives with ``resuming: true``. */
  const sawResumingStreamEndRef = useRef(false);
  /** Empty ``turnWaiting`` bubble pinned at the top of the in-flight agent turn. */
  const turnAnchorIdRef = useRef<string | null>(null);
  const [streamError, setStreamError] = useState<StreamError | null>(null);
  const buffer = useRef<StreamBuffer | null>(null);
  const onReplyEndRef = useRef(options?.onReplyEnd);
  onReplyEndRef.current = options?.onReplyEnd;

  useEffect(() => {
    return client.onError((err) => setStreamError(err));
  }, [client]);

  const dismissStreamError = useCallback(() => setStreamError(null), []);

  useEffect(() => {
    if (!chatId) {
      buffer.current = null;
      setIsStreaming(false);
      setTurnComplete(true);
      sawResumingStreamEndRef.current = false;
      turnAnchorIdRef.current = null;
      return;
    }

    // Seed the target chat before onChat: offline replay is scheduled in a
    // microtask and must not read the previous session's messagesRef.
    // Seed messagesRef before subscribe; defer React state until after offline
    // replay microtasks so a switch-back does not wipe replayed deltas.
    messagesRef.current = initialMessages;
    buffer.current = null;
    const restoredAnchorId = findTurnAnchorId(initialMessages);
    turnAnchorIdRef.current = restoredAnchorId;
    const openTurn = restoredAnchorId !== null;
    setIsStreaming(openTurn);
    setTurnComplete(!openTurn);
    sawResumingStreamEndRef.current = false;
    setStreamError(null);

    const stripTurnAnchor = (list: UIMessage[]): UIMessage[] => {
      turnAnchorIdRef.current = null;
      return list.filter(
        (m) => !(m.turnWaiting && !m.content.trim() && !m.questions?.length),
      );
    };

    const finalizeTurnEnd = (list: UIMessage[]): UIMessage[] => {
      return stripAskUserFollowupBlurbs(
        stripTurnAnchor(
          list
            .map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m))
            .filter(
              (m) =>
                !(
                  m.role === "assistant" &&
                  !m.content.trim() &&
                  !m.isStreaming &&
                  !m.questions?.length
                ),
            ),
        ),
      );
    };

    const commitMessages = (
      updater: UIMessage[] | ((prev: UIMessage[]) => UIMessage[]),
    ) => {
      const next =
        typeof updater === "function" ? updater(messagesRef.current) : updater;
      messagesRef.current = next;
      setMessages(next);
    };

    /** Bind or create the single top-of-turn wait bubble (TypingDots). */
    const bindWaitBuffer = (): string => {
      if (buffer.current) return buffer.current.messageId;
      const anchorId = turnAnchorIdRef.current;
      if (anchorId) {
        const anchor = messagesRef.current.find((m) => m.id === anchorId);
        if (anchor?.isStreaming && !anchor.content.trim()) {
          buffer.current = { messageId: anchor.id, parts: [] };
          setIsStreaming(true);
          return anchor.id;
        }
      }
      const last = messagesRef.current.at(-1);
      if (
        last?.role === "assistant" &&
        last.isStreaming &&
        !last.content.trim()
      ) {
        if (last.turnWaiting) {
          turnAnchorIdRef.current = last.id;
        }
        buffer.current = { messageId: last.id, parts: [] };
        setIsStreaming(true);
        return last.id;
      }
      const id = randomId();
      turnAnchorIdRef.current = id;
      const placeholder: UIMessage = {
        id,
        role: "assistant",
        content: "",
        isStreaming: true,
        turnWaiting: true,
        createdAt: Date.now(),
      };
      buffer.current = { messageId: id, parts: [] };
      setIsStreaming(true);
      const next = [...messagesRef.current, placeholder];
      messagesRef.current = next;
      commitMessages(next);
      return id;
    };

    /** Tail segment for streamed deltas — never reuses the top turn anchor. */
    const beginDeltaStream = (): string => {
      if (buffer.current) return buffer.current.messageId;
      const anchorId = turnAnchorIdRef.current;
      const last = messagesRef.current.at(-1);
      if (
        last?.role === "assistant" &&
        last.isStreaming &&
        !last.content.trim() &&
        last.id !== anchorId &&
        !last.turnWaiting
      ) {
        buffer.current = { messageId: last.id, parts: [] };
        setIsStreaming(true);
        return last.id;
      }
      const id = randomId();
      const placeholder: UIMessage = {
        id,
        role: "assistant",
        content: "",
        isStreaming: true,
        createdAt: Date.now(),
      };
      buffer.current = { messageId: id, parts: [] };
      setIsStreaming(true);
      const next = [...messagesRef.current, placeholder];
      messagesRef.current = next;
      commitMessages(next);
      return id;
    };

    const beginStreamingTurn = bindWaitBuffer;

    const handle = (ev: InboundEvent) => {
      if (ev.event === "delta") {
        const anchorId = turnAnchorIdRef.current;
        const bufferTargetsAnchor = (): boolean => {
          const id = buffer.current?.messageId;
          if (!id) return false;
          if (anchorId && id === anchorId) return true;
          return (
            messagesRef.current.find((m) => m.id === id)?.turnWaiting === true
          );
        };
        if (!buffer.current || bufferTargetsAnchor()) {
          if (buffer.current && bufferTargetsAnchor()) {
            buffer.current = null;
          }
          beginDeltaStream();
        }
        buffer.current!.parts.push(ev.text);
        const combined = buffer.current!.parts.join("");
        const targetId = buffer.current!.messageId;
        commitMessages(
          messagesRef.current.map((m) =>
            m.id === targetId ? { ...m, content: combined } : m,
          ),
        );
        return;
      }

      if (ev.event === "stream_end") {
        if (ev.resuming === true) {
          sawResumingStreamEndRef.current = true;
          setTurnComplete(false);
        } else if (ev.resuming === false) {
          sawResumingStreamEndRef.current = false;
          setTurnComplete(true);
        }

        if (!buffer.current) {
          if (ev.resuming === false) {
            setIsStreaming(false);
            commitMessages((prev) => finalizeTurnEnd(prev));
            onReplyEndRef.current?.();
            return;
          }
          // Director may lead with stream_end on replay; keep the wait UI alive.
          beginStreamingTurn();
          return;
        }
        const finalId = buffer.current.messageId;
        const combined = buffer.current.parts.join("");
        const hadContent = combined.trim().length > 0;
        if (!hadContent) {
          if (ev.resuming === false) {
            buffer.current = null;
            setIsStreaming(false);
            commitMessages((prev) => finalizeTurnEnd(prev));
            onReplyEndRef.current?.();
          }
          return;
        }

        if (ev.resuming === true) {
          buffer.current = null;
          setIsStreaming(true);
          commitMessages((prev) =>
            prev.map((m) =>
              m.id === finalId
                ? { ...m, content: combined, isStreaming: false }
                : m,
            ),
          );
          return;
        }

        buffer.current = null;
        setIsStreaming(false);
        if (ev.resuming === undefined) {
          sawResumingStreamEndRef.current = false;
          setTurnComplete(true);
        }
        commitMessages((prev) => {
          let next = prev.map((m) =>
            m.id === finalId
              ? { ...m, content: combined, isStreaming: false }
              : m,
          );
          if (ev.resuming === false) {
            next = stripTurnAnchor(next);
          }
          return next;
        });
        onReplyEndRef.current?.();
        return;
      }

      if (ev.event === "message") {
        // Tool hints / progress are not surfaced in the thread UI; the send
        // placeholder progress bar covers the wait instead.
        if (ev.kind === "tool_hint" || ev.kind === "progress") {
          return;
        }

        // A complete (non-streamed) assistant message. If a stream was in
        // flight, drop the placeholder so we don't render the text twice.
        const midTurn = sawResumingStreamEndRef.current;
        const activeId = buffer.current?.messageId;
        const anchorId = turnAnchorIdRef.current;
        buffer.current = null;
        if (!midTurn) {
          setIsStreaming(false);
          setTurnComplete(true);
          turnAnchorIdRef.current = null;
        } else {
          setIsStreaming(true);
        }
        commitMessages((prev) => {
          const shouldRemoveActive =
            !!activeId && (!midTurn || activeId !== anchorId);
          const base = shouldRemoveActive
            ? prev.filter((m) => m.id !== activeId)
            : prev;
          const media = splitMediaByKind(
            wireMediaRefs(ev.media_urls, ev.media),
          );
          const questions =
            ev.questions && ev.questions.length > 0
              ? wireQuestionCards(ev.questions)
              : undefined;

          const attachCards = (list: UIMessage[]): UIMessage[] => {
            if (!questions?.length) {
              const assistantMsg: UIMessage = {
                id: randomId(),
                role: "assistant",
                content: ev.text,
                createdAt: Date.now(),
                ...(media.images ? { images: media.images } : {}),
                ...(media.videos ? { videos: media.videos } : {}),
              };
              return [...list, assistantMsg];
            }

            const cardMessage = (): UIMessage => ({
              id: randomId(),
              role: "assistant",
              content: "",
              createdAt: Date.now(),
              ...(media.images ? { images: media.images } : {}),
              ...(media.videos ? { videos: media.videos } : {}),
              questions,
              ...(ev.question_batch_id
                ? { questionBatchId: ev.question_batch_id }
                : {}),
            });

            const last = list.at(-1);
            if (last?.role === "assistant") {
              if (
                last.questions?.length &&
                ev.question_batch_id &&
                last.questionBatchId === ev.question_batch_id
              ) {
                return list;
              }
              if (
                last.questions?.length &&
                ev.question_batch_id &&
                last.questionBatchId !== ev.question_batch_id
              ) {
                return stripAskUserFollowupBlurbs([...list, cardMessage()]);
              }
              if (
                last.turnWaiting ||
                (last.isStreaming &&
                  !last.content.trim() &&
                  !last.questions?.length)
              ) {
                return stripAskUserFollowupBlurbs([
                  ...list.slice(0, -1),
                  {
                    ...last,
                    content: "",
                    isStreaming: false,
                    turnWaiting: undefined,
                    questions,
                    ...(ev.question_batch_id
                      ? { questionBatchId: ev.question_batch_id }
                      : {}),
                  },
                ]);
              }
              return stripAskUserFollowupBlurbs([
                ...list.slice(0, -1),
                {
                  ...last,
                  content: "",
                  isStreaming: false,
                  turnWaiting: undefined,
                  questions,
                  ...(ev.question_batch_id
                    ? { questionBatchId: ev.question_batch_id }
                    : {}),
                },
              ]);
            }

            return stripAskUserFollowupBlurbs([...list, cardMessage()]);
          };

          const resolvedAnchorId = anchorId ?? findTurnAnchorId(base);
          if (midTurn && resolvedAnchorId) {
            turnAnchorIdRef.current = resolvedAnchorId;
            const anchorIdx = base.findIndex((m) => m.id === resolvedAnchorId);
            const next = [...base];
            if (anchorIdx >= 0) {
              const prefix = next.slice(0, anchorIdx + 1);
              const suffix = next.slice(anchorIdx + 1);
              return [...prefix, ...attachCards(suffix)];
            }
            return attachCards(next);
          }

          return attachCards(stripTurnAnchor(base));
        });
        if (!midTurn) {
          onReplyEndRef.current?.();
        }
        return;
      }

      if (ev.event === "question_answer_ok") {
        commitMessages((prev) =>
          prev.map((m) => {
            if (m.questionBatchId !== ev.question_batch_id || !m.questions) {
              return m;
            }
            return {
              ...m,
              questions: m.questions.map((q) =>
                q.id === ev.card_id ? { ...q, answered: ev.value } : q,
              ),
            };
          }),
        );
        return;
      }
      // ``attached`` / ``error`` frames aren't actionable here; the client
      // shell handles them separately.
    };

    const pendingReplay =
      typeof client.hasOfflineBuffer === "function"
        ? client.hasOfflineBuffer(chatId)
        : false;
    const unsub = client.onChat(chatId, handle);
    if (pendingReplay && typeof client.replayOfflineNow === "function") {
      client.replayOfflineNow(chatId);
    }
    setMessages(messagesRef.current);
    return () => {
      unsub();
      buffer.current = null;
    };
    // Only re-subscribe on chat switch; history HTTP refresh is merged via ThreadShell.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId, client]);

  const send = useCallback(
    (
      content: string,
      images?: SendImage[],
      extras?: {
        autoGenerate?: boolean;
        duration_sec?: number;
        reference_image_url?: string;
        reference_image_name?: string;
        reference_image_width?: number;
        reference_image_height?: number;
      },
    ) => {
      if (!chatId) return;
      const hasImages = !!images && images.length > 0;
      // Text is optional when images are attached — the agent will still see
      // the image blocks via ``media`` paths.
      if (!hasImages && !content.trim()) return;

      const pendingId = randomId();
      turnAnchorIdRef.current = pendingId;
      buffer.current = null;
      sawResumingStreamEndRef.current = false;
      setTurnComplete(false);
      setIsStreaming(true);
      const previews = hasImages ? images!.map((i) => i.preview) : undefined;
      setMessages((prev) => [
        ...prev,
        {
          id: randomId(),
          role: "user",
          content,
          createdAt: Date.now(),
          ...(previews ? { images: previews } : {}),
        },
        {
          id: pendingId,
          role: "assistant",
          content: "",
          isStreaming: true,
          turnWaiting: true,
          createdAt: Date.now(),
        },
      ]);
      const wireMedia = hasImages
        ? images!
            .map((i) => i.media)
            .filter((media): media is OutboundMedia => media !== undefined)
        : undefined;
      const mergedExtras = {
        ...(extras ?? {}),
      };
      client.sendMessage(
        chatId,
        content,
        wireMedia && wireMedia.length > 0 ? wireMedia : undefined,
        Object.keys(mergedExtras).length > 0
          ? (mergedExtras as {
              temperature?: number;
              top_p?: number;
              top_k?: number;
              autoGenerate?: boolean;
              duration_sec?: number;
              reference_image_url?: string;
              reference_image_name?: string;
              reference_image_width?: number;
              reference_image_height?: number;
            })
          : undefined,
      );
    },
    [chatId, client],
  );

  const answerQuestion = useCallback(
    (messageId: string, cardId: string, value: string) => {
      if (!chatId) return;
      if (!turnCompleteRef.current) return;
      const trimmed = value.trim();
      if (!trimmed) return;

      const target = messagesRef.current.find((m) => m.id === messageId);
      const batchId = target?.questionBatchId;
      const pendingId = randomId();
      turnAnchorIdRef.current = pendingId;
      buffer.current = null;
      sawResumingStreamEndRef.current = false;
      setTurnComplete(false);
      setIsStreaming(true);

      setMessages((prev) => {
        const withAnswer = prev.map((m) => {
          if (m.id !== messageId || !m.questions) return m;
          return {
            ...m,
            questions: m.questions.map((q) =>
              q.id === cardId ? { ...q, answered: trimmed } : q,
            ),
          };
        });
        return [
          ...withAnswer,
          {
            id: randomId(),
            role: "user",
            content: trimmed,
            createdAt: Date.now(),
          },
          {
            id: pendingId,
            role: "assistant",
            content: "",
            isStreaming: true,
            turnWaiting: true,
            createdAt: Date.now(),
          },
        ];
      });

      if (batchId) {
        client.answerQuestion(chatId, batchId, cardId, trimmed);
      }
      client.sendMessage(chatId, trimmed);
    },
    [chatId, client],
  );

  return {
    messages,
    isStreaming,
    turnComplete,
    send,
    answerQuestion,
    setMessages,
    streamError,
    dismissStreamError,
  };
}
