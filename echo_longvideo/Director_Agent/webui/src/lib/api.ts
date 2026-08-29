import type {
  ChatSummary,
  EchoTrackingResponse,
  WireMediaRef,
  WireQuestionCard,
  WorkplaceData,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

function utf8ToBase64(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

function nanobotBody(payload: Record<string, unknown>): HeadersInit {
  return { "X-Nanobot-Body": utf8ToBase64(JSON.stringify(payload)) };
}

async function request<T>(
  url: string,
  token: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(url, {
    ...(init ?? {}),
    headers: {
      ...(init?.headers ?? {}),
      Authorization: `Bearer ${token}`,
    },
    credentials: "same-origin",
  });
  if (!res.ok) {
    const detail = (await res.text()).trim();
    throw new ApiError(res.status, detail || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

async function workplaceRequest<T>(
  url: string,
  token: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(url, {
    ...(init ?? {}),
    headers: {
      ...(init?.headers ?? {}),
      Authorization: `Bearer ${token}`,
    },
    credentials: "same-origin",
  });
  if (!res.ok) {
    const detail = (await res.text()).trim();
    throw new ApiError(res.status, detail || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

function splitKey(key: string): { channel: string; chatId: string } {
  // Keys may include user-scoped prefixes, e.g. "websocket:<userId>:<chatId>".
  // The UI routes WS messages by the raw chatId, which is always the last segment.
  const first = key.indexOf(":");
  if (first === -1) return { channel: "", chatId: key };
  const last = key.lastIndexOf(":");
  return {
    channel: key.slice(0, first),
    chatId: key.slice(last + 1),
  };
}

export type WorkflowActionResponse = {
  ok: boolean;
  work_id: string;
  scheduled: boolean;
  workplace: WorkplaceData;
};

export interface SessionWireToolCall {
  id: string;
  type?: string;
  function?: { name?: string; arguments?: string };
}

export interface SessionWireMessage {
  role: string;
  content: string;
  timestamp?: string;
  tool_calls?: SessionWireToolCall[];
  tool_call_id?: string;
  name?: string;
  media_urls?: WireMediaRef[];
  questions?: WireQuestionCard[];
  question_batch_id?: string;
}

export async function listSessions(
  token: string,
  base: string = "",
): Promise<ChatSummary[]> {
  type Row = {
    key: string;
    created_at: string | null;
    updated_at: string | null;
    preview?: string;
    source?: "stepwise" | null;
    autoGenerate?: boolean | null;
  };
  const body = await request<{ sessions: Row[] }>(
    `${base}/api/sessions`,
    token,
  );
  return body.sessions.map((s) => ({
    key: s.key,
    ...splitKey(s.key),
    createdAt: s.created_at,
    updatedAt: s.updated_at,
    preview: s.preview ?? "",
    source: s.source ?? null,
    autoGenerate: Boolean(s.autoGenerate),
  }));
}

export async function fetchSessionMessages(
  token: string,
  key: string,
  base: string = "",
): Promise<{
  key: string;
  created_at: string | null;
  updated_at: string | null;
  messages: SessionWireMessage[];
}> {
  return request(
    `${base}/api/sessions/${encodeURIComponent(key)}/messages`,
    token,
  );
}

export async function deleteSession(
  token: string,
  key: string,
  base: string = "",
): Promise<boolean> {
  const body = await request<{ deleted: boolean }>(
    `${base}/api/sessions/${encodeURIComponent(key)}/delete`,
    token,
  );
  return body.deleted;
}

export type GenerationSettingsResponse = {
  ok: boolean;
  session_key: string;
  n_shots: number;
  duration_sec: number;
  width: number;
  height: number;
  language: string;
  temperature: number | null;
  top_p: number | null;
  top_k: number | null;
};

export type GenerationLlmSettings = Pick<
  GenerationSettingsResponse,
  "temperature" | "top_p" | "top_k"
>;

export async function fetchGenerationSettings(
  token: string,
  key: string,
  base: string = "",
): Promise<GenerationSettingsResponse> {
  return workplaceRequest(
    `${base}/api/sessions/${encodeURIComponent(key)}/generation-settings`,
    token,
  );
}

export async function saveGenerationSettings(
  token: string,
  key: string,
  durationSec?: number,
  width?: number,
  height?: number,
  language?: string,
  base: string = "",
): Promise<GenerationSettingsResponse> {
  const params = new URLSearchParams({
    ...(durationSec === undefined ? {} : { duration_sec: String(durationSec) }),
    ...(width === undefined ? {} : { width: String(width) }),
    ...(height === undefined ? {} : { height: String(height) }),
    ...(language === undefined ? {} : { language }),
  });
  return workplaceRequest(
    `${base}/api/sessions/${encodeURIComponent(key)}/generation-settings/save?${params}`,
    token,
  );
}

/** Keep the first-frame image self-contained in the local application. */
export async function uploadFirstFrameImage(
  file: File | Blob,
  name = "first-frame.jpg",
): Promise<{ url: string; width: number; height: number }> {
  const payload =
    file instanceof File
      ? file
      : new File([file], name, { type: file.type || "image/jpeg" });
  const url = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("failed to read first-frame image"));
    reader.readAsDataURL(payload);
  });
  if (!url.startsWith("data:image/")) {
    throw new Error("failed to encode first-frame image");
  }
  return { url, width: 0, height: 0 };
}

export async function fetchWorkplace(
  token: string,
  key: string,
  base: string = "",
): Promise<WorkplaceData> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}`,
    token,
  );
}

export async function updateEchoLike(
  token: string,
  key: string,
  likeStatus: 1 | 2,
  base: string = "",
): Promise<EchoTrackingResponse> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/echo/like?like_status=${likeStatus}`,
    token,
  );
}

export async function recordEchoDownloadPrompt(
  token: string,
  key: string,
  base: string = "",
): Promise<EchoTrackingResponse> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/echo/download-prompt`,
    token,
  );
}

export async function acceptShot(
  token: string,
  key: string,
  shotId: number,
  base: string = "",
): Promise<{ workplace: WorkplaceData }> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/${shotId}/accept`,
    token,
  );
}

export type MemoryReviewAction = {
  review_id: string;
  attempt: number;
  memory_id?: string;
  timestamp_sec?: number;
  retained_memory_ids?: string[];
};

export async function approveMemoryReview(
  token: string,
  key: string,
  shotId: number,
  action: MemoryReviewAction,
  base: string = "",
): Promise<{ workplace: WorkplaceData }> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/${shotId}/memory-review/approve`,
    token,
    { headers: nanobotBody(action) },
  );
}

export async function reselectMemoryReview(
  token: string,
  key: string,
  shotId: number,
  action: MemoryReviewAction,
  base: string = "",
): Promise<{ workplace: WorkplaceData }> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/${shotId}/memory-review/reselect`,
    token,
    { headers: nanobotBody(action) },
  );
}

export async function selectMemoryReviewFrame(
  token: string,
  key: string,
  shotId: number,
  action: MemoryReviewAction,
  base: string = "",
): Promise<{ workplace: WorkplaceData }> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/${shotId}/memory-review/manual-select`,
    token,
    { headers: nanobotBody(action) },
  );
}

export async function selectMemoryReviewMode(
  token: string,
  key: string,
  shotId: number,
  action: MemoryReviewAction & { selection_mode: "manual" | "vlm" },
  base: string = "",
): Promise<{ workplace: WorkplaceData }> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/${shotId}/memory-review/select-mode`,
    token,
    { headers: nanobotBody(action) },
  );
}

export async function reviseShot(
  token: string,
  key: string,
  shotId: number,
  feedback: string,
  base: string = "",
): Promise<{ workplace: WorkplaceData }> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/${shotId}/revise?feedback=${encodeURIComponent(feedback)}`,
    token,
  );
}

export async function confirmStory(
  token: string,
  key: string,
  storyMd?: string,
  base: string = "",
): Promise<WorkflowActionResponse & { action: "confirm_story" }> {
  const trimmed = storyMd?.trim();
  const init: RequestInit | undefined = trimmed
    ? { headers: nanobotBody({ story_md: trimmed }) }
    : undefined;
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/workflow/confirm-story`,
    token,
    init,
  );
}

export async function startGeneration(
  token: string,
  key: string,
  base: string = "",
): Promise<WorkflowActionResponse & { action: "start_generation" }> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/workflow/start-generation`,
    token,
  );
}

export async function startAutoGenerate(
  token: string,
  key: string,
  base: string = "",
): Promise<WorkflowActionResponse & { action: "auto_generate" }> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/workflow/auto-generate`,
    token,
  );
}

export type ReferenceImagePayload = {
  url: string;
  name?: string;
  width?: number;
  height?: number;
};

export async function putReferenceImage(
  token: string,
  key: string,
  image: ReferenceImagePayload,
  base: string = "",
): Promise<{ ok?: boolean; workplace?: WorkplaceData }> {
  // websockets HTTP 只接受 GET；动作写在 path 上，body 走 X-Nanobot-Body。
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/reference-image/save`,
    token,
    {
      headers: nanobotBody({
        url: image.url,
        name: image.name ?? "",
        width: image.width ?? 0,
        height: image.height ?? 0,
      }),
    },
  );
}

export async function deleteReferenceImage(
  token: string,
  key: string,
  base: string = "",
): Promise<{ ok?: boolean; workplace?: WorkplaceData }> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/reference-image/delete`,
    token,
  );
}

export async function abortGeneration(
  token: string,
  key: string,
  base: string = "",
): Promise<WorkflowActionResponse & { action: "abort_generation" }> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/workflow/abort-generation`,
    token,
  );
}

export async function generateAll(
  token: string,
  key: string,
  base: string = "",
): Promise<{
  ok: boolean;
  action: "generate_all";
  work_id: string;
  submitted_shot_ids: number[];
  workplace: WorkplaceData;
}> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/workflow/generate-all`,
    token,
  );
}

export async function acceptAllShots(
  token: string,
  key: string,
  base: string = "",
): Promise<{
  ok: boolean;
  work_id: string;
  accepted_shot_ids: number[];
  workplace: WorkplaceData;
}> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/accept-all`,
    token,
  );
}

const ACCEPTABLE_SHOT_STATUSES = new Set(["generated", "review_pass"]);

export function mockAcceptAllShots(workplace: WorkplaceData): {
  ok: boolean;
  work_id: string;
  accepted_shot_ids: number[];
  workplace: WorkplaceData;
} {
  const acceptedShotIds: number[] = [];
  const newShots = (workplace.shots ?? []).map((shot) => {
    if (!ACCEPTABLE_SHOT_STATUSES.has(shot.status)) {
      return shot;
    }
    acceptedShotIds.push(shot.shot_id);
    return {
      ...shot,
      status: "approved",
      accepted: true,
      last_review: "accepted",
      review_notes: "",
    };
  });

  return {
    ok: true,
    work_id: workplace.work_id ?? "",
    accepted_shot_ids: acceptedShotIds,
    workplace: {
      ...workplace,
      shots: newShots,
    },
  };
}

export type GenerateShotReferenceImage = {
  url: string;
  name?: string;
  width?: number;
  height?: number;
};

export async function generateShot(
  token: string,
  key: string,
  shotId: number,
  referenceImage?: GenerateShotReferenceImage | null,
  base: string = "",
): Promise<{
  ok: boolean;
  action: "generate_shot";
  work_id: string;
  shot_id: number;
  workplace: WorkplaceData;
}> {
  const init: RequestInit | undefined = referenceImage?.url
    ? {
        headers: nanobotBody({
          reference_image_url: referenceImage.url,
          reference_image_name: referenceImage.name ?? "",
          reference_image_width: referenceImage.width ?? 0,
          reference_image_height: referenceImage.height ?? 0,
        }),
      }
    : undefined;
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/${shotId}/generate`,
    token,
    init,
  );
}

/** 设置 Shot 首尾衔接模式 */
export async function setShotContinuousMode(
  token: string,
  key: string,
  shotId: number,
  enabled: boolean,
  base: string = "",
): Promise<{
  ok: boolean;
  action: string;
  workplace: WorkplaceData;
}> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/${shotId}/continuous-mode?enabled=${enabled}`,
    token,
  );
}

/** 使用尾帧连续生成（I2V） */
export async function continuousGenerateShot(
  token: string,
  key: string,
  shotId: number,
  base: string = "",
): Promise<{
  ok: boolean;
  action: "continuous_generate";
  work_id: string;
  shot_id: number;
  workplace: WorkplaceData;
}> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/${shotId}/continuous-generate`,
    token,
  );
}

export async function updateShotDuration(
  token: string,
  key: string,
  shotId: number,
  durationSec: number,
  base: string = "",
): Promise<{
  ok: boolean;
  work_id: string;
  shot_id: number;
  duration_sec: number;
  workplace: WorkplaceData;
}> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/${shotId}/duration?duration_sec=${durationSec}`,
    token,
  );
}

/** Placeholder until backend implements shots/{id}/save. */
export async function saveShotPrompt(
  token: string,
  key: string,
  shotId: number,
  summary: string,
  base: string = "",
): Promise<{ ok: boolean; workplace: WorkplaceData }> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/${shotId}/save?summary=${encodeURIComponent(summary)}`,
    token,
  );
}

export async function startMerge(
  token: string,
  key: string,
  base: string = "",
): Promise<WorkflowActionResponse & { action: "start_merge" }> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/workflow/start-merge`,
    token,
  );
}

export async function regenerate(
  token: string,
  key: string,
  base: string = "",
): Promise<{
  ok: boolean;
  action: "regenerate";
  work_id: string;
  workplace: WorkplaceData;
}> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/workflow/regenerate`,
    token,
  );
}

export type SplitShotPayload =
  | { cursor_pos: number }
  | { before_text: string; after_text: string };

export async function splitShot(
  token: string,
  key: string,
  shotId: number,
  payload: SplitShotPayload,
  base: string = "",
): Promise<{
  ok: boolean;
  work_id: string;
  split_shot_id: number;
  new_shot_id: number;
  workplace: WorkplaceData;
}> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/${shotId}/split-shot`,
    token,
    { headers: nanobotBody(payload) },
  );
}

export async function mergeShotUp(
  token: string,
  key: string,
  shotId: number,
  mergedText?: string,
  base: string = "",
): Promise<{
  ok: boolean;
  work_id: string;
  merged_shot_id: number;
  into_shot_id: number;
  workplace: WorkplaceData;
}> {
  const trimmed = mergedText?.trim();
  const init: RequestInit | undefined = trimmed
    ? { headers: nanobotBody({ merged_text: trimmed }) }
    : undefined;
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/${shotId}/merge-up`,
    token,
    init,
  );
}

export async function deleteShot(
  token: string,
  key: string,
  shotId: number,
  base: string = "",
): Promise<{
  ok: boolean;
  work_id: string;
  removed_shot_id: number;
  workplace: WorkplaceData;
}> {
  return workplaceRequest(
    `${base}/api/workplace/${encodeURIComponent(key)}/shots/${shotId}/remove-shot`,
    token,
  );
}

export function mockDeleteShot(
  workplace: WorkplaceData,
  shotId: number,
): {
  ok: boolean;
  work_id: string;
  removed_shot_id: number;
  workplace: WorkplaceData;
} {
  const beats = workplace.story_profile?.beats ?? [];
  if (beats.length <= 1) {
    throw new ApiError(400, "cannot delete the last beat");
  }

  const deleteIndex = beats.findIndex(
    (beat) => Number(beat?.shot_id) === shotId,
  );
  if (deleteIndex < 0) {
    throw new ApiError(404, "shot not found");
  }

  const remainingBeats = beats.filter((_, index) => index !== deleteIndex);
  const normalizedBeats = remainingBeats.map((beat, index) => ({
    shot_id: index + 1,
    summary: String(beat?.summary ?? "").trim(),
  }));

  const sourceShots = workplace.shots ?? [];
  const newShots = normalizedBeats.map((beat, index) => {
    const sourceIndex = index >= deleteIndex ? index + 1 : index;
    const sourceShot = sourceShots[sourceIndex];
    const shotKey = `shot_${String(beat.shot_id).padStart(3, "0")}`;
    if (sourceShot) {
      return {
        ...sourceShot,
        shot_id: beat.shot_id,
        shot_key: shotKey,
        summary: beat.summary,
      };
    }
    return {
      shot_id: beat.shot_id,
      shot_key: shotKey,
      status: "planned",
      summary: beat.summary,
      cut: true,
      video: null,
      has_video: false,
      has_actions: false,
      accepted: false,
      last_review: null,
      review_notes: "",
      updated_at: workplace.updated_at ?? null,
      timeline: {
        start_seconds: 0,
        end_seconds: 4,
        duration_seconds: 4,
        label: "00:00 - 00:04",
      },
    };
  });

  return {
    ok: true,
    work_id: workplace.work_id ?? "",
    removed_shot_id: shotId,
    workplace: {
      ...workplace,
      story_profile: workplace.story_profile
        ? { ...workplace.story_profile, beats: normalizedBeats }
        : { beats: normalizedBeats },
      shots: newShots,
    },
  };
}
