import type {
  MemoryReview,
  MemoryReviewStatus,
  MemorySelection,
  UIMessage,
  WorkplaceData,
} from "@/lib/types";

const STATUSES = new Set<MemoryReviewStatus>([
  "awaiting_method",
  "selecting",
  "awaiting_review",
  "reselecting",
  "approved",
  "error",
]);

function isSelection(value: unknown): value is MemorySelection {
  if (!value || typeof value !== "object") return false;
  const item = value as Partial<MemorySelection>;
  return (
    typeof item.memory_id === "string" &&
    (item.kind === "character" || item.kind === "previous_shot") &&
    typeof item.candidate_index === "number" &&
    typeof item.frame_index === "number" &&
    typeof item.timestamp_sec === "number" &&
    typeof item.confidence === "number" &&
    typeof item.reasoning === "string" &&
    Boolean(item.image && typeof item.image.url === "string")
  );
}

function normalizeReview(
  shotId: number,
  value: unknown,
): MemoryReview | null {
  if (!value || typeof value !== "object") return null;
  const review = value as Partial<MemoryReview>;
  if (
    typeof review.review_id !== "string" ||
    !review.review_id ||
    typeof review.status !== "string" ||
    !STATUSES.has(review.status as MemoryReviewStatus) ||
    typeof review.attempt !== "number" ||
    !Array.isArray(review.selections) ||
    !review.selections.every(isSelection)
  ) {
    return null;
  }
  return {
    review_id: review.review_id,
    shot_id: shotId,
    status: review.status as MemoryReviewStatus,
    attempt: review.attempt,
    candidate_count: Number(review.candidate_count ?? 0),
    rejected_candidate_indices: Array.isArray(
      review.rejected_candidate_indices,
    )
      ? review.rejected_candidate_indices.filter(
          (item): item is number => typeof item === "number",
        )
      : [],
    selections: review.selections,
    selection_mode:
      review.selection_mode === "manual" || review.selection_mode === "vlm"
        ? review.selection_mode
        : null,
    required_memory_ids: Array.isArray(review.required_memory_ids)
      ? review.required_memory_ids.filter((item): item is string => typeof item === "string")
      : [],
    manual_selected_ids: Array.isArray(review.manual_selected_ids)
      ? review.manual_selected_ids.filter((item): item is string => typeof item === "string")
      : [],
    source_video:
      review.source_video && typeof review.source_video.url === "string"
        ? review.source_video
        : null,
    history: Array.isArray(review.history) ? review.history : [],
    error: typeof review.error === "string" ? review.error : null,
    updated_at:
      typeof review.updated_at === "string" ? review.updated_at : "",
  };
}

function pendingSelectingReview(shotId: number): MemoryReview {
  return {
    review_id: `pending:${shotId}`,
    shot_id: shotId,
    status: "selecting",
    attempt: 1,
    candidate_count: 0,
    rejected_candidate_indices: [],
    selections: [],
    history: [],
    error: null,
    updated_at: "",
  };
}

function toMessage(
  workplace: WorkplaceData,
  shotId: number,
  review: MemoryReview,
): UIMessage {
  const parsedTime = Date.parse(review.updated_at);
  return {
    id: `memory-review:${workplace.work_id}:${shotId}:${review.review_id}`,
    role: "assistant",
    content: "",
    createdAt: Number.isFinite(parsedTime) ? parsedTime : shotId,
    memoryReview: review,
  };
}

export function memoryReviewMessages(workplace: WorkplaceData): UIMessage[] {
  if (!workplace.work_id) return [];
  return [...(workplace.shots ?? [])]
    .sort((left, right) => left.shot_id - right.shot_id)
    .flatMap((shot) => {
      // Only surface Memory review after the user accepts the shot video.
      if (shot.status !== "approved") return [];
      const review = normalizeReview(shot.shot_id, shot.memory_review);
      if (
        workplace.auto_generate &&
        review?.status !== "awaiting_method" &&
        review?.status !== "error" &&
        review?.selection_mode !== "manual"
      ) {
        return [];
      }
      if (review) return [toMessage(workplace, shot.shot_id, review)];
      if (workplace.auto_generate) return [];
      return [
        toMessage(workplace, shot.shot_id, pendingSelectingReview(shot.shot_id)),
      ];
    });
}
