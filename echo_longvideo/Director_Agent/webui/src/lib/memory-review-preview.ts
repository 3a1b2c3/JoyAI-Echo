import type { MemoryReview, UIMessage } from "@/lib/types";

export function careerMemoryReviewPreview(
  status: MemoryReview["status"] = "awaiting_method",
  attempt = 1,
  selectionMode: MemoryReview["selection_mode"] = null,
  manualSelectedIds: string[] = [],
): UIMessage {
  const awaitingMethod = status === "awaiting_method";
  return {
    id: "memory-review:career-preview:1:shot-001",
    role: "assistant",
    content: awaitingMethod
      ? "Shot 1 is ready. Choose how its identity and scene memories should be selected before generating Shot 2."
      : "Shot 1 is ready. Review its identity and scene memories before generating Shot 2.",
    createdAt: Date.now(),
    memoryReview: {
      review_id: "career-shot-001",
      shot_id: 1,
      status,
      attempt,
      candidate_count: awaitingMethod ? 0 : 12,
      rejected_candidate_indices: attempt > 1 ? [2, 4, 9] : [],
      selection_mode: selectionMode,
      required_memory_ids: ["ID_A", "ID_B", "PREVIOUS_SHOT"],
      manual_selected_ids: manualSelectedIds,
      history: [],
      error: null,
      updated_at: new Date().toISOString(),
      source_video: {
        url: "/memory-review-preview/source.mp4",
        name: "source.mp4",
      },
      selections: awaitingMethod ? [] : [
        {
          memory_id: "ID_A",
          kind: "character",
          candidate_index: attempt > 1 ? 5 : 2,
          frame_index: attempt > 1 ? 111 : 51,
          timestamp_sec: attempt > 1 ? 4.44 : 2.059,
          confidence: 0.9,
          visual_status: "provisional",
          source_shot_id: 1,
          audio_source_shot_id: 1,
          reasoning:
            "清晰展示 ID_A 的深棕色齐肩头发、精致面部轮廓和炭灰色毛衣。当前画面仍包含 ID_B，因此作为 provisional memory。",
          image: { url: "/memory-review-preview/ID_A.jpg", name: "ID_A.jpg" },
          audio: { url: "/memory-review-preview/ID_A.wav", name: "ID_A.wav" },
        },
        {
          memory_id: "ID_B",
          kind: "character",
          candidate_index: 9,
          frame_index: 190,
          timestamp_sec: 7.581,
          confidence: 0.97,
          visual_status: "confirmed",
          source_shot_id: 1,
          audio_source_shot_id: 1,
          reasoning:
            "单人近景清晰展示 ID_B 的短黑发、浅胡茬、宽阔面部结构和深色夹克，可确认为 target-only memory。",
          image: { url: "/memory-review-preview/ID_B.jpg", name: "ID_B.jpg" },
          audio: { url: "/memory-review-preview/ID_B.wav", name: "ID_B.wav" },
        },
        {
          memory_id: "PREVIOUS_SHOT",
          kind: "previous_shot",
          candidate_index: 4,
          frame_index: 91,
          timestamp_sec: 3.637,
          confidence: 0.96,
          visual_status: "representative",
          source_shot_id: 1,
          audio_source_shot_id: 1,
          reasoning:
            "信息最完整的双人镜头：角色、方向盘互动、雨夜车内环境以及冷暖光线关系均清楚可见。",
          image: {
            url: "/memory-review-preview/PREVIOUS_SHOT.jpg",
            name: "PREVIOUS_SHOT.jpg",
          },
          audio: {
            url: "/memory-review-preview/PREVIOUS_SHOT.wav",
            name: "PREVIOUS_SHOT.wav",
          },
        },
      ],
    },
  };
}
