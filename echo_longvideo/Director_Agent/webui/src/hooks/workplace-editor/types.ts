import type { StoryProfile, WorkplaceData } from "@/lib/types";

/** Canonical overlay keys: `story` or `beat:{shotId}`. */
export type FieldPath = "story" | `beat:${number}`;

export type LeaseMode = "follow" | "held";

export type EditorSegment = {
  id: string;
  shotId: number;
  text: string;
};

export type CommitAction =
  | { type: "confirm_story" }
  | { type: "start_generation" }
  | { type: "start_merge" }
  | {
      type: "split_shot";
      shotId: number;
      beforeText: string;
      afterText: string;
    }
  | { type: "merge_shot"; shotId: number; mergedText: string }
  | { type: "delete_shot"; shotId: number };

export type WorkplaceEditorState = {
  server: WorkplaceData | null;
  overlay: Partial<Record<FieldPath, string>>;
  lease: Partial<Record<FieldPath, LeaseMode>>;
  dirty: Partial<Record<FieldPath, boolean>>;
  committing: boolean;
};

export type WorkplaceMutations = {
  saveStory: (storyMd: string) => Promise<void>;
  saveStoryProfile: (profile: StoryProfile) => Promise<void>;
  confirmStory: (storyMd?: string) => Promise<void>;
  startGeneration: () => Promise<void>;
  startMerge: () => Promise<void>;
  splitShot: (
    shotId: number,
    payload: { before_text: string; after_text: string },
  ) => Promise<void>;
  mergeShotUp: (shotId: number, mergedText?: string) => Promise<void>;
  deleteShot: (shotId: number) => Promise<void>;
};

export function beatPath(shotId: number): FieldPath {
  return `beat:${shotId}`;
}

export function parseBeatPath(path: FieldPath): number | null {
  if (path === "story") return null;
  const id = Number(path.slice("beat:".length));
  return Number.isFinite(id) ? id : null;
}

export function segmentIdForShot(shotId: number): string {
  return `shot_${String(shotId).padStart(3, "0")}`;
}
