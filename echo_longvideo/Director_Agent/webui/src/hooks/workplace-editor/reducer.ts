/**
 * Overlay + field lease merge for workplace text fields.
 *
 * Poll, WebSocket, and mutation responses all funnel through
 * `applyServerSnapshot`. Held/dirty paths keep local overlay values so
 * textarea editing is not clobbered by background refresh.
 */
import type { StoryProfileBeat, WorkplaceData } from "@/lib/types";

import {
  beatPath,
  type EditorSegment,
  type FieldPath,
  type LeaseMode,
  segmentIdForShot,
  type WorkplaceEditorState,
} from "./types";

export const initialEditorState: WorkplaceEditorState = {
  server: null,
  overlay: {},
  lease: {},
  dirty: {},
  committing: false,
};

function beatsFromServer(server: WorkplaceData): StoryProfileBeat[] {
  const beats = server.story_profile?.beats;
  if (!Array.isArray(beats)) return [];
  return beats
    .map((beat, index) => {
      const shotId = Number(beat?.shot_id ?? index + 1);
      const summary = String(beat?.summary ?? "").trim();
      if (!Number.isFinite(shotId) || !summary) return null;
      return { shot_id: shotId, summary };
    })
    .filter((beat): beat is StoryProfileBeat => beat !== null);
}

export function serverValueForPath(
  server: WorkplaceData,
  path: FieldPath,
): string {
  if (path === "story") return server.story_md ?? "";
  const shotId = Number(path.slice("beat:".length));
  const beat = beatsFromServer(server).find((item) => item.shot_id === shotId);
  return beat?.summary ?? "";
}

export function buildSegmentsFromServer(
  server: WorkplaceData,
  overlay: Partial<Record<FieldPath, string>>,
  lease: Partial<Record<FieldPath, LeaseMode>>,
  dirty: Partial<Record<FieldPath, boolean>>,
): EditorSegment[] {
  return beatsFromServer(server).map((beat) => {
    const path = beatPath(beat.shot_id);
    const held = lease[path] === "held" || dirty[path];
    const text =
      held && overlay[path] !== undefined ? overlay[path]! : beat.summary;
    return {
      id: segmentIdForShot(beat.shot_id),
      shotId: beat.shot_id,
      text,
    };
  });
}

function shouldPreservePath(
  state: WorkplaceEditorState,
  path: FieldPath,
): boolean {
  if (state.committing) return true;
  return state.lease[path] === "held" || Boolean(state.dirty[path]);
}

export function applyServerSnapshot(
  state: WorkplaceEditorState,
  next: WorkplaceData,
): WorkplaceEditorState {
  const stageChanged = state.server?.stage !== next.stage;
  const workChanged = state.server?.work_id !== next.work_id;

  if (stageChanged || workChanged) {
    return {
      server: next,
      overlay: {},
      lease: {},
      dirty: {},
      committing: false,
    };
  }

  const overlay = { ...state.overlay };
  const lease = { ...state.lease };
  const dirty = { ...state.dirty };

  if (!shouldPreservePath(state, "story")) {
    delete overlay.story;
    delete lease.story;
    delete dirty.story;
  }

  const nextBeatIds = new Set(
    beatsFromServer(next).map((beat) => beat.shot_id),
  );
  for (const path of Object.keys(overlay) as FieldPath[]) {
    if (path === "story") continue;
    const shotId = Number(path.slice("beat:".length));
    if (!nextBeatIds.has(shotId)) {
      delete overlay[path];
      delete lease[path];
      delete dirty[path];
    }
  }

  for (const beat of beatsFromServer(next)) {
    const path = beatPath(beat.shot_id);
    if (!shouldPreservePath(state, path)) {
      delete overlay[path];
      delete lease[path];
      delete dirty[path];
    }
  }

  return {
    ...state,
    server: next,
    overlay,
    lease,
    dirty,
    committing: state.committing,
  };
}

export function editorGet(
  state: WorkplaceEditorState,
  path: FieldPath,
): string {
  if (path in state.overlay) return state.overlay[path] ?? "";
  if (state.server) return serverValueForPath(state.server, path);
  return "";
}

export function editorHold(
  state: WorkplaceEditorState,
  path: FieldPath,
): WorkplaceEditorState {
  const value = editorGet(state, path);
  return {
    ...state,
    overlay: { ...state.overlay, [path]: value },
    lease: { ...state.lease, [path]: "held" },
  };
}

function fieldValueDiffersFromServer(
  server: WorkplaceData,
  path: FieldPath,
  value: string,
): boolean {
  return value.trim() !== serverValueForPath(server, path).trim();
}

export function editorEdit(
  state: WorkplaceEditorState,
  path: FieldPath,
  value: string,
): WorkplaceEditorState {
  const overlay = { ...state.overlay };
  const dirty = { ...state.dirty };
  const lease = { ...state.lease, [path]: "held" as LeaseMode };

  if (state.server && !fieldValueDiffersFromServer(state.server, path, value)) {
    delete overlay[path];
    delete dirty[path];
  } else {
    overlay[path] = value;
    dirty[path] = true;
  }

  return { ...state, overlay, lease, dirty };
}

export function editorRelease(
  state: WorkplaceEditorState,
  path: FieldPath,
): WorkplaceEditorState {
  const lease = { ...state.lease };
  delete lease[path];
  return { ...state, lease };
}

export function editorClearOverlays(
  state: WorkplaceEditorState,
  paths?: FieldPath[],
): WorkplaceEditorState {
  if (!paths) {
    return {
      ...state,
      overlay: {},
      lease: {},
      dirty: {},
    };
  }
  const overlay = { ...state.overlay };
  const lease = { ...state.lease };
  const dirty = { ...state.dirty };
  for (const path of paths) {
    delete overlay[path];
    delete lease[path];
    delete dirty[path];
  }
  return { ...state, overlay, lease, dirty };
}

export function editorSetCommitting(
  state: WorkplaceEditorState,
  committing: boolean,
): WorkplaceEditorState {
  return { ...state, committing };
}

export function mergeStoryProfileWithOverlay(
  server: WorkplaceData,
  overlay: Partial<Record<FieldPath, string>>,
  dirty: Partial<Record<FieldPath, boolean>>,
): Record<string, unknown> | null {
  const base = server.story_profile;
  if (!base || typeof base !== "object") return null;
  const beats = beatsFromServer(server).map((beat) => {
    const path = beatPath(beat.shot_id);
    const text =
      dirty[path] && overlay[path] !== undefined
        ? overlay[path]!
        : beat.summary;
    return { shot_id: beat.shot_id, summary: text.trim() };
  });
  if (beats.some((beat) => !beat.summary)) return null;
  return {
    ...base,
    beats,
  };
}

export function hasDirtyBeats(
  dirty: Partial<Record<FieldPath, boolean>>,
): boolean {
  return Object.entries(dirty).some(
    ([path, isDirty]) => isDirty && path.startsWith("beat:"),
  );
}
