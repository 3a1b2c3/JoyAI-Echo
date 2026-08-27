import { useCallback, useEffect, useMemo, useReducer, useRef } from "react";

import type { StoryProfile } from "@/lib/types";

import {
  applyServerSnapshot,
  buildSegmentsFromServer,
  editorClearOverlays,
  editorEdit,
  editorGet,
  editorHold,
  editorRelease,
  editorSetCommitting,
  hasDirtyBeats,
  initialEditorState,
  mergeStoryProfileWithOverlay,
} from "./reducer";
import {
  beatPath,
  type CommitAction,
  type EditorSegment,
  type FieldPath,
  type WorkplaceEditorState,
  type WorkplaceMutations,
} from "./types";

type EditorAction =
  | {
      type: "SERVER_SNAPSHOT";
      workplace: NonNullable<WorkplaceEditorState["server"]>;
    }
  | { type: "RESET" }
  | { type: "HOLD"; path: FieldPath }
  | { type: "EDIT"; path: FieldPath; value: string }
  | { type: "RELEASE"; path: FieldPath }
  | { type: "CLEAR_OVERLAYS"; paths?: FieldPath[] }
  | { type: "SET_COMMITTING"; committing: boolean };

function reducer(
  state: WorkplaceEditorState,
  action: EditorAction,
): WorkplaceEditorState {
  switch (action.type) {
    case "SERVER_SNAPSHOT":
      return applyServerSnapshot(state, action.workplace);
    case "RESET":
      return initialEditorState;
    case "HOLD":
      return editorHold(state, action.path);
    case "EDIT":
      return editorEdit(state, action.path, action.value);
    case "RELEASE":
      return editorRelease(state, action.path);
    case "CLEAR_OVERLAYS":
      return editorClearOverlays(state, action.paths);
    case "SET_COMMITTING":
      return editorSetCommitting(state, action.committing);
    default:
      return state;
  }
}

export type UseWorkplaceEditorResult = {
  get: (path: FieldPath) => string;
  hold: (path: FieldPath) => void;
  edit: (path: FieldPath, value: string) => void;
  release: (path: FieldPath) => void;
  saveStory: () => Promise<void>;
  saveBeats: () => Promise<void>;
  commit: (action: CommitAction) => Promise<void>;
  segments: EditorSegment[];
  storyReadOnly: boolean;
  beatsReadOnly: boolean;
  storyDirty: boolean;
  beatsDirty: boolean;
  committing: boolean;
  /** True while any story/beat textarea holds a field lease (focus). */
  isFieldFocused: boolean;
};

export function useWorkplaceEditor(
  workplace: WorkplaceEditorState["server"],
  mutations: WorkplaceMutations,
): UseWorkplaceEditorResult {
  const [state, dispatch] = useReducer(reducer, initialEditorState);
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    if (!workplace) {
      dispatch({ type: "RESET" });
      return;
    }
    dispatch({ type: "SERVER_SNAPSHOT", workplace });
  }, [workplace]);

  const get = useCallback(
    (path: FieldPath) => editorGet(stateRef.current, path),
    [],
  );

  const flushStoryIfDirty = useCallback(async () => {
    const current = stateRef.current;

    if (!current.server?.story_editable || !current.dirty.story) return;
    const storyMd = editorGet(current, "story").trim();
    if (!storyMd) return;
    await mutations.saveStory(storyMd);
    dispatch({ type: "CLEAR_OVERLAYS", paths: ["story"] });
  }, [mutations]);

  const flushBeatsIfDirty = useCallback(async () => {
    const current = stateRef.current;
    if (!current.server?.beats_editable || !hasDirtyBeats(current.dirty))
      return;
    const profile = mergeStoryProfileWithOverlay(
      current.server,
      current.overlay,
      current.dirty,
    );
    if (!profile) return;
    await mutations.saveStoryProfile(profile as StoryProfile);
    dispatch({ type: "CLEAR_OVERLAYS" });
  }, [mutations]);

  const saveStory = useCallback(async () => {
    dispatch({ type: "SET_COMMITTING", committing: true });
    try {
      await flushStoryIfDirty();
    } finally {
      dispatch({ type: "SET_COMMITTING", committing: false });
    }
  }, [flushStoryIfDirty]);

  const saveBeats = useCallback(async () => {
    dispatch({ type: "SET_COMMITTING", committing: true });
    try {
      await flushBeatsIfDirty();
    } finally {
      dispatch({ type: "SET_COMMITTING", committing: false });
    }
  }, [flushBeatsIfDirty]);

  const hold = useCallback((path: FieldPath) => {
    dispatch({ type: "HOLD", path });
  }, []);

  const edit = useCallback((path: FieldPath, value: string) => {
    dispatch({ type: "EDIT", path, value });
  }, []);

  const release = useCallback((path: FieldPath) => {
    dispatch({ type: "RELEASE", path });
  }, []);

  const commit = useCallback(
    async (action: CommitAction) => {
      dispatch({ type: "SET_COMMITTING", committing: true });
      try {
        switch (action.type) {
          case "confirm_story": {
            // Workflow 步骤不隐式 save；须先点「保存」再确认，避免重复调度 edit LLM。
            await mutations.confirmStory();
            dispatch({ type: "CLEAR_OVERLAYS", paths: ["story"] });
            break;
          }
          case "start_generation": {
            // 同上：分镜 overlay 须经 saveBeats 持久化后再进入生成流程。
            await mutations.startGeneration();
            dispatch({ type: "CLEAR_OVERLAYS" });
            break;
          }
          case "start_merge": {
            await mutations.startMerge();
            break;
          }
          case "split_shot": {
            await mutations.splitShot(action.shotId, {
              before_text: action.beforeText.trim(),
              after_text: action.afterText.trim(),
            });
            dispatch({ type: "CLEAR_OVERLAYS" });
            break;
          }
          case "merge_shot": {
            await mutations.mergeShotUp(action.shotId, action.mergedText);
            dispatch({ type: "CLEAR_OVERLAYS" });
            break;
          }
          case "delete_shot": {
            await mutations.deleteShot(action.shotId);
            dispatch({ type: "CLEAR_OVERLAYS" });
            break;
          }
        }
      } finally {
        dispatch({ type: "SET_COMMITTING", committing: false });
      }
    },
    [mutations],
  );

  const segments = useMemo(() => {
    if (!state.server) return [];
    return buildSegmentsFromServer(
      state.server,
      state.overlay,
      state.lease,
      state.dirty,
    );
  }, [state.dirty, state.lease, state.overlay, state.server]);

  const storyReadOnly = !state.server?.story_editable;
  const beatsReadOnly = !state.server?.beats_editable;
  const storyDirty = Boolean(state.dirty.story);
  const beatsDirty = hasDirtyBeats(state.dirty);
  const isFieldFocused = useMemo(
    () => Object.values(state.lease).some((mode) => mode === "held"),
    [state.lease],
  );

  return {
    get,
    hold,
    edit,
    release,
    saveStory,
    saveBeats,
    commit,
    segments,
    storyReadOnly,
    beatsReadOnly,
    storyDirty,
    beatsDirty,
    committing: state.committing,
    isFieldFocused,
  };
}

export { beatPath };
