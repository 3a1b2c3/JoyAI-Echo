import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import {
  useWorkplaceEditor,
  type UseWorkplaceEditorResult,
} from "@/hooks/workplace-editor/useWorkplaceEditor";
import { useWorkplace, type UseWorkplaceResult } from "@/hooks/useWorkplace";

/**
 * Per-session workplace subscription. A single Provider call avoids duplicate
 * HTTP polling and WebSocket handlers when both chat and panel need workplace.
 *
 * `composerSendBlocked` reflects right-side textarea focus (not dirty state):
 * while the user edits in WorkplacePanel, the left chat send button is disabled.
 */
type WorkplaceContextValue = UseWorkplaceResult & {
  editor: UseWorkplaceEditorResult;
  composerSendBlocked: boolean;
  setAuxTextEditing: (active: boolean) => void;
};

const WorkplaceContext = createContext<WorkplaceContextValue | null>(null);

export function WorkplaceProvider({
  sessionKey,
  children,
}: {
  sessionKey: string | null;
  children: ReactNode;
}) {
  const workplaceResult = useWorkplace(sessionKey);
  const editor = useWorkplaceEditor(workplaceResult.workplace, {
    saveStory: workplaceResult.saveStory,
    saveStoryProfile: workplaceResult.saveStoryProfile,
    confirmStory: workplaceResult.confirmStory,
    startGeneration: workplaceResult.startGeneration,
    startMerge: workplaceResult.startMerge,
    splitShot: (shotId, payload) => workplaceResult.splitShot(shotId, payload),
    mergeShotUp: workplaceResult.mergeShotUp,
    deleteShot: workplaceResult.deleteShot,
  });

  const [auxEditingCount, setAuxEditingCount] = useState(0);
  const setAuxTextEditing = useCallback((active: boolean) => {
    setAuxEditingCount((count) => Math.max(0, active ? count + 1 : count - 1));
  }, []);

  const composerSendBlocked =
    editor.isFieldFocused ||
    auxEditingCount > 0 ||
    editor.storyDirty ||
    editor.beatsDirty;

  const value = useMemo(
    () => ({
      ...workplaceResult,
      editor,
      composerSendBlocked,
      setAuxTextEditing,
    }),
    [workplaceResult, editor, composerSendBlocked, setAuxTextEditing],
  );

  return (
    <WorkplaceContext.Provider value={value}>
      {children}
    </WorkplaceContext.Provider>
  );
}

export function useWorkplaceContext(): WorkplaceContextValue {
  const ctx = useContext(WorkplaceContext);
  if (!ctx) {
    throw new Error(
      "useWorkplaceContext must be used within a WorkplaceProvider",
    );
  }
  return ctx;
}
