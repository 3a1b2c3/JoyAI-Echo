import { useCallback, useEffect, useRef, useState } from "react";

import {
  acceptAllShots as acceptAllShotsApi,
  acceptShot,
  approveMemoryReview as approveMemoryReviewApi,
  reselectMemoryReview as reselectMemoryReviewApi,
  selectMemoryReviewFrame as selectMemoryReviewFrameApi,
  selectMemoryReviewMode as selectMemoryReviewModeApi,
  ApiError,
  abortGeneration as abortGenerationApi,
  confirmStory as confirmStoryApi,
  deleteShot as deleteShotApi,
  fetchWorkplace,
  startGeneration as startGenerationApi,
  startAutoGenerate as startAutoGenerateApi,
  generateAll as generateAllApi,
  generateShot as generateShotApi,
  setShotContinuousMode as setShotContinuousModeApi,
  continuousGenerateShot as continuousGenerateShotApi,
  mergeShotUp as mergeShotUpApi,
  mockAcceptAllShots,
  mockDeleteShot,
  reviseShot,
  saveShotPrompt as saveShotPromptApi,
  updateShotDuration as updateShotDurationApi,
  splitShot as splitShotApi,
  startMerge as startMergeApi,
  regenerate as regenerateApi,
  updateEchoLike as updateEchoLikeApi,
  recordEchoDownloadPrompt as recordEchoDownloadPromptApi,
  type SplitShotPayload,
} from "@/lib/api";
import { webuiChatIdFromKey } from "@/lib/session-key";
import type {
  ConnectionStatus,
  MemoryAssetUpload,
  ShotMemoryAssetCreate,
  MemoryReview,
  MemorySlotReference,
  StoryProfile,
  WorkplaceData,
} from "@/lib/types";
import { useClient } from "@/providers/ClientProvider";

const POLL_MS = 4_000;

const USE_MOCK_DELETE_SHOT = false;

const USE_MOCK_ACCEPT_ALL = false;

/** Tracks which async workflow action is in flight for smarter busy clearing. */
type WorkflowAction = "confirm_story" | "start_merge";

function shouldClearWorkflowBusy(
  startedStage: string | null | undefined,
  next: WorkplaceData,
  action: WorkflowAction | null,
): boolean {
  if (action === "confirm_story") {
    // confirm-story may land on story_confirmed (01) until shot_count is locked.
    return (
      next.stage === "shot_planning" ||
      next.stage === "story_confirmed" ||
      (next.shots?.length ?? 0) > 0
    );
  }
  return next.stage !== startedStage;
}

/** Return shape of {@link useWorkplace}; shared by WorkplaceProvider context. */
export type UseWorkplaceResult = {
  workplace: WorkplaceData | null;
  loading: boolean;
  error: string | null;
  mutatingShotId: number | null;
  workflowBusy: boolean;
  splitMergeBusy: boolean;
  acceptAllBusy: boolean;
  memoryReviewBusy: {
    shotId: number;
    action: "approve" | "reselect" | "manual_select" | "select_mode";
  } | null;
  memoryWorkspaceBusy: boolean;
  refresh: () => Promise<void>;
  accept: (shotId: number) => Promise<void>;
  acceptAll: () => Promise<void>;
  approveMemoryReview: (
    review: MemoryReview,
    retainedMemoryIds: string[],
  ) => Promise<void>;
  reselectMemoryReview: (
    review: MemoryReview,
    memoryId?: string,
  ) => Promise<void>;
  selectMemoryReviewFrame: (
    review: MemoryReview,
    memoryId: string,
    timestampSec: number,
  ) => Promise<void>;
  selectMemoryReviewMode: (
    review: MemoryReview,
    mode: "manual" | "vlm",
  ) => Promise<void>;
  saveMemoryAsset: (asset: MemoryAssetUpload) => Promise<void>;
  createShotMemoryAsset: (
    shotId: number,
    asset: ShotMemoryAssetCreate,
  ) => Promise<void>;
  deleteMemoryAsset: (assetId: string) => Promise<void>;
  saveShotMemorySlots: (
    shotId: number,
    slots: MemorySlotReference[],
  ) => Promise<void>;
  revise: (shotId: number, feedback: string) => Promise<void>;
  confirmStory: (storyMd?: string) => Promise<void>;
  saveStory: (storyMd: string) => Promise<void>;
  saveStoryProfile: (profile: StoryProfile) => Promise<void>;
  startGeneration: () => Promise<void>;
  startAutoGenerate: () => Promise<void>;
  abortGeneration: () => Promise<void>;
  generateAll: () => Promise<void>;
  startMerge: () => Promise<void>;
  regenerate: () => Promise<void>;
  splitShot: (shotId: number, payload: SplitShotPayload) => Promise<void>;
  mergeShotUp: (shotId: number, mergedText?: string) => Promise<void>;
  deleteShot: (shotId: number) => Promise<void>;
  generateOneShot: (
    shotId: number,
    referenceImage?: {
      url: string;
      name?: string;
      width?: number;
      height?: number;
    } | null,
  ) => Promise<void>;
  setContinuousMode: (shotId: number, enabled: boolean) => Promise<void>;
  continuousGenerateOneShot: (shotId: number) => Promise<void>;
  saveShotPrompt: (shotId: number, summary: string) => Promise<void>;
  updateShotDuration: (shotId: number, durationSec: number) => Promise<void>;
  updateEchoLike: (action: 1 | 2) => Promise<void>;
  recordEchoDownloadPrompt: () => Promise<void>;
  promptOverrides: Record<number, string>;
};

export function useWorkplace(sessionKey: string | null): UseWorkplaceResult {
  const { token, client } = useClient();
  const tokenRef = useRef(token);
  tokenRef.current = token;
  const [workplace, setWorkplace] = useState<WorkplaceData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mutatingShotId, setMutatingShotId] = useState<number | null>(null);
  const [promptOverrides, setPromptOverrides] = useState<
    Record<number, string>
  >({});
  const [workflowBusy, setWorkflowBusy] = useState(false);
  const [splitMergeBusy, setSplitMergeBusy] = useState(false);
  const [acceptAllBusy, setAcceptAllBusy] = useState(false);
  const [memoryReviewBusy, setMemoryReviewBusy] = useState<{
    shotId: number;
    action: "approve" | "reselect" | "manual_select" | "select_mode";
  } | null>(null);
  const [memoryWorkspaceBusy, setMemoryWorkspaceBusy] = useState(false);
  const workflowStageRef = useRef<string | null | undefined>(null);
  const workflowActionRef = useRef<WorkflowAction | null>(null);
  const prevSessionKeyRef = useRef<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>(
    client.status,
  );

  const refresh = useCallback(async () => {
    if (!sessionKey) {
      setWorkplace(null);
      setError(null);
      return;
    }
    setLoading(true);
    try {
      // Read token from ref so auth refresh does not recreate ``refresh`` and
      // retrigger the session effect (which clears workplace and flashes overlay).
      const next = await fetchWorkplace(tokenRef.current, sessionKey);
      setWorkplace(next);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [sessionKey]);

  const applyWorkplaceUpdate = useCallback(
    (next: WorkplaceData) => {
      setWorkplace(next);
      setError(null);
      if (
        workflowBusy &&
        workflowStageRef.current !== undefined &&
        shouldClearWorkflowBusy(
          workflowStageRef.current,
          next,
          workflowActionRef.current,
        )
      ) {
        setWorkflowBusy(false);
        workflowStageRef.current = undefined;
        workflowActionRef.current = null;
      }
    },
    [workflowBusy],
  );

  useEffect(() => client.onStatus(setConnectionStatus), [client]);

  useEffect(() => {
    const sessionChanged = prevSessionKeyRef.current !== sessionKey;
    prevSessionKeyRef.current = sessionKey;
    if (!sessionKey) {
      setWorkplace(null);
      setError(null);
      setWorkflowBusy(false);
      setPromptOverrides({});
      workflowStageRef.current = undefined;
      workflowActionRef.current = null;
      setLoading(false);
      return;
    }
    // Clear only when the chat changes. Reconnect/status flaps must keep the
    // current workplace mounted so step-4 video does not remount and autoPlay.
    if (sessionChanged) {
      setWorkplace(null);
      setError(null);
      setLoading(true);
    }
    if (connectionStatus !== "open") return;
    void refresh();
    const id = window.setInterval(() => {
      void refresh();
    }, POLL_MS);
    return () => window.clearInterval(id);
  }, [connectionStatus, refresh, sessionKey]);

  useEffect(() => {
    if (!sessionKey) return;
    const chatId = webuiChatIdFromKey(sessionKey);
    if (!chatId) return;
    return client.onChat(chatId, (event) => {
      if (event.event !== "workplace_updated") return;
      if (event.workplace) {
        applyWorkplaceUpdate(event.workplace);
        return;
      }
      void refresh();
    });
  }, [applyWorkplaceUpdate, client, refresh, sessionKey]);

  const beginWorkflow = useCallback(
    (
      currentStage: string | null | undefined,
      action: WorkflowAction | null = null,
    ) => {
      workflowStageRef.current = currentStage ?? null;
      workflowActionRef.current = action;
      setWorkflowBusy(true);
    },
    [],
  );

  const confirmStory = useCallback(
    async (storyMd?: string) => {
      if (!sessionKey) return;
      beginWorkflow(workplace?.stage, "confirm_story");
      try {
        const result = await confirmStoryApi(token, sessionKey, storyMd);
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setWorkflowBusy(false);
        workflowStageRef.current = undefined;
        workflowActionRef.current = null;
        setError((err as Error).message);
      }
    },
    [applyWorkplaceUpdate, beginWorkflow, sessionKey, token, workplace?.stage],
  );

  const saveStory = useCallback(
    async (storyMd: string) => {
      if (!sessionKey) return;
      const chatId = webuiChatIdFromKey(sessionKey);
      try {
        const { workplace } = await client.saveWorkplaceStory(chatId, storyMd);
        applyWorkplaceUpdate(workplace);
      } catch (err) {
        setError((err as Error).message);
        throw err;
      }
    },
    [applyWorkplaceUpdate, client, sessionKey],
  );

  const saveStoryProfile = useCallback(
    async (profile: StoryProfile) => {
      if (!sessionKey) return;
      const chatId = webuiChatIdFromKey(sessionKey);
      try {
        const { workplace } = await client.saveWorkplaceStoryProfile(
          chatId,
          profile,
        );
        applyWorkplaceUpdate(workplace);
      } catch (err) {
        setError((err as Error).message);
        throw err;
      }
    },
    [applyWorkplaceUpdate, client, sessionKey],
  );

  const startGeneration = useCallback(async () => {
    if (!sessionKey) return;
    beginWorkflow(workplace?.stage);
    try {
      const result = await startGenerationApi(token, sessionKey);
      applyWorkplaceUpdate(result.workplace);
      // generate-all submits jobs synchronously; no agent stage transition to wait on.
      setWorkflowBusy(false);
      workflowStageRef.current = undefined;
      workflowActionRef.current = null;
    } catch (err) {
      setWorkflowBusy(false);
      workflowStageRef.current = undefined;
      workflowActionRef.current = null;
      setError((err as Error).message);
    }
  }, [
    applyWorkplaceUpdate,
    beginWorkflow,
    sessionKey,
    token,
    workplace?.stage,
  ]);

  const startAutoGenerate = useCallback(async () => {
    if (!sessionKey) return;
    beginWorkflow(workplace?.stage);
    try {
      const result = await startAutoGenerateApi(token, sessionKey);
      applyWorkplaceUpdate(result.workplace);
      setWorkflowBusy(false);
      workflowStageRef.current = undefined;
      workflowActionRef.current = null;
    } catch (err) {
      setWorkflowBusy(false);
      workflowStageRef.current = undefined;
      workflowActionRef.current = null;
      setError((err as Error).message);
    }
  }, [
    applyWorkplaceUpdate,
    beginWorkflow,
    sessionKey,
    token,
    workplace?.stage,
  ]);

  /** Roll stuck shot_generating prep back to shot_planning so the user can retry. */
  const abortGeneration = useCallback(async () => {
    if (!sessionKey) return;
    try {
      const result = await abortGenerationApi(token, sessionKey);
      applyWorkplaceUpdate(result.workplace);
      setWorkflowBusy(false);
      workflowStageRef.current = undefined;
      workflowActionRef.current = null;
    } catch (err) {
      setError((err as Error).message);
      throw err;
    }
  }, [applyWorkplaceUpdate, sessionKey, token]);

  /** 批量提交可生成镜头的 Echo 任务（FramesPanel「全部生成」）。 */
  const generateAll = useCallback(async () => {
    if (!sessionKey) return;
    if (
      workplace?.session_key &&
      workplace.session_key !== sessionKey
    ) {
      await refresh();
      return;
    }
    try {
      const result = await generateAllApi(token, sessionKey);
      if (result.workplace.session_key !== sessionKey) {
        await refresh();
        return;
      }
      applyWorkplaceUpdate(result.workplace);
    } catch (err) {
      setError((err as Error).message);
      throw err;
    }
  }, [applyWorkplaceUpdate, refresh, sessionKey, token, workplace?.session_key]);

  const startMerge = useCallback(async () => {
    if (!sessionKey) return;
    beginWorkflow(workplace?.stage, "start_merge");
    try {
      const result = await startMergeApi(token, sessionKey);
      applyWorkplaceUpdate(result.workplace);
    } catch (err) {
      setWorkflowBusy(false);
      workflowStageRef.current = undefined;
      workflowActionRef.current = null;
      setError((err as Error).message);
    }
  }, [
    applyWorkplaceUpdate,
    beginWorkflow,
    sessionKey,
    token,
    workplace?.stage,
  ]);

  /** 成片后回到分镜编辑；与 startMerge（重新触发合成）职责不同。 */
  const regenerate = useCallback(async () => {
    if (!sessionKey) return;
    try {
      const result = await regenerateApi(token, sessionKey);
      applyWorkplaceUpdate(result.workplace);
    } catch (err) {
      setError((err as Error).message);
      throw err;
    }
  }, [applyWorkplaceUpdate, sessionKey, token]);

  const splitShot = useCallback(
    async (shotId: number, payload: SplitShotPayload) => {
      if (!sessionKey) return;
      setSplitMergeBusy(true);
      try {
        const result = await splitShotApi(token, sessionKey, shotId, payload);
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setSplitMergeBusy(false);
      }
    },
    [applyWorkplaceUpdate, sessionKey, token],
  );

  const mergeShotUp = useCallback(
    async (shotId: number, mergedText?: string) => {
      if (!sessionKey) return;
      setSplitMergeBusy(true);
      try {
        const result = await mergeShotUpApi(
          token,
          sessionKey,
          shotId,
          mergedText,
        );
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setSplitMergeBusy(false);
      }
    },
    [applyWorkplaceUpdate, sessionKey, token],
  );

  const deleteShot = useCallback(
    async (shotId: number) => {
      if (!sessionKey || !workplace) return;
      setSplitMergeBusy(true);
      try {
        const result = USE_MOCK_DELETE_SHOT
          ? mockDeleteShot(workplace, shotId)
          : await deleteShotApi(token, sessionKey, shotId);
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setSplitMergeBusy(false);
      }
    },
    [applyWorkplaceUpdate, sessionKey, token, workplace],
  );

  const accept = useCallback(
    async (shotId: number) => {
      if (!sessionKey) return;
      setMutatingShotId(shotId);
      try {
        const result = await acceptShot(token, sessionKey, shotId);
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setMutatingShotId(null);
      }
    },
    [applyWorkplaceUpdate, sessionKey, token],
  );

  /** 批量接收已生成分镜（FramesPanel「全部接收」）。 */
  const acceptAll = useCallback(async () => {
    if (!sessionKey || !workplace) return;
    setAcceptAllBusy(true);
    try {
      const result = USE_MOCK_ACCEPT_ALL
        ? mockAcceptAllShots(workplace)
        : await acceptAllShotsApi(token, sessionKey);
      applyWorkplaceUpdate(result.workplace);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setAcceptAllBusy(false);
    }
  }, [applyWorkplaceUpdate, sessionKey, token, workplace]);

  const approveMemoryReview = useCallback(
    async (review: MemoryReview, retainedMemoryIds: string[]) => {
      if (!sessionKey) return;
      setMemoryReviewBusy({ shotId: review.shot_id, action: "approve" });
      try {
        const result = await approveMemoryReviewApi(
          token,
          sessionKey,
          review.shot_id,
          {
            review_id: review.review_id,
            attempt: review.attempt,
            retained_memory_ids: retainedMemoryIds,
          },
        );
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
        throw err;
      } finally {
        setMemoryReviewBusy(null);
      }
    },
    [applyWorkplaceUpdate, sessionKey, token],
  );

  const reselectMemoryReview = useCallback(
    async (review: MemoryReview, memoryId?: string) => {
      if (!sessionKey) return;
      setMemoryReviewBusy({ shotId: review.shot_id, action: "reselect" });
      try {
        const result = await reselectMemoryReviewApi(
          token,
          sessionKey,
          review.shot_id,
          {
            review_id: review.review_id,
            attempt: review.attempt,
            memory_id: memoryId,
          },
        );
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
        throw err;
      } finally {
        setMemoryReviewBusy(null);
      }
    },
    [applyWorkplaceUpdate, sessionKey, token],
  );

  const selectMemoryReviewFrame = useCallback(
    async (review: MemoryReview, memoryId: string, timestampSec: number) => {
      if (!sessionKey) return;
      setMemoryReviewBusy({ shotId: review.shot_id, action: "manual_select" });
      try {
        const result = await selectMemoryReviewFrameApi(
          token,
          sessionKey,
          review.shot_id,
          {
            review_id: review.review_id,
            attempt: review.attempt,
            memory_id: memoryId,
            timestamp_sec: timestampSec,
          },
        );
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
        throw err;
      } finally {
        setMemoryReviewBusy(null);
      }
    },
    [applyWorkplaceUpdate, sessionKey, token],
  );

  const selectMemoryReviewMode = useCallback(
    async (review: MemoryReview, mode: "manual" | "vlm") => {
      if (!sessionKey) return;
      setMemoryReviewBusy({ shotId: review.shot_id, action: "select_mode" });
      try {
        const result = await selectMemoryReviewModeApi(
          token,
          sessionKey,
          review.shot_id,
          {
            review_id: review.review_id,
            attempt: review.attempt,
            selection_mode: mode,
          },
        );
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
        throw err;
      } finally {
        setMemoryReviewBusy(null);
      }
    },
    [applyWorkplaceUpdate, sessionKey, token],
  );

  const saveMemoryAsset = useCallback(
    async (asset: MemoryAssetUpload) => {
      if (!sessionKey) return;
      setMemoryWorkspaceBusy(true);
      try {
        const chatId = webuiChatIdFromKey(sessionKey);
        const result = await client.saveWorkplaceMemoryAsset(chatId, asset);
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
        throw err;
      } finally {
        setMemoryWorkspaceBusy(false);
      }
    },
    [applyWorkplaceUpdate, client, sessionKey],
  );

  const createShotMemoryAsset = useCallback(
    async (shotId: number, asset: ShotMemoryAssetCreate) => {
      if (!sessionKey) return;
      setMemoryWorkspaceBusy(true);
      try {
        const chatId = webuiChatIdFromKey(sessionKey);
        const result = await client.createWorkplaceShotMemoryAsset(
          chatId,
          shotId,
          asset,
        );
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
        throw err;
      } finally {
        setMemoryWorkspaceBusy(false);
      }
    },
    [applyWorkplaceUpdate, client, sessionKey],
  );

  const deleteMemoryAsset = useCallback(
    async (assetId: string) => {
      if (!sessionKey) return;
      setMemoryWorkspaceBusy(true);
      try {
        const chatId = webuiChatIdFromKey(sessionKey);
        const result = await client.deleteWorkplaceMemoryAsset(chatId, assetId);
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
        throw err;
      } finally {
        setMemoryWorkspaceBusy(false);
      }
    },
    [applyWorkplaceUpdate, client, sessionKey],
  );

  const saveShotMemorySlots = useCallback(
    async (shotId: number, slots: MemorySlotReference[]) => {
      if (!sessionKey) return;
      setMemoryWorkspaceBusy(true);
      try {
        const chatId = webuiChatIdFromKey(sessionKey);
        const result = await client.saveWorkplaceShotMemorySlots(
          chatId,
          shotId,
          slots,
        );
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
        throw err;
      } finally {
        setMemoryWorkspaceBusy(false);
      }
    },
    [applyWorkplaceUpdate, client, sessionKey],
  );

  const revise = useCallback(
    async (shotId: number, feedback: string) => {
      if (!sessionKey) return;
      setMutatingShotId(shotId);
      try {
        const result = await reviseShot(token, sessionKey, shotId, feedback);
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setMutatingShotId(null);
      }
    },
    [applyWorkplaceUpdate, sessionKey, token],
  );

  const generateOneShot = useCallback(
    async (
      shotId: number,
      referenceImage?: {
        url: string;
        name?: string;
        width?: number;
        height?: number;
      } | null,
    ) => {
      if (!sessionKey) return;
      setMutatingShotId(shotId);
      try {
        const result = await generateShotApi(
          token,
          sessionKey,
          shotId,
          referenceImage,
        );
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setMutatingShotId(null);
      }
    },
    [applyWorkplaceUpdate, sessionKey, token],
  );

  const setContinuousMode = useCallback(
    async (shotId: number, enabled: boolean) => {
      if (!sessionKey) return;
      try {
        const result = await setShotContinuousModeApi(
          token,
          sessionKey,
          shotId,
          enabled,
        );
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
      }
    },
    [applyWorkplaceUpdate, sessionKey, token],
  );

  const continuousGenerateOneShot = useCallback(
    async (shotId: number) => {
      if (!sessionKey) return;
      setMutatingShotId(shotId);
      try {
        const result = await continuousGenerateShotApi(
          token,
          sessionKey,
          shotId,
        );
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setMutatingShotId(null);
      }
    },
    [applyWorkplaceUpdate, sessionKey, token],
  );

  const saveShotPrompt = useCallback(
    async (shotId: number, summary: string) => {
      if (!sessionKey) return;
      try {
        const result = await saveShotPromptApi(
          token,
          sessionKey,
          shotId,
          summary,
        );
        applyWorkplaceUpdate(result.workplace);
        setPromptOverrides((prev) => {
          if (!(shotId in prev)) return prev;
          const next = { ...prev };
          delete next[shotId];
          return next;
        });
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) {
          console.warn(
            "saveShotPrompt: backend not ready (404), using local override",
          );
          setPromptOverrides((prev) => ({ ...prev, [shotId]: summary }));
          return;
        }
        setError((err as Error).message);
        throw err;
      }
    },
    [applyWorkplaceUpdate, sessionKey, token],
  );

  const updateShotDuration = useCallback(
    async (shotId: number, durationSec: number) => {
      if (!sessionKey) return;
      try {
        const result = await updateShotDurationApi(
          token,
          sessionKey,
          shotId,
          durationSec,
        );
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
      }
    },
    [applyWorkplaceUpdate, sessionKey, token],
  );

  const updateEchoLike = useCallback(
    async (action: 1 | 2) => {
      if (!sessionKey) return;
      try {
        const result = await updateEchoLikeApi(token, sessionKey, action);
        applyWorkplaceUpdate(result.workplace);
      } catch (err) {
        setError((err as Error).message);
        throw err;
      }
    },
    [applyWorkplaceUpdate, sessionKey, token],
  );

  const recordEchoDownloadPrompt = useCallback(async () => {
    if (!sessionKey) return;
    try {
      const result = await recordEchoDownloadPromptApi(token, sessionKey);
      applyWorkplaceUpdate(result.workplace);
    } catch (err) {
      console.warn("failed to record prompt download", err);
    }
  }, [applyWorkplaceUpdate, sessionKey, token]);

  return {
    workplace,
    loading,
    error,
    mutatingShotId,
    workflowBusy,
    splitMergeBusy,
    acceptAllBusy,
    memoryReviewBusy,
    memoryWorkspaceBusy,
    refresh,
    accept,
    acceptAll,
    approveMemoryReview,
    reselectMemoryReview,
    selectMemoryReviewFrame,
    selectMemoryReviewMode,
    saveMemoryAsset,
    createShotMemoryAsset,
    deleteMemoryAsset,
    saveShotMemorySlots,
    revise,
    confirmStory,
    saveStory,
    saveStoryProfile,
    startGeneration,
    startAutoGenerate,
    abortGeneration,
    generateAll,
    startMerge,
    regenerate,
    splitShot,
    mergeShotUp,
    deleteShot,
    generateOneShot,
    setContinuousMode,
    continuousGenerateOneShot,
    saveShotPrompt,
    updateShotDuration,
    updateEchoLike,
    recordEchoDownloadPrompt,
    promptOverrides,
  };
}
