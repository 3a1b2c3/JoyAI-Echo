import { useCallback, useMemo, useState } from "react";

import { SplitPane } from "@/components/longvideo/SplitPane";
import { ThreadShell } from "@/components/thread/ThreadShell";
import { WorkplacePanel } from "@/components/workplace/WorkplacePanel";
import { useWorkplaceContext } from "@/providers/WorkplaceProvider";
import { useThreadStream } from "@/hooks/useThreadStream";
import { memoryReviewMessages } from "@/lib/memory-review";
import type { ChatSummary, MemoryReview } from "@/lib/types";

interface ThreadWorkplaceShellProps {
  session: ChatSummary | null;
  title: string;
  onToggleSidebar: () => void;
  onGoHome: () => void;
  onNewChat: () => Promise<string | null>;
  hideSidebarToggleOnDesktop?: boolean;
  onReplyEnd?: () => void;
  quickMode?: boolean;
}

/** Chat and Director workplace shown side by side. */
export function ThreadWorkplaceShell({
  session,
  title,
  onToggleSidebar,
  onGoHome,
  onNewChat,
  hideSidebarToggleOnDesktop,
  onReplyEnd,
  quickMode = false,
}: ThreadWorkplaceShellProps) {
  const [activeTab, setActiveTab] = useState<"story" | "shots">("shots");
  const {
    approveMemoryReview,
    composerSendBlocked,
    createShotMemoryAsset,
    reselectMemoryReview,
    selectMemoryReviewFrame,
    selectMemoryReviewMode,
    setContinuousMode,
    workplace,
    refresh,
  } = useWorkplaceContext();
  const videoSizeLocked =
    workplace?.shots.some((shot) =>
      ["queued", "generated", "review_pass", "approved"].includes(shot.status),
    ) ?? false;
  const stream = useThreadStream(session, { onReplyEnd });
  const autoGenerateMode = Boolean(
    quickMode || workplace?.auto_generate || session?.autoGenerate,
  );
  const streamWithMemoryReviews = useMemo(() => {
    const reviewMessages = workplace ? memoryReviewMessages(workplace) : [];
    const regularMessages = stream.messages.filter(
      (message) => !message.id.startsWith("memory-review:"),
    );
    return {
      ...stream,
      messages: [...regularMessages, ...reviewMessages],
    };
  }, [stream, workplace]);
  const hasVisibleMemoryReview = streamWithMemoryReviews.messages.some((message) =>
    message.id.startsWith("memory-review:"),
  );
  const hideMemoryReviews = autoGenerateMode && !hasVisibleMemoryReview;
  const showThinking = stream.messages.length > 0;

  const onMemoryReviewAction = useCallback(
    async (
      review: Parameters<typeof approveMemoryReview>[0],
      action: "approve" | "reselect" | "manual_select" | "select_mode",
      memoryId?: string,
      timestampSec?: number,
      selectionMode?: "manual" | "vlm",
      retainedMemoryIds?: string[],
    ) => {
      if (action === "select_mode" && selectionMode) {
        await selectMemoryReviewMode(review, selectionMode);
        return;
      }
      if (action === "approve") {
        await approveMemoryReview(review, retainedMemoryIds ?? []);
        return;
      }
      if (action === "reselect") {
        await reselectMemoryReview(review, memoryId);
        return;
      }
      if (memoryId !== undefined && timestampSec !== undefined) {
        await selectMemoryReviewFrame(review, memoryId, timestampSec);
      }
    },
    [
      approveMemoryReview,
      reselectMemoryReview,
      selectMemoryReviewFrame,
      selectMemoryReviewMode,
    ],
  );

  const getNextContinuous = useCallback(
    (review: MemoryReview) => {
      if (!workplace) return null;
      const nextShotId = review.shot_id + 1;
      const nextShot = workplace.shots.find(
        (shot) => shot.shot_id === nextShotId,
      );
      if (!nextShot) return null;
      const status = String(nextShot.status || "");
      const locked =
        status !== "planned" && status !== "prompt_ready";
      return {
        nextShotId,
        enabled: nextShot.continuous_enabled ?? false,
        disabled: locked,
        onToggle: (enabled: boolean) => {
          void setContinuousMode(nextShotId, enabled);
        },
      };
    },
    [setContinuousMode, workplace],
  );

  return (
    <SplitPane
      left={
        <ThreadShell
          session={session}
          stream={streamWithMemoryReviews}
          title={title}
          onToggleSidebar={onToggleSidebar}
          onGoHome={onGoHome}
          onNewChat={onNewChat}
          hideSidebarToggleOnDesktop={hideSidebarToggleOnDesktop}
          composerSendDisabled={composerSendBlocked}
          onMemoryReviewAction={
            hideMemoryReviews ? undefined : onMemoryReviewAction
          }
          memoryAssets={workplace?.memory_workspace_assets ?? []}
          onCreateMemoryAsset={createShotMemoryAsset}
          getNextContinuous={
            hideMemoryReviews ? undefined : getNextContinuous
          }
          videoSizeLocked={videoSizeLocked}
          referenceImageLocked={Boolean(workplace?.reference_image_locked)}
          persistedReferenceImage={workplace?.reference_image ?? null}
          persistedAutoGenerate={Boolean(
            quickMode || workplace?.auto_generate || session?.autoGenerate,
          )}
          forceAutoGenerate={quickMode}
          onReferenceImageSynced={refresh}
        />
      }
      right={
        <WorkplacePanel
          showThinking={showThinking}
          sessionKey={session?.key ?? null}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          onNewChat={onNewChat}
          chatBusy={stream.isStreaming}
        />
      }
    />
  );
}
