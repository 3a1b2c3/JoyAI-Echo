import { MessageBubble } from "@/components/MessageBubble";
import type { NextContinuousControl } from "@/components/thread/MemoryReviewCard";
import type {
  MemoryReview,
  MemoryWorkspaceAsset,
  ShotMemoryAssetCreate,
  UIMessage,
} from "@/lib/types";

interface ThreadMessagesProps {
  messages: UIMessage[];
  onAnswerQuestion?: (messageId: string, cardId: string, value: string) => void;
  questionsReady?: boolean;
  onMemoryReviewAction?: (
    review: MemoryReview,
    action: "approve" | "reselect" | "manual_select" | "select_mode",
    memoryId?: string,
    timestampSec?: number,
    selectionMode?: "manual" | "vlm",
    retainedMemoryIds?: string[],
  ) => void | Promise<void>;
  memoryAssets?: MemoryWorkspaceAsset[];
  onCreateMemoryAsset?: (
    shotId: number,
    asset: ShotMemoryAssetCreate,
  ) => Promise<void>;
  getNextContinuous?: (
    review: MemoryReview,
  ) => NextContinuousControl | null;
}

export function ThreadMessages({
  messages,
  onAnswerQuestion,
  questionsReady = true,
  onMemoryReviewAction,
  memoryAssets = [],
  onCreateMemoryAsset,
  getNextContinuous,
}: ThreadMessagesProps) {
  return (
    <div className="flex w-full flex-col gap-5">
      {messages.map((message) => (
        <MessageBubble
          key={message.id}
          message={message}
          questionsReady={questionsReady}
          onAnswerQuestion={onAnswerQuestion}
          onMemoryReviewAction={onMemoryReviewAction}
          memoryAssets={memoryAssets}
          onCreateMemoryAsset={onCreateMemoryAsset}
          getNextContinuous={getNextContinuous}
        />
      ))}
    </div>
  );
}
