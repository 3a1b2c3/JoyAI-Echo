import { useState } from "react";
import { ImageIcon } from "lucide-react";
import { useTranslation } from "react-i18next";

import { ImageLightbox } from "@/components/ImageLightbox";
import { MarkdownText } from "@/components/MarkdownText";
import { QuestionCards } from "@/components/thread/QuestionCards";
import { MemoryReviewCard } from "@/components/thread/MemoryReviewCard";
import type { NextContinuousControl } from "@/components/thread/MemoryReviewCard";
import { cn } from "@/lib/utils";
import type {
  MemoryReview,
  MemoryWorkspaceAsset,
  ShotMemoryAssetCreate,
  UIImage,
  UIMessage,
  UIVideo,
} from "@/lib/types";

interface MessageBubbleProps {
  message: UIMessage;
  onAnswerQuestion?: (messageId: string, cardId: string, value: string) => void;
  /** Agent turn finished (stream_end resuming:false); cards become tappable. */
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
  /** Resolve Shot a Memory → Shot a+1 continuous control; null hides switch. */
  getNextContinuous?: (
    review: MemoryReview,
  ) => NextContinuousControl | null;
}

/**
 * Render a single message. Following agent-chat-ui: user turns are a rounded
 * "pill" right-aligned with a muted fill; assistant turns render as bare
 * markdown so prose/code read like a document rather than a chat bubble.
 * Each turn fades+slides in for a touch of motion polish.
 *
 * Trace rows are not rendered in the thread UI.
 */
export function MessageBubble({
  message,
  onAnswerQuestion,
  questionsReady = true,
  onMemoryReviewAction,
  memoryAssets = [],
  onCreateMemoryAsset,
  getNextContinuous,
}: MessageBubbleProps) {
  const baseAnim = "animate-in fade-in-0 slide-in-from-bottom-1 duration-300";
  if (message.kind === "trace") {
    return null;
  }

  if (message.role === "user") {
    const images = message.images ?? [];
    const hasImages = images.length > 0;
    const hasText = message.content.trim().length > 0;
    return (
      <div
        className={cn(
          "group ml-auto flex max-w-[min(85%,36rem)] flex-col items-end gap-1.5",
          baseAnim,
        )}
      >
        {hasImages ? <UserImages images={images} /> : null}
        {hasText ? (
          <p
            className={cn(
              "ml-auto w-fit rounded-[18px] bg-secondary/70 px-4 py-2",
              "text-left text-[15px]/[1.8] whitespace-pre-wrap break-words",
            )}
          >
            {message.content}
          </p>
        ) : null}
      </div>
    );
  }

  const empty = message.content.trim().length === 0;
  const showTypingDots = message.turnWaiting === true && empty;
  const videos = message.videos ?? [];
  const hasVideos = videos.length > 0;
  const hasQuestions = (message.questions?.length ?? 0) > 0;

  // message.questions = [
  //   {
  //     id: "1",
  //     question: "What is the capital of France?",
  //     options: [{ label: "Paris" }, { label: "London" }, { label: "Berlin" }],
  //     allowCustom: false,
  //     answered: null,
  //   },
  // ];

  return (
    <div
      className={cn("w-full text-sm", baseAnim)}
      style={{ lineHeight: "var(--cjk-line-height)" }}
    >
      {showTypingDots ? (
        <TypingDots />
      ) : (
        <>
          {!empty ? <MarkdownText>{message.content}</MarkdownText> : null}
          {hasVideos ? <AssistantVideos videos={videos} /> : null}
          {hasQuestions && !message.isStreaming ? (
            <QuestionCards
              cards={message.questions!}
              disabled={!questionsReady}
              onAnswer={(cardId, value) =>
                onAnswerQuestion?.(message.id, cardId, value)
              }
            />
          ) : null}
          {message.memoryReview ? (
            <MemoryReviewCard
              review={message.memoryReview}
              memoryAssets={memoryAssets}
              onCreateMemoryAsset={onCreateMemoryAsset}
              nextContinuous={
                getNextContinuous?.(message.memoryReview) ?? null
              }
              onApprove={
                onMemoryReviewAction
                  ? (retainedMemoryIds) =>
                      onMemoryReviewAction(
                        message.memoryReview!,
                        "approve",
                        undefined,
                        undefined,
                        undefined,
                        retainedMemoryIds,
                      )
                  : undefined
              }
              onReselect={
                onMemoryReviewAction
                  ? (memoryId) =>
                      onMemoryReviewAction(
                        message.memoryReview!,
                        "reselect",
                        memoryId,
                      )
                  : undefined
              }
              onManualSelect={
                onMemoryReviewAction
                  ? (memoryId, timestampSec) =>
                      onMemoryReviewAction(
                        message.memoryReview!,
                        "manual_select",
                        memoryId,
                        timestampSec,
                      )
                  : undefined
              }
              onSelectMode={
                onMemoryReviewAction
                  ? (selectionMode) =>
                      onMemoryReviewAction(
                        message.memoryReview!,
                        "select_mode",
                        undefined,
                        undefined,
                        selectionMode,
                      )
                  : undefined
              }
            />
          ) : null}
          {message.isStreaming && !empty ? <StreamCursor /> : null}
        </>
      )}
    </div>
  );
}

/**
 * Right-aligned preview row for images attached to a user turn.
 *
 * Visual follows agent-chat-ui: a single wrapping row of fixed-size square
 * thumbnails that stay modest next to the text pill regardless of how many
 * images are attached.
 *
 * The URL is expected to be a self-contained ``data:`` URL (the Composer
 * hands the normalized base64 payload to the optimistic bubble so that the
 * preview survives React StrictMode double-mount — blob URLs would be
 * revoked by the Composer's cleanup before remount). Historical replays
 * have no URL (the backend strips data URLs before persisting), so we
 * render a labelled placeholder tile instead of a broken ``<img>``.
 */
function UserImages({ images }: { images: UIImage[] }) {
  const { t } = useTranslation();
  // Only real-URL images can open in the lightbox; historical-replay
  // placeholders (no URL) have nothing to zoom into.
  const viewable = images
    .map((img, i) => ({ img, i }))
    .filter(({ img }) => typeof img.url === "string" && img.url.length > 0);
  const viewableImages = viewable.map(({ img }) => img);
  const originalToViewable = new Map<number, number>(
    viewable.map(({ i }, v) => [i, v]),
  );

  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  return (
    <>
      <div className="ml-auto flex flex-wrap items-end justify-end gap-2">
        {images.map((img, i) => (
          <UserImageCell
            key={`${img.url ?? "placeholder"}-${i}`}
            image={img}
            placeholderLabel={t("message.imageAttachment")}
            openLabel={t("lightbox.open")}
            onOpen={
              originalToViewable.has(i)
                ? () => setLightboxIndex(originalToViewable.get(i)!)
                : undefined
            }
          />
        ))}
      </div>
      <ImageLightbox
        images={viewableImages}
        index={lightboxIndex}
        onIndexChange={setLightboxIndex}
        onOpenChange={(open) => {
          if (!open) setLightboxIndex(null);
        }}
      />
    </>
  );
}

function UserImageCell({
  image,
  placeholderLabel,
  openLabel,
  onOpen,
}: {
  image: UIImage;
  placeholderLabel: string;
  openLabel: string;
  onOpen?: () => void;
}) {
  const hasUrl = typeof image.url === "string" && image.url.length > 0;
  const tileClasses = cn(
    "relative h-24 w-24 overflow-hidden rounded-[14px] border border-border/60 bg-muted/40",
    "shadow-[0_6px_18px_-14px_rgba(0,0,0,0.45)]",
  );

  if (hasUrl && onOpen) {
    return (
      <button
        type="button"
        onClick={onOpen}
        aria-label={image.name ? `${openLabel}: ${image.name}` : openLabel}
        title={image.name ?? undefined}
        className={cn(
          tileClasses,
          "cursor-zoom-in transition-transform duration-150 motion-reduce:transition-none",
          "hover:scale-[1.02] hover:ring-2 hover:ring-primary/30",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50",
        )}
      >
        <img
          src={image.url}
          alt={image.name ?? ""}
          loading="lazy"
          decoding="async"
          draggable={false}
          className="h-full w-full object-cover"
        />
      </button>
    );
  }

  return (
    <div className={tileClasses} title={image.name ?? undefined}>
      <div
        className="flex h-full w-full flex-col items-center justify-center gap-1 px-2 text-[11px] text-muted-foreground"
        aria-label={placeholderLabel}
      >
        <ImageIcon className="h-4 w-4 flex-none" aria-hidden />
        <span className="line-clamp-2 text-center leading-tight">
          {image.name ?? placeholderLabel}
        </span>
      </div>
    </div>
  );
}

function AssistantVideos({ videos }: { videos: UIVideo[] }) {
  return (
    <div className="mt-3 flex w-full flex-col gap-3">
      {videos.map((video, index) => (
        <figure
          key={`${video.url}-${index}`}
          className={cn(
            "overflow-hidden rounded-2xl border border-border/60 bg-card/80",
            "shadow-[0_18px_45px_-32px_rgba(0,0,0,0.5)]",
          )}
        >
          <video
            controls
            playsInline
            preload="metadata"
            src={video.url}
            className="block max-h-[26rem] w-full bg-black"
          />
          {video.name ? (
            <figcaption className="border-t border-border/50 px-3 py-2 text-xs text-muted-foreground">
              {video.name}
            </figcaption>
          ) : null}
        </figure>
      ))}
    </div>
  );
}

/** Blinking cursor appended at the end of streaming text. */
function StreamCursor() {
  const { t } = useTranslation();
  return (
    <span
      aria-label={t("message.streaming")}
      className={cn(
        "ml-0.5 inline-block h-[1em] w-[3px] translate-y-[2px] align-middle",
        "rounded-sm bg-foreground/70 animate-pulse",
      )}
    />
  );
}

/** Pre-token-arrival placeholder: three bouncing dots. */
function TypingDots() {
  const { t } = useTranslation();
  const label = t("message.assistantTyping");
  return (
    <div
      aria-label={label}
      className="flex items-center gap-3 py-2 text-xs text-muted-foreground"
    >
      <style>{`
        @keyframes thinking-shimmer {
          0%   { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
      `}</style>
      <span className="shrink-0">{label}</span>
      <div
        className="h-1.5 min-w-20 max-w-44 flex-1 rounded-full"
        style={{
          background:
            "linear-gradient(90deg, transparent 0%, hsl(var(--foreground) / 0.03) 15%, hsl(var(--foreground) / 0.10) 50%, hsl(var(--foreground) / 0.03) 85%, transparent 100%)",
          backgroundSize: "200% 100%",
          animation: "thinking-shimmer 1.8s ease-in-out infinite",
        }}
      />
    </div>
  );
}
