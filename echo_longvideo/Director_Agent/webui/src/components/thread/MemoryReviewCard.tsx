import {
  Check,
  Film,
  HelpCircle,
  LoaderCircle,
  Plus,
  RefreshCw,
  Sparkles,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { ImageLightbox } from "@/components/ImageLightbox";
import { CompactAudioPlayer } from "@/components/thread/CompactAudioPlayer";
import { ShotReferenceDialog } from "@/components/workplace/ShotMemorySlots";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  memoryDisplayName,
  type MemoryReview,
  type MemorySelection,
  type MemoryWorkspaceAsset,
  type ShotMemoryAssetCreate,
  type UIImage,
} from "@/lib/types";

export type NextContinuousControl = {
  /** Display shot number for a+1 (e.g. 3 when this review is Shot 2). */
  nextShotId: number;
  enabled: boolean;
  /** Locked after the next shot has started/finished generation. */
  disabled?: boolean;
  onToggle: (enabled: boolean) => void;
};

interface MemoryReviewCardProps {
  review: MemoryReview;
  onApprove?: (retainedMemoryIds: string[]) => void;
  onReselect?: (memoryId: string) => void;
  onManualSelect?: (
    memoryId: string,
    timestampSec: number,
  ) => void | Promise<void>;
  onSelectMode?: (mode: "manual" | "vlm") => void | Promise<void>;
  memoryAssets?: MemoryWorkspaceAsset[];
  onCreateMemoryAsset?: (
    shotId: number,
    asset: ShotMemoryAssetCreate,
  ) => Promise<void>;
  busyAction?:
    | "approve"
    | "reselect"
    | "manual_select"
    | "select_mode"
    | null;
  /** Shot a Memory footer controls Shot a+1 continuous mode. */
  nextContinuous?: NextContinuousControl | null;
}

function initialRetainedMemoryIds(review: MemoryReview): Set<string> {
  const available = new Set(
    review.selections.map((selection) => selection.memory_id),
  );
  const defaults =
    review.retained_memory_ids !== undefined
      ? review.retained_memory_ids
      : review.selection_mode === "manual"
        ? (review.manual_selected_ids ?? [])
        : review.selections.map((selection) => selection.memory_id);
  return new Set(defaults.filter((memoryId) => available.has(memoryId)));
}

export function MemoryReviewCard({
  review,
  onApprove,
  onReselect,
  onManualSelect,
  onSelectMode,
  memoryAssets = [],
  onCreateMemoryAsset,
  busyAction = null,
  nextContinuous = null,
}: MemoryReviewCardProps) {
  const canAct =
    review.status === "awaiting_review" || review.status === "error";
  const awaitingMethod = review.status === "awaiting_method";
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);
  const [manualSelection, setManualSelection] =
    useState<MemorySelection | null>(null);
  const [memoryBuilderOpen, setMemoryBuilderOpen] = useState(false);
  const [referenceDialogOpen, setReferenceDialogOpen] = useState(false);
  const [retainedIds, setRetainedIds] = useState<Set<string>>(() =>
    initialRetainedMemoryIds(review),
  );
  const selectionIdsKey = review.selections
    .map((selection) => selection.memory_id)
    .join("\u0000");
  const retainedIdsKey = review.retained_memory_ids?.join("\u0000") ?? "";
  const manualIdsKey = review.manual_selected_ids?.join("\u0000") ?? "";

  useEffect(() => {
    setRetainedIds(initialRetainedMemoryIds(review));
  }, [
    review.review_id,
    review.attempt,
    review.selection_mode,
    selectionIdsKey,
    retainedIdsKey,
    manualIdsKey,
  ]);

  useEffect(() => {
    setMemoryBuilderOpen(false);
    setReferenceDialogOpen(false);
  }, [review.review_id]);

  const createdAssets = useMemo(
    () =>
      memoryAssets.filter(
        (asset) =>
          asset.source === "local" &&
          asset.provenance?.shot_id === review.shot_id,
      ),
    [memoryAssets, review.shot_id],
  );
  const showMemoryBuilder =
    !awaitingMethod || memoryBuilderOpen || createdAssets.length > 0;

  const previewImages = useMemo<UIImage[]>(
    () =>
      [
        ...review.selections
          .filter((s) => Boolean(s.image?.url))
          .map((s) => ({
            url: s.image.url,
            name: s.image.name ?? memoryDisplayName(s),
          })),
        ...createdAssets
          .filter((asset) => Boolean(asset.image?.url))
          .map((asset) => ({
            url: asset.image!.url,
            name: asset.image!.name ?? asset.display_name,
          })),
      ],
    [createdAssets, review.selections],
  );

  const previewIndexByMemoryId = useMemo(() => {
    const map = new Map<string, number>();
    let i = 0;
    for (const s of review.selections) {
      if (!s.image?.url) continue;
      map.set(s.memory_id, i);
      i += 1;
    }
    return map;
  }, [review.selections]);
  const createdAssetPreviewIndex = useMemo(() => {
    const map = new Map<string, number>();
    let index = review.selections.filter((item) => Boolean(item.image?.url)).length;
    for (const asset of createdAssets) {
      if (!asset.image?.url) continue;
      map.set(asset.asset_id, index);
      index += 1;
    }
    return map;
  }, [createdAssets, review.selections]);
  const manuallySelectedIds = new Set(review.manual_selected_ids ?? []);
  const retainedMemoryIds = review.selections
    .map((selection) => selection.memory_id)
    .filter((memoryId) => retainedIds.has(memoryId));

  const renderSelection = (selection: MemorySelection) => (
    <MemorySelectionTile
      key={`${selection.kind}:${selection.memory_id}`}
      selection={selection}
      selectionMode={review.selection_mode ?? null}
      manuallySelected={manuallySelectedIds.has(selection.memory_id)}
      retained={retainedIds.has(selection.memory_id)}
      canToggle={canAct && busyAction === null}
      canReselect={canAct && busyAction === null && Boolean(onReselect)}
      canSelectManually={
        canAct &&
        busyAction === null &&
        Boolean(onManualSelect) &&
        Boolean(review.source_video?.url)
      }
      onReselect={() => onReselect?.(selection.memory_id)}
      onSelectManually={() => setManualSelection(selection)}
      onToggleRetained={() =>
        setRetainedIds((current) => {
          const next = new Set(current);
          if (next.has(selection.memory_id)) {
            next.delete(selection.memory_id);
          } else {
            next.add(selection.memory_id);
          }
          return next;
        })
      }
      onPreview={
        previewIndexByMemoryId.has(selection.memory_id)
          ? () =>
              setLightboxIndex(
                previewIndexByMemoryId.get(selection.memory_id)!,
              )
          : undefined
      }
    />
  );

  return (
    <section
      aria-label={`Shot ${review.shot_id} memory review`}
      className="mt-3 overflow-hidden rounded-2xl border border-border/70 bg-card/80 shadow-sm"
    >
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border/60 px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span className="grid size-8 place-items-center rounded-xl bg-foreground/[0.06] text-foreground/60">
            <Sparkles className="size-4" aria-hidden />
          </span>
          <div>
            <h3 className="font-semibold tracking-tight">
              Shot {review.shot_id} · Memory Review
            </h3>
            <p className="text-xs text-muted-foreground">
              <span>Pass {review.attempt}</span> · Echo Director ·{" "}
              {review.candidate_count} candidates
            </p>
          </div>
        </div>

        <StatusPill review={review} />
        <div className="text-xs text-muted-foreground">
          {awaitingMethod
            ? "Choose whether this shot should create any new Memory."
            : "Keep any references that help the next shot, or continue with none."}
        </div>
      </header>

      {awaitingMethod && !showMemoryBuilder ? (
        <div className="p-4">
          <div className="rounded-xl border border-border/70 bg-muted/20 p-4">
            <h4 className="text-sm font-semibold">
              Create new Memory from this video?
            </h4>
            <p className="mt-1 max-w-2xl text-xs leading-relaxed text-muted-foreground">
              You can add exact frames yourself or let VLM extract several
              profiled character, scene, and other references. This step is
              optional.
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              <Button
                type="button"
                disabled={busyAction !== null}
                onClick={() => setMemoryBuilderOpen(true)}
              >
                <Plus className="mr-2 size-4" />
                Create New Memory
              </Button>
              <Button
                type="button"
                variant="outline"
                disabled={busyAction !== null || !onApprove}
                onClick={() => onApprove?.([])}
              >
                Continue Without New Memory
              </Button>
            </div>
          </div>
        </div>
      ) : review.selections.length > 0 || createdAssets.length > 0 || awaitingMethod ? (
        <div className="p-4">
          <section aria-label="Memory references">
            <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
              <div>
                <h4 className="text-sm font-semibold">New Memory</h4>
                <p className="text-xs text-muted-foreground">
                  Add a frame and finish its type and profile here, or let VLM
                  prepare a starting set.
                </p>
              </div>
              {awaitingMethod ? (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={busyAction !== null || !onSelectMode}
                  onClick={() => onSelectMode?.("vlm")}
                >
                  <Sparkles className="mr-2 size-3.5" />
                  Extract with VLM
                </Button>
              ) : null}
            </div>
            <div className="mb-2">
              <p className="text-xs text-muted-foreground">
                Keep any number of references, including none. Type labels are
                descriptive, not requirements.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
              {review.selections.map(renderSelection)}
              {createdAssets.map((asset) => (
                <CreatedMemoryTile
                  key={asset.asset_id}
                  asset={asset}
                  onPreview={
                    createdAssetPreviewIndex.has(asset.asset_id)
                      ? () => setLightboxIndex(createdAssetPreviewIndex.get(asset.asset_id)!)
                      : undefined
                  }
                />
              ))}
              {review.source_video?.url && onCreateMemoryAsset ? (
                <button
                  type="button"
                  aria-label="Add memory from video"
                  disabled={busyAction !== null}
                  onClick={() => setReferenceDialogOpen(true)}
                  className="flex min-h-64 flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border bg-muted/15 p-5 text-center text-muted-foreground transition-colors hover:border-foreground/35 hover:bg-muted/30 hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <span className="grid size-11 place-items-center rounded-full border border-border bg-background">
                    <Plus className="size-5" />
                  </span>
                  <span>
                    <strong className="block text-sm text-foreground">Add Memory</strong>
                    <span className="mt-1 block text-xs">Choose a frame, type, and profile</span>
                  </span>
                </button>
              ) : null}
            </div>
          </section>
        </div>
      ) : null}

      {review.error ? (
        <p
          role="alert"
          className="mx-4 mb-3 rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive"
        >
          {review.error}
        </p>
      ) : null}

      {canAct || nextContinuous || (awaitingMethod && showMemoryBuilder) ? (
        <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-border/60 bg-background/45 px-4 py-3">
          {nextContinuous ? (
            <div className="flex items-center gap-1.5">
              <span className="text-[12px] text-foreground/70">
                Shot {nextContinuous.nextShotId} continuous first frame
              </span>
              <Switch
                checked={nextContinuous.enabled}
                disabled={nextContinuous.disabled}
                onCheckedChange={nextContinuous.onToggle}
                aria-label={`Enable continuous first frame for Shot ${nextContinuous.nextShotId}`}
              />
              <TooltipProvider delayDuration={200}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      className="inline-flex h-5 w-5 items-center justify-center rounded-full text-muted-foreground/40 transition-colors hover:text-foreground/60"
                      aria-label="Continuous first frame details"
                    >
                      <HelpCircle className="h-3.5 w-3.5" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="top" className="max-w-[220px] text-xs">
                    Use the previous shot's final frame as the next shot's opening frame.
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          ) : (
            <span />
          )}
          {(canAct || (awaitingMethod && showMemoryBuilder)) && onApprove ? (
            <div className="ml-auto flex flex-col items-end gap-1">
              <span className="text-xs text-muted-foreground">
                {retainedMemoryIds.length + createdAssets.length} reference
                {retainedMemoryIds.length + createdAssets.length === 1 ? "" : "s"} ready
              </span>
              <Button
                type="button"
                disabled={busyAction !== null}
                onClick={() => onApprove(retainedMemoryIds)}
              >
                <Check className="mr-2 size-4" />
                Continue
              </Button>
            </div>
          ) : null}
        </footer>
      ) : null}

      <ImageLightbox
        images={previewImages}
        index={lightboxIndex}
        onIndexChange={setLightboxIndex}
        onOpenChange={(open) => {
          if (!open) setLightboxIndex(null);
        }}
      />
      <ManualFrameDialog
        open={manualSelection !== null}
        sourceVideoUrl={review.source_video?.url ?? ""}
        selection={manualSelection}
        onOpenChange={(open) => {
          if (!open) setManualSelection(null);
        }}
        onConfirm={async (timestampSec) => {
          if (!manualSelection || !onManualSelect) return;
          await onManualSelect(manualSelection.memory_id, timestampSec);
          setManualSelection(null);
        }}
      />
      {review.source_video?.url && onCreateMemoryAsset ? (
        <ShotReferenceDialog
          open={referenceDialogOpen}
          shotId={review.shot_id}
          videoUrl={review.source_video.url}
          onOpenChange={setReferenceDialogOpen}
          onSave={(asset) => onCreateMemoryAsset(review.shot_id, asset)}
        />
      ) : null}
    </section>
  );
}

function StatusPill({ review }: { review: MemoryReview }) {
  if (review.status === "awaiting_method") {
    return (
      <span className="rounded-full bg-muted px-2.5 py-1 text-xs font-medium text-foreground/70">
        Memory is optional
      </span>
    );
  }
  if (review.status === "selecting" || review.status === "reselecting") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1 text-xs font-medium text-foreground/70">
        <LoaderCircle className="size-3.5 animate-spin" aria-hidden />
        Selecting memories
      </span>
    );
  }
  if (review.status === "approved") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-2.5 py-1 text-xs font-medium text-emerald-700 dark:text-emerald-300">
        <Check className="size-3.5" />
        Approved and stored
      </span>
    );
  }
  if (review.status === "error") {
    return (
      <span className="rounded-full bg-destructive/10 px-2.5 py-1 text-xs text-destructive">
        Selection failed · Retry available
      </span>
    );
  }
  return (
    <span className="rounded-full bg-amber-500/15 px-2.5 py-1 text-xs font-medium text-amber-700 dark:text-amber-300">
      Awaiting approval
    </span>
  );
}

function CreatedMemoryTile({
  asset,
  onPreview,
}: {
  asset: MemoryWorkspaceAsset;
  onPreview?: () => void;
}) {
  const typeLabel = asset.reference_type
    ? asset.reference_type.charAt(0).toUpperCase() + asset.reference_type.slice(1)
    : "Reference";
  return (
    <article className="overflow-hidden rounded-xl border border-emerald-500/35 bg-card/80">
      <div className="relative aspect-video overflow-hidden bg-muted">
        {asset.image?.url ? (
          <button
            type="button"
            onClick={onPreview}
            disabled={!onPreview}
            aria-label={`Preview ${asset.display_name}`}
            className="group relative block size-full text-left disabled:cursor-default"
          >
            <img
              src={asset.image.url}
              alt={asset.display_name}
              className="size-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
            />
          </button>
        ) : (
          <div className="grid size-full place-items-center text-xs text-muted-foreground">
            Audio reference
          </div>
        )}
        <span className="pointer-events-none absolute right-2 top-2 rounded-full bg-black/65 px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-white">
          {typeLabel}
        </span>
      </div>
      <div className="space-y-2 p-3">
        <div>
          <strong className="block truncate text-xs">{asset.display_name}</strong>
          <span className="text-[11px] text-emerald-700 dark:text-emerald-300">
            Saved to Memory Bank
          </span>
        </div>
        {asset.profile_text ? (
          <p className="line-clamp-3 text-xs leading-relaxed text-muted-foreground">
            {asset.profile_text}
          </p>
        ) : (
          <p className="text-xs text-muted-foreground">Profile unavailable</p>
        )}
        {asset.audio?.url ? (
          <CompactAudioPlayer
            src={asset.audio.url}
            label={`${asset.display_name} memory audio`}
            className="w-full"
          />
        ) : null}
      </div>
    </article>
  );
}

function MemorySelectionTile({
  selection,
  selectionMode,
  manuallySelected,
  retained,
  canToggle,
  canReselect,
  canSelectManually,
  onReselect,
  onSelectManually,
  onToggleRetained,
  onPreview,
}: {
  selection: MemorySelection;
  selectionMode: "manual" | "vlm" | null;
  manuallySelected: boolean;
  retained: boolean;
  canToggle: boolean;
  canReselect: boolean;
  canSelectManually: boolean;
  onReselect: () => void;
  onSelectManually: () => void;
  onToggleRetained: () => void;
  onPreview?: () => void;
}) {
  const label = memoryDisplayName(selection);
  const kindLabel =
    selection.kind === "character"
      ? "Character"
      : selection.kind === "previous_shot"
        ? "Scene"
        : "Reference";
  return (
    <article
      className={`overflow-hidden rounded-xl border bg-card/80 transition-opacity ${
        retained ? "border-border/70" : "border-border/40 opacity-55"
      }`}
    >
      <div className="relative aspect-video overflow-hidden bg-muted">
        <button
          type="button"
          onClick={onPreview}
          disabled={!onPreview}
          aria-label={`Preview ${label} memory frame`}
          className="group relative block size-full text-left disabled:cursor-default"
        >
          <img
            src={selection.image.url}
            alt={`Selected ${label} memory frame`}
            className="size-full object-cover transition-transform duration-300 group-hover:scale-[1.02] group-disabled:group-hover:scale-100"
          />
          <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between bg-gradient-to-t from-black/80 to-transparent px-3 pb-2 pt-8 text-white">
            <strong className="min-w-0 truncate text-xs tracking-wide">
              {label}
            </strong>
          </div>
        </button>
        <span className="pointer-events-none absolute right-2 top-2 rounded-full bg-black/65 px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-white">
          {kindLabel}
        </span>
      </div>
      <div className="space-y-2.5 p-3">
        {selectionMode === "manual" ? (
          <p className="text-[11px] font-medium text-foreground/60">
            {!retained
              ? "Not included"
              : manuallySelected
                ? "Frame selected"
                : "Using the current candidate"}
          </p>
        ) : null}
        <p className="line-clamp-3 text-xs leading-relaxed text-muted-foreground">
          {selection.reasoning}
        </p>
        {selection.audio?.url ? (
          <CompactAudioPlayer
            src={selection.audio.url}
            label={`${label} memory audio`}
            className="w-full"
          />
        ) : (
          <p className="text-[11px] text-muted-foreground">No reference audio</p>
        )}
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="w-full"
          disabled={!canToggle}
          aria-pressed={retained}
          onClick={onToggleRetained}
        >
          {retained ? (
            <X className="mr-2 size-3.5" />
          ) : (
            <Check className="mr-2 size-3.5" />
          )}
          {retained ? "Remove from Memory" : "Keep in Memory"}
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="w-full"
          disabled={!canReselect}
          onClick={onReselect}
        >
          <RefreshCw className="mr-2 size-3.5" />
          {selectionMode === "manual"
            ? "Ask VLM for This Memory"
            : "Reselect This Memory"}
        </Button>
        {canSelectManually ? (
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="w-full"
            onClick={onSelectManually}
          >
            <Film className="mr-2 size-3.5" />
            Choose from Video
          </Button>
        ) : null}
      </div>
    </article>
  );
}

function formatTime(seconds: number): string {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${minutes}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function ManualFrameDialog({
  open,
  sourceVideoUrl,
  selection,
  onOpenChange,
  onConfirm,
}: {
  open: boolean;
  sourceVideoUrl: string;
  selection: MemorySelection | null;
  onOpenChange: (open: boolean) => void;
  onConfirm: (timestampSec: number) => void | Promise<void>;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [duration, setDuration] = useState(0);
  const [timestampSec, setTimestampSec] = useState(0);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!open || !selection) return;
    setDuration(0);
    setTimestampSec(Math.max(0, selection.timestamp_sec));
    setSubmitting(false);
  }, [open, selection]);

  const seek = (next: number) => {
    const value = Math.min(Math.max(next, 0), duration || next);
    setTimestampSec(value);
    if (videoRef.current) {
      videoRef.current.currentTime = value;
      videoRef.current.pause();
    }
  };

  const label = selection ? memoryDisplayName(selection) : "Memory";
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>Choose Memory Frame</DialogTitle>
          <DialogDescription>
            Drag through Shot {selection?.source_shot_id ?? ""} and choose the
            frame to use for {label}. Its reference audio will stay unchanged.
          </DialogDescription>
        </DialogHeader>

        <div className="overflow-hidden rounded-lg border border-border bg-black">
          {sourceVideoUrl ? (
            <video
              ref={videoRef}
              src={sourceVideoUrl}
              controls
              playsInline
              preload="metadata"
              className="aspect-video w-full object-contain"
              onLoadedMetadata={(event) => {
                const nextDuration = Number(event.currentTarget.duration || 0);
                setDuration(nextDuration);
                const initial = Math.min(
                  Math.max(selection?.timestamp_sec ?? 0, 0),
                  nextDuration || Number.POSITIVE_INFINITY,
                );
                event.currentTarget.currentTime = initial;
                setTimestampSec(initial);
              }}
              onTimeUpdate={(event) =>
                setTimestampSec(event.currentTarget.currentTime)
              }
              onSeeked={(event) =>
                setTimestampSec(event.currentTarget.currentTime)
              }
            />
          ) : null}
        </div>

        <div className="space-y-2">
          <input
            aria-label="Memory frame time"
            type="range"
            min={0}
            max={Math.max(duration, 0.01)}
            step={0.01}
            value={Math.min(timestampSec, Math.max(duration, 0.01))}
            disabled={duration <= 0 || submitting}
            onChange={(event) => seek(Number(event.currentTarget.value))}
            className="h-2 w-full cursor-pointer accent-foreground disabled:cursor-not-allowed"
          />
          <div className="flex justify-between text-xs tabular-nums text-muted-foreground">
            <span>Selected {formatTime(timestampSec)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            disabled={submitting}
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            type="button"
            disabled={duration <= 0 || submitting}
            onClick={async () => {
              setSubmitting(true);
              try {
                await onConfirm(timestampSec);
              } finally {
                setSubmitting(false);
              }
            }}
          >
            {submitting ? (
              <LoaderCircle className="mr-2 size-4 animate-spin" />
            ) : (
              <Check className="mr-2 size-4" />
            )}
            Use This Frame
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
