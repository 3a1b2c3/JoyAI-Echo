import { GripVertical, Images, LoaderCircle, Music2, Plus, Save, Scissors, X } from "lucide-react";
import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import type { DragEvent } from "react";

import { ImageLightbox } from "@/components/ImageLightbox";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type {
  GenerationMemory,
  MemoryReferenceType,
  MemorySlotReference,
  MemoryWorkspaceAsset,
  ShotMemoryAssetCreate,
  UIImage,
  WorkplaceShot,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  MEMORY_SLOT_DRAG_TYPE,
  MemoryAudioWaveform,
  readMemoryAssetDrag,
} from "./MemoryBankBoard";

const MAX_SLOTS = 7;

type DraftSlot = {
  image_asset_id?: string;
  audio_asset_id?: string;
};

interface ShotMemorySlotsProps {
  assets: MemoryWorkspaceAsset[];
  shot: WorkplaceShot;
  conditionImage?: UIImage | null;
  busy?: boolean;
  onApplySlots: (
    shotId: number,
    slots: MemorySlotReference[],
  ) => Promise<void>;
  onCreateAsset?: (
    shotId: number,
    asset: ShotMemoryAssetCreate,
  ) => Promise<void>;
}

export interface ShotMemorySlotsHandle {
  /** Persist the visible draft before a generation or revision is submitted. */
  applyPending: () => Promise<void>;
}

function appliedAssetId(
  memory: GenerationMemory,
  assets: MemoryWorkspaceAsset[],
): string | null {
  if (memory.workspace_asset_id) return memory.workspace_asset_id;
  const sourceShotId = memory.metadata.source_shot_id;
  const frameIndex = memory.metadata.frame_index;
  return (
    assets.find(
      (asset) =>
        asset.source === "automatic" &&
        asset.memory_id === memory.id &&
        (sourceShotId == null || asset.source_shot_id === sourceShotId) &&
        (frameIndex == null || asset.frame_index === frameIndex),
    )?.asset_id ?? null
  );
}

function appliedRefsForShot(
  shot: WorkplaceShot,
  assets: MemoryWorkspaceAsset[],
): MemorySlotReference[] {
  if (shot.approved_memory_slot_refs) return shot.approved_memory_slot_refs;
  return (shot.memory_slots ?? shot.generation_memories ?? []).flatMap(
    (memory) => {
      const imageAssetId = appliedAssetId(memory, assets);
      if (!imageAssetId) return [];
      return [{
        image_asset_id: imageAssetId,
        ...(memory.metadata.audio_workspace_asset_id
          ? { audio_asset_id: memory.metadata.audio_workspace_asset_id }
          : {}),
      }];
    },
  );
}

function recommendedRefsForShot(
  shot: WorkplaceShot,
  assets: MemoryWorkspaceAsset[],
): MemorySlotReference[] {
  if (shot.recommended_memory_slot_refs !== undefined) {
    return shot.recommended_memory_slot_refs;
  }
  return (shot.recommended_memory_slots ?? []).flatMap((memory) => {
    const imageAssetId = appliedAssetId(memory, assets);
    return imageAssetId ? [{ image_asset_id: imageAssetId }] : [];
  });
}

function refsAsDraft(refs: MemorySlotReference[]): DraftSlot[] {
  return refs.slice(0, MAX_SLOTS).flatMap((ref) => {
    const imageAssetId = ref.image_asset_id ?? ref.asset_id;
    return imageAssetId || ref.audio_asset_id
      ? [{
          ...(imageAssetId ? { image_asset_id: imageAssetId } : {}),
          ...(ref.audio_asset_id
            ? { audio_asset_id: ref.audio_asset_id }
            : {}),
        }]
      : [];
  });
}

function compactRefs(slots: DraftSlot[]): MemorySlotReference[] {
  return slots.flatMap((slot) =>
    slot.image_asset_id
      ? [{
          image_asset_id: slot.image_asset_id,
          ...(slot.audio_asset_id
            ? { audio_asset_id: slot.audio_asset_id }
            : {}),
        }]
      : [],
  );
}

function refsSignature(refs: MemorySlotReference[]): string {
  return refs
    .map((ref) =>
      `${ref.image_asset_id ?? ref.asset_id ?? ""}:${ref.audio_asset_id ?? ""}`,
    )
    .join("|");
}

export const ShotMemorySlots = forwardRef<
  ShotMemorySlotsHandle,
  ShotMemorySlotsProps
>(function ShotMemorySlots({
  assets,
  shot,
  conditionImage = null,
  busy = false,
  onApplySlots,
  onCreateAsset,
}, ref) {
  const appliedRefs = useMemo(
    () => appliedRefsForShot(shot, assets),
    [assets, shot],
  );
  const recommendedRefs = useMemo(
    () => recommendedRefsForShot(shot, assets),
    [assets, shot],
  );
  const serverRefs = shot.memory_slots_configured
    ? appliedRefs
    : recommendedRefs;
  const serverSignature = refsSignature(serverRefs);
  const [slots, setSlots] = useState<DraftSlot[]>(() => refsAsDraft(serverRefs));
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);
  const [conditionLightboxOpen, setConditionLightboxOpen] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [referenceDialogOpen, setReferenceDialogOpen] = useState(false);
  const sourceVideo = shot.video ?? null;
  const savedFromShot = assets.filter(
    (asset) => asset.provenance?.shot_id === shot.shot_id,
  ).length;

  useEffect(() => {
    setSlots(refsAsDraft(serverRefs));
    setStatus(null);
    // Reset only when this shot's server-side memory draft changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shot.shot_id, serverSignature, shot.memory_slots_configured]);

  const assetById = useMemo(
    () => new Map(assets.map((asset) => [asset.asset_id, asset])),
    [assets],
  );
  const previewImages = useMemo<UIImage[]>(
    () =>
      assets
        .filter((asset) => Boolean(asset.image?.url))
        .map((asset) => ({
          url: asset.image?.url,
          name: asset.image?.name ?? asset.display_name,
        })),
    [assets],
  );
  const previewIndexByAssetId = useMemo(
    () => new Map(
      assets
        .filter((asset) => Boolean(asset.image?.url))
        .map((asset, index) => [asset.asset_id, index]),
    ),
    [assets],
  );
  const draftRefs = compactRefs(slots);
  const dirty =
    !shot.memory_slots_configured ||
    refsSignature(draftRefs) !== refsSignature(appliedRefs);

  const setSlotMedia = (
    index: number,
    mediaType: "image" | "audio",
    assetId: string,
  ) => {
    setStatus(null);
    setSlots((current) => {
      if (
        mediaType === "image" &&
        current.some(
          (slot, slotIndex) =>
            slotIndex !== index && slot.image_asset_id === assetId,
        )
      ) {
        return current;
      }
      if (index === current.length) {
        if (current.length >= MAX_SLOTS) return current;
        return [...current, {
          [mediaType === "image" ? "image_asset_id" : "audio_asset_id"]:
            assetId,
        }];
      }
      if (index < 0 || index >= current.length) return current;
      const next = [...current];
      next[index] = {
        ...next[index],
        [mediaType === "image" ? "image_asset_id" : "audio_asset_id"]:
          assetId,
      };
      return next;
    });
  };

  const clearMedia = (index: number, mediaType: "image" | "audio") => {
    setSlots((current) => {
      const slot = current[index];
      if (!slot) return current;
      const nextSlot = { ...slot };
      delete nextSlot[mediaType === "image" ? "image_asset_id" : "audio_asset_id"];
      const next = [...current];
      if (nextSlot.image_asset_id || nextSlot.audio_asset_id) {
        next[index] = nextSlot;
      } else {
        next.splice(index, 1);
      }
      return next;
    });
  };

  const moveSlot = (sourceIndex: number, targetIndex: number) => {
    if (sourceIndex === targetIndex) return;
    setSlots((current) => {
      if (
        sourceIndex < 0 ||
        sourceIndex >= current.length ||
        targetIndex < 0 ||
        targetIndex > current.length
      ) {
        return current;
      }
      const next = [...current];
      const [slot] = next.splice(sourceIndex, 1);
      next.splice(Math.min(targetIndex, next.length), 0, slot);
      return next;
    });
  };

  const readDragPayload = (event: DragEvent) => {
    const slotText = event.dataTransfer.getData(MEMORY_SLOT_DRAG_TYPE);
    if (slotText) {
      const sourceIndex = Number(slotText);
      return Number.isInteger(sourceIndex)
        ? { type: "slot" as const, sourceIndex }
        : null;
    }
    const asset = readMemoryAssetDrag(event.dataTransfer);
    return asset ? { type: "asset" as const, ...asset } : null;
  };

  const dropAt = (event: DragEvent, targetIndex: number) => {
    event.preventDefault();
    const payload = readDragPayload(event);
    setDragOverIndex(null);
    if (!payload) return;
    if (payload.type === "slot") {
      moveSlot(payload.sourceIndex, targetIndex);
    } else {
      setSlotMedia(targetIndex, payload.mediaType, payload.assetId);
    }
  };

  const dragOver = (event: DragEvent, index: number) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = Array.from(event.dataTransfer.types).includes(
      MEMORY_SLOT_DRAG_TYPE,
    )
      ? "move"
      : "copy";
    setDragOverIndex(index);
  };

  const apply = async () => {
    if (!dirty) return;
    setStatus(null);
    try {
      await onApplySlots(shot.shot_id, draftRefs);
      setStatus("Memory slots applied.");
    } catch (error) {
      setStatus((error as Error).message);
      throw error;
    }
  };

  useImperativeHandle(ref, () => ({ applyPending: apply }));

  return (
    <section
      aria-label={`Memory slots for Shot ${shot.shot_id}`}
      className="mb-2 rounded-xl border border-border/50 bg-foreground/[0.018] p-2.5"
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-[10px] font-semibold text-foreground/70">
            Shot {shot.shot_id} Inputs
          </span>
          <span className="text-[9px] tabular-nums text-muted-foreground">
            {slots.length}/{MAX_SLOTS}
          </span>
          {!shot.memory_slots_configured && recommendedRefs.length > 0 ? (
            <span className="rounded bg-sky-500/10 px-1.5 py-0.5 text-[9px] text-sky-700 dark:text-sky-300">
              Agent draft
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-1.5">
          {sourceVideo?.url && onCreateAsset ? (
            <button
              type="button"
              disabled={busy}
              onClick={() => setReferenceDialogOpen(true)}
              className="inline-flex h-6 items-center gap-1 rounded-md border border-border px-2 text-[9px] text-muted-foreground hover:text-foreground disabled:opacity-35"
            >
              <Scissors className="size-2.5" aria-hidden />
              Save reference{savedFromShot > 0 ? ` · ${savedFromShot}` : ""}
            </button>
          ) : null}
          <button
            type="button"
            disabled={busy || !dirty}
            onClick={() => void apply().catch(() => undefined)}
            className="inline-flex h-6 items-center gap-1 rounded-md bg-foreground px-2 text-[9px] font-medium text-background disabled:opacity-35"
          >
            <Save className="size-2.5" aria-hidden />
            Apply
          </button>
        </div>
      </div>

      <div className="flex min-w-0 gap-2">
        <div className="w-24 shrink-0">
          <div className="mb-1 text-[8px] font-semibold uppercase tracking-wider text-muted-foreground/60">
            Condition
          </div>
          <div className="aspect-[4/3] overflow-hidden rounded-lg border border-border/70 bg-muted/40">
            {conditionImage?.url ? (
              <button
                type="button"
                aria-label={`Preview Shot ${shot.shot_id} condition image`}
                onClick={() => setConditionLightboxOpen(true)}
                className="size-full"
              >
                <img
                  src={conditionImage.url}
                  alt={`Shot ${shot.shot_id} condition`}
                  className="size-full object-cover"
                />
              </button>
            ) : (
              <div className="grid size-full place-items-center px-2 text-center text-[8px] text-muted-foreground">
                T2V · no condition
              </div>
            )}
          </div>
        </div>

        <div className="min-w-0 flex-1">
          <div className="mb-1 text-[8px] font-semibold uppercase tracking-wider text-muted-foreground/60">
            Memory
          </div>
          <div className="overflow-x-auto pb-1 scrollbar-thin">
        <ol aria-label={`Shot ${shot.shot_id} memory slots`} className="flex min-h-32 gap-2">
          {slots.map((slot, index) => {
            const imageAsset = slot.image_asset_id
              ? assetById.get(slot.image_asset_id)
              : undefined;
            const audioAsset = slot.audio_asset_id
              ? assetById.get(slot.audio_asset_id)
              : undefined;
            return (
              <li
                key={`${slot.image_asset_id ?? "_"}:${slot.audio_asset_id ?? "_"}:${index}`}
                data-testid={`shot-${shot.shot_id}-memory-slot-${index + 1}`}
                aria-label={`Shot ${shot.shot_id} memory slot ${index + 1}`}
                draggable
                onDragStart={(event) => {
                  event.dataTransfer.effectAllowed = "move";
                  event.dataTransfer.setData(MEMORY_SLOT_DRAG_TYPE, String(index));
                }}
                onDragEnd={() => setDragOverIndex(null)}
                onDragOver={(event) => dragOver(event, index)}
                onDragLeave={() =>
                  setDragOverIndex((current) => current === index ? null : current)
                }
                onDrop={(event) => dropAt(event, index)}
                className={cn(
                  "group/slot relative w-24 shrink-0 cursor-grab overflow-hidden rounded-lg border bg-background transition-all active:cursor-grabbing",
                  dragOverIndex === index
                    ? "-translate-y-0.5 border-sky-500 ring-2 ring-sky-500/20"
                    : "border-border/70",
                )}
              >
                <div className="pointer-events-none absolute left-1 top-1 z-10 flex items-center gap-0.5">
                  <span className="grid size-5 place-items-center rounded bg-black/60 text-[9px] font-semibold text-white">
                    {index + 1}
                  </span>
                  <span className="grid size-5 place-items-center rounded bg-black/60 text-white">
                    <GripVertical className="size-3" aria-hidden />
                  </span>
                </div>
                <button
                  type="button"
                  aria-label={`Clear Shot ${shot.shot_id} memory slot ${index + 1}`}
                  onClick={() =>
                    setSlots((current) =>
                      current.filter((_, currentIndex) => currentIndex !== index),
                    )
                  }
                  className="absolute right-1 top-1 z-10 grid size-5 place-items-center rounded bg-black/60 text-white opacity-0 group-hover/slot:opacity-100 focus:opacity-100"
                >
                  <X className="size-3" aria-hidden />
                </button>

                <div className="relative aspect-[4/3] border-b border-dashed border-border/70 bg-muted/50">
                  {imageAsset?.image?.url ? (
                    <>
                      <button
                        type="button"
                        aria-label={`Preview Shot ${shot.shot_id} slot ${index + 1} image`}
                        onClick={() =>
                          setLightboxIndex(
                            previewIndexByAssetId.get(imageAsset.asset_id) ?? 0,
                          )
                        }
                        className="block size-full overflow-hidden"
                      >
                        <img
                          src={imageAsset.image.url}
                          alt={imageAsset.display_name}
                          className="size-full object-cover"
                        />
                      </button>
                      <button
                        type="button"
                        aria-label={`Remove image from Shot ${shot.shot_id} slot ${index + 1}`}
                        onClick={() => clearMedia(index, "image")}
                        className="absolute bottom-1 right-1 grid size-5 place-items-center rounded bg-black/60 text-white opacity-0 group-hover/slot:opacity-100 focus:opacity-100"
                      >
                        <X className="size-3" aria-hidden />
                      </button>
                    </>
                  ) : (
                    <div className="grid size-full place-items-center text-[9px] text-muted-foreground">
                      <span className="text-center">
                        <Images className="mx-auto mb-1 size-3.5" aria-hidden />
                        Drop image
                      </span>
                    </div>
                  )}
                </div>

                <div className="relative flex h-12 items-center border-t-2 border-foreground/15 px-1">
                  {audioAsset?.audio?.url ? (
                    <>
                      <MemoryAudioWaveform
                        src={audioAsset.audio.url}
                        label={`Shot ${shot.shot_id} slot ${index + 1}`}
                        compact
                      />
                      <button
                        type="button"
                        aria-label={`Remove audio from Shot ${shot.shot_id} slot ${index + 1}`}
                        onClick={() => clearMedia(index, "audio")}
                        className="absolute right-1 top-1 grid size-4 place-items-center rounded bg-background/85 text-muted-foreground opacity-0 group-hover/slot:opacity-100 focus:opacity-100"
                      >
                        <X className="size-2.5" aria-hidden />
                      </button>
                    </>
                  ) : (
                    <div className="flex w-full items-center justify-center gap-1 text-[9px] text-muted-foreground">
                      <Music2 className="size-3" aria-hidden />
                      Drop audio
                    </div>
                  )}
                </div>
              </li>
            );
          })}

          {slots.length < MAX_SLOTS ? (
            <li
              data-testid={`shot-${shot.shot_id}-memory-slot-add`}
              aria-label={`Add memory slot to Shot ${shot.shot_id}`}
              onDragOver={(event) => dragOver(event, slots.length)}
              onDragLeave={() =>
                setDragOverIndex((current) =>
                  current === slots.length ? null : current,
                )
              }
              onDrop={(event) => dropAt(event, slots.length)}
              className={cn(
                "grid w-24 shrink-0 place-items-center rounded-lg border border-dashed text-center transition-all",
                dragOverIndex === slots.length
                  ? "scale-[1.02] border-sky-500 bg-sky-500/[0.08] text-sky-600 ring-2 ring-sky-500/20"
                  : "border-border/80 text-muted-foreground",
              )}
            >
              <span className="px-2 text-[9px]">
                <span className="mx-auto mb-1 grid size-7 place-items-center rounded-full border border-current/40">
                  <Plus className="size-3.5" aria-hidden />
                </span>
                Drop to add
              </span>
            </li>
          ) : null}
        </ol>
          </div>
        </div>
      </div>
      {status ? (
        <p role="status" className="mt-1 text-[9px] text-muted-foreground">
          {status}
        </p>
      ) : null}

      <ImageLightbox
        images={previewImages}
        index={lightboxIndex}
        onIndexChange={setLightboxIndex}
        onOpenChange={(open) => {
          if (!open) setLightboxIndex(null);
        }}
      />
      <ImageLightbox
        images={conditionImage?.url ? [conditionImage] : []}
        index={conditionLightboxOpen ? 0 : null}
        onIndexChange={() => {}}
        onOpenChange={(open) => {
          if (!open) setConditionLightboxOpen(false);
        }}
      />
      {sourceVideo?.url && onCreateAsset ? (
        <ShotReferenceDialog
          open={referenceDialogOpen}
          shotId={shot.shot_id}
          videoUrl={sourceVideo.url}
          onOpenChange={setReferenceDialogOpen}
          onSave={(asset) => onCreateAsset(shot.shot_id, asset)}
        />
      ) : null}
    </section>
  );
});

const REFERENCE_TYPE_OPTIONS: Array<{
  value: MemoryReferenceType;
  label: string;
}> = [
  { value: "character", label: "Character" },
  { value: "scene", label: "Scene" },
  { value: "style", label: "Style" },
  { value: "object", label: "Object" },
  { value: "other", label: "Other" },
];

function formatClipTime(value: number): string {
  const safe = Number.isFinite(value) ? Math.max(0, value) : 0;
  return `${Math.floor(safe / 60)}:${(safe % 60).toFixed(2).padStart(5, "0")}`;
}

export function ShotReferenceDialog({
  open,
  shotId,
  videoUrl,
  onOpenChange,
  onSave,
}: {
  open: boolean;
  shotId: number;
  videoUrl: string;
  onOpenChange: (open: boolean) => void;
  onSave: (asset: ShotMemoryAssetCreate) => Promise<void>;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [duration, setDuration] = useState(0);
  const [frameTime, setFrameTime] = useState(0);
  const [referenceType, setReferenceType] = useState<MemoryReferenceType>("character");
  const [referenceLabel, setReferenceLabel] = useState("");
  const [profileText, setProfileText] = useState("");
  const [includeAudio, setIncludeAudio] = useState(true);
  const [audioStart, setAudioStart] = useState(0);
  const [audioEnd, setAudioEnd] = useState(0);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!open) return;
    setDuration(0);
    setFrameTime(0);
    setReferenceType("character");
    setReferenceLabel("");
    setProfileText("");
    setIncludeAudio(true);
    setAudioStart(0);
    setAudioEnd(0);
    setSubmitting(false);
  }, [open]);

  const seekFrame = (value: number) => {
    const next = Math.min(Math.max(value, 0), duration || value);
    setFrameTime(next);
    if (videoRef.current) {
      videoRef.current.currentTime = next;
      videoRef.current.pause();
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>Save reference from Shot {shotId}</DialogTitle>
          <DialogDescription>
            Save any number of character, scene, style, or object references. Choose the frame and an optional audio clip independently.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 md:grid-cols-[minmax(0,1.3fr)_minmax(15rem,0.7fr)]">
          <div className="space-y-3">
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              playsInline
              preload="metadata"
              className="aspect-video w-full rounded-lg border border-border bg-black object-contain"
              onLoadedMetadata={(event) => {
                const nextDuration = Number(event.currentTarget.duration || 0);
                const middle = nextDuration / 2;
                setDuration(nextDuration);
                setFrameTime(middle);
                setAudioStart(Math.max(0, middle - 1));
                setAudioEnd(Math.min(nextDuration, middle + 1));
                event.currentTarget.currentTime = middle;
              }}
              onTimeUpdate={(event) => setFrameTime(event.currentTarget.currentTime)}
              onSeeked={(event) => setFrameTime(event.currentTarget.currentTime)}
            />
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Image frame {formatClipTime(frameTime)}</span>
                <span>{formatClipTime(duration)}</span>
              </div>
              <input
                aria-label="Reference image time"
                type="range"
                min={0}
                max={Math.max(duration, 0.01)}
                step={0.01}
                value={Math.min(frameTime, Math.max(duration, 0.01))}
                disabled={duration <= 0 || submitting}
                onChange={(event) => seekFrame(Number(event.currentTarget.value))}
                className="h-2 w-full cursor-pointer accent-foreground"
              />
            </div>
            <label className="flex items-center gap-2 text-xs text-foreground/75">
              <input
                type="checkbox"
                checked={includeAudio}
                disabled={submitting}
                onChange={(event) => setIncludeAudio(event.currentTarget.checked)}
              />
              Include a clipped audio reference
            </label>
            {includeAudio ? (
              <div className="rounded-lg border border-border p-2.5">
                <div className="mb-2 flex justify-between text-xs text-muted-foreground">
                  <span>Audio {formatClipTime(audioStart)}</span>
                  <span>to {formatClipTime(audioEnd)}</span>
                </div>
                <input
                  aria-label="Audio clip start"
                  type="range"
                  min={0}
                  max={Math.max(duration, 0.01)}
                  step={0.01}
                  value={Math.min(audioStart, Math.max(duration, 0.01))}
                  disabled={duration <= 0 || submitting}
                  onChange={(event) => {
                    const value = Number(event.currentTarget.value);
                    setAudioStart(Math.min(value, Math.max(0, audioEnd - 0.05)));
                  }}
                  className="h-2 w-full cursor-pointer accent-foreground"
                />
                <input
                  aria-label="Audio clip end"
                  type="range"
                  min={0}
                  max={Math.max(duration, 0.01)}
                  step={0.01}
                  value={Math.min(audioEnd, Math.max(duration, 0.01))}
                  disabled={duration <= 0 || submitting}
                  onChange={(event) => {
                    const value = Number(event.currentTarget.value);
                    setAudioEnd(Math.max(value, audioStart + 0.05));
                  }}
                  className="h-2 w-full cursor-pointer accent-foreground"
                />
              </div>
            ) : null}
          </div>

          <div className="space-y-3">
            <label className="block space-y-1 text-xs text-muted-foreground">
              <span>Reference type</span>
              <select
                aria-label="Reference type"
                value={referenceType}
                disabled={submitting}
                onChange={(event) => setReferenceType(event.target.value as MemoryReferenceType)}
                className="h-9 w-full rounded-md border border-border bg-background px-2 text-sm text-foreground"
              >
                {REFERENCE_TYPE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label className="block space-y-1 text-xs text-muted-foreground">
              <span>Reference label</span>
              <input
                aria-label="Reference label"
                value={referenceLabel}
                maxLength={80}
                disabled={submitting}
                onChange={(event) => setReferenceLabel(event.target.value)}
                placeholder="角色_A / 雨夜街道"
                className="h-9 w-full rounded-md border border-border bg-background px-2 text-sm text-foreground"
              />
            </label>
            <label className="block space-y-1 text-xs text-muted-foreground">
              <span>Profile</span>
              <textarea
                aria-label="Reference profile"
                value={profileText}
                rows={6}
                disabled={submitting}
                onChange={(event) => setProfileText(event.target.value)}
                placeholder="Describe identity, scene, appearance, motion, or sound. Leave blank to ask the configured VLM."
                className="w-full resize-y rounded-md border border-border bg-background px-2 py-1.5 text-xs text-foreground"
              />
            </label>
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              The saved profile and Shot {shotId} provenance are visible to the Agent when it recommends Memory Slots.
            </p>
          </div>
        </div>

        <DialogFooter>
          <Button type="button" variant="outline" disabled={submitting} onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            type="button"
            disabled={
              duration <= 0
              || submitting
              || (includeAudio && audioEnd <= audioStart)
            }
            onClick={async () => {
              setSubmitting(true);
              try {
                await onSave({
                  timestamp_sec: frameTime,
                  reference_type: referenceType,
                  reference_label: referenceLabel.trim(),
                  profile_text: profileText.trim(),
                  include_audio: includeAudio,
                  ...(includeAudio
                    ? { audio_start_sec: audioStart, audio_end_sec: audioEnd }
                    : {}),
                });
                onOpenChange(false);
              } finally {
                setSubmitting(false);
              }
            }}
          >
            {submitting ? <LoaderCircle className="mr-2 size-4 animate-spin" /> : <Save className="mr-2 size-4" />}
            Save to Memory Bank
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
