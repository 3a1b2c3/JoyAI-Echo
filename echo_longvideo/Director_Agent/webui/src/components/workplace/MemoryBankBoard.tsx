import {
  Database,
  GripVertical,
  Images,
  Info,
  Music2,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Save,
  Trash2,
  Upload,
  UserRound,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent } from "react";

import { ImageLightbox } from "@/components/ImageLightbox";
import type {
  GenerationMemory,
  MemoryAssetUpload,
  MemoryReferenceType,
  MemorySlotReference,
  MemoryWorkspaceAsset,
  UIImage,
  WorkplaceShot,
} from "@/lib/types";
import { cn } from "@/lib/utils";

type MemoryFilter = "all" | "automatic" | "local";

interface MemoryBankBoardProps {
  assets: MemoryWorkspaceAsset[];
  shots: WorkplaceShot[];
  busy?: boolean;
  showSlots?: boolean;
  onSaveAsset: (asset: MemoryAssetUpload) => Promise<void>;
  onDeleteAsset: (assetId: string) => Promise<void>;
  onApplySlots: (
    shotId: number,
    slots: MemorySlotReference[],
  ) => Promise<void>;
}

const FILTERS: Array<{ id: MemoryFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "automatic", label: "Auto" },
  { id: "local", label: "Local" },
];

const REFERENCE_TYPES: Array<{ value: MemoryReferenceType | ""; label: string }> = [
  { value: "", label: "Unassigned" },
  { value: "character", label: "Character" },
  { value: "scene", label: "Scene reference" },
  { value: "style", label: "Style reference" },
  { value: "object", label: "Object reference" },
  { value: "other", label: "Other" },
];

const MAX_SLOTS = 7;
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const MAX_AUDIO_BYTES = 20 * 1024 * 1024;
export const MEMORY_ASSET_DRAG_TYPE = "application/x-echo-memory-asset";
export const MEMORY_SLOT_DRAG_TYPE = "application/x-echo-memory-slot";

type DraftSlot = {
  image_asset_id?: string;
  audio_asset_id?: string;
};

export type MemoryAssetDrag = {
  assetId: string;
  mediaType: "image" | "audio";
};

const MEMORY_DRAG_TEXT_PREFIX = "echo-memory:";

export function writeMemoryAssetDrag(
  dataTransfer: DataTransfer,
  payload: MemoryAssetDrag,
) {
  const serialized = JSON.stringify(payload);
  dataTransfer.effectAllowed = "copy";
  dataTransfer.setData(MEMORY_ASSET_DRAG_TYPE, serialized);
  // Chromium can omit custom MIME data in some nested draggable controls.
  dataTransfer.setData("text/plain", `${MEMORY_DRAG_TEXT_PREFIX}${serialized}`);
}

export function readMemoryAssetDrag(
  dataTransfer: DataTransfer,
): MemoryAssetDrag | null {
  const custom = dataTransfer.getData(MEMORY_ASSET_DRAG_TYPE);
  const plain = dataTransfer.getData("text/plain");
  const serialized = custom || (
    plain.startsWith(MEMORY_DRAG_TEXT_PREFIX)
      ? plain.slice(MEMORY_DRAG_TEXT_PREFIX.length)
      : ""
  );
  if (!serialized) return null;
  try {
    const payload = JSON.parse(serialized) as MemoryAssetDrag;
    if (
      typeof payload.assetId !== "string" ||
      !["image", "audio"].includes(payload.mediaType)
    ) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}

function refsAsDraftSlots(
  refs: MemorySlotReference[],
): DraftSlot[] {
  return refs.slice(0, MAX_SLOTS).flatMap((ref) => {
    const imageAssetId = ref.image_asset_id ?? ref.asset_id;
    const audioAssetId = ref.audio_asset_id;
    return imageAssetId || audioAssetId
      ? [{
          ...(imageAssetId ? { image_asset_id: imageAssetId } : {}),
          ...(audioAssetId ? { audio_asset_id: audioAssetId } : {}),
        }]
      : [];
  });
}

function compactSlotRefs(slots: DraftSlot[]): MemorySlotReference[] {
  return slots.flatMap((slot) =>
    slot?.image_asset_id
      ? [{
          image_asset_id: slot.image_asset_id,
          ...(slot.audio_asset_id
            ? { audio_asset_id: slot.audio_asset_id }
            : {}),
        }]
      : [],
  );
}

function refSignature(refs: MemorySlotReference[]): string {
  return refs
    .map((ref) =>
      `${ref.image_asset_id ?? ref.asset_id ?? ""}:${ref.audio_asset_id ?? ""}`,
    )
    .join("|");
}

function fileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`Cannot read ${file.name}`));
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.readAsDataURL(file);
  });
}

function nameWithoutExtension(name: string): string {
  return name.replace(/\.[^.]+$/, "").trim() || "Local asset";
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

export function MemoryBankBoard({
  assets,
  shots,
  busy = false,
  showSlots = true,
  onSaveAsset,
  onDeleteAsset,
  onApplySlots,
}: MemoryBankBoardProps) {
  const [filter, setFilter] = useState<MemoryFilter>("all");
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);
  const [targetShotId, setTargetShotId] = useState<number | null>(
    shots[0]?.shot_id ?? null,
  );
  const [draftSlots, setDraftSlots] = useState<DraftSlot[]>([]);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);
  const uploadTargetRef = useRef<string | null>(null);

  useEffect(() => {
    if (
      targetShotId != null &&
      shots.some((shot) => shot.shot_id === targetShotId)
    ) {
      return;
    }
    setTargetShotId(shots[0]?.shot_id ?? null);
  }, [shots, targetShotId]);

  const targetShot = shots.find((shot) => shot.shot_id === targetShotId);
  const appliedRefs = useMemo<MemorySlotReference[]>(() => {
    if (targetShot?.approved_memory_slot_refs) {
      return targetShot.approved_memory_slot_refs;
    }
    return (targetShot?.memory_slots ?? targetShot?.generation_memories ?? [])
      .reduce<MemorySlotReference[]>((refs, memory) => {
        const imageAssetId = appliedAssetId(memory, assets);
        if (!imageAssetId) return refs;
        refs.push({
          image_asset_id: imageAssetId,
          ...(memory.metadata.audio_workspace_asset_id
            ? { audio_asset_id: memory.metadata.audio_workspace_asset_id }
            : {}),
        });
        return refs;
      }, []);
  }, [assets, targetShot?.approved_memory_slot_refs, targetShot?.generation_memories, targetShot?.memory_slots]);
  const recommendedRefs = useMemo<MemorySlotReference[]>(() => {
    const refs = targetShot?.recommended_memory_slot_refs ?? [];
    if (refs.length > 0) return refs;
    return (targetShot?.recommended_memory_slots ?? [])
      .flatMap((memory) => {
        const imageAssetId = appliedAssetId(memory, assets);
        return imageAssetId ? [{ image_asset_id: imageAssetId }] : [];
      });
  }, [assets, targetShot?.recommended_memory_slot_refs, targetShot?.recommended_memory_slots]);
  const serverDraftRefs = targetShot?.memory_slots_configured
    ? appliedRefs
    : recommendedRefs;
  const serverDraftSignature = refSignature(serverDraftRefs);

  useEffect(() => {
    setDraftSlots(refsAsDraftSlots(serverDraftRefs));
    setMessage(null);
    // The signature changes only when the server-side draft changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetShotId, serverDraftSignature, targetShot?.memory_slots_configured]);

  const automaticCount = assets.filter(
    (asset) => asset.source === "automatic",
  ).length;
  const identityCount = assets.filter(
    (asset) => asset.kind === "character",
  ).length;
  const localCount = assets.length - automaticCount;
  const visible = useMemo(
    () =>
      filter === "all"
        ? assets
        : assets.filter((asset) => asset.source === filter),
    [assets, filter],
  );
  const assetById = useMemo(
    () => new Map(assets.map((asset) => [asset.asset_id, asset])),
    [assets],
  );
  const draftRefs = compactSlotRefs(draftSlots);
  const draftSignature = refSignature(draftRefs);
  const appliedRefSignature = refSignature(appliedRefs);
  const dirty = !targetShot?.memory_slots_configured || draftSignature !== appliedRefSignature;
  const occupiedSlotCount = draftSlots.length;
  const imageSlotCount = draftRefs.length;

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

  const uploadFile = async (
    mediaKind: "image" | "audio",
    file: File | undefined,
  ) => {
    if (!file) return;
    const targetId = uploadTargetRef.current;
    uploadTargetRef.current = null;
    const isImage = mediaKind === "image";
    if (isImage && !file.type.startsWith("image/")) {
      setMessage("Choose a PNG, JPEG, WebP, or GIF image.");
      return;
    }
    if (!isImage && !file.type.startsWith("audio/")) {
      setMessage("Choose an audio file.");
      return;
    }
    if (file.size > (isImage ? MAX_IMAGE_BYTES : MAX_AUDIO_BYTES)) {
      setMessage(
        isImage
          ? "Image must be 8 MB or smaller."
          : "Audio must be 20 MB or smaller.",
      );
      return;
    }
    setMessage(null);
    try {
      const upload = {
        data_url: await fileAsDataUrl(file),
        name: file.name,
      };
      await onSaveAsset({
        ...(targetId ? { asset_id: targetId } : {}),
        ...(!targetId
          ? { display_name: nameWithoutExtension(file.name) }
          : {}),
        [mediaKind]: upload,
      });
      setMessage(
        targetId
          ? `${mediaKind === "image" ? "Image" : "Audio"} replaced.`
          : "Asset added.",
      );
    } catch (error) {
      setMessage((error as Error).message);
    }
  };

  const addImageToDraft = (assetId: string) => {
    setMessage(null);
    setDraftSlots((current) => {
      if (
        current.length >= MAX_SLOTS ||
        current.some((slot) => slot.image_asset_id === assetId)
      ) {
        return current;
      }
      return [...current, { image_asset_id: assetId }];
    });
  };

  const setSlotMedia = (
    index: number,
    mediaType: "image" | "audio",
    assetId: string,
  ) => {
    setMessage(null);
    setDraftSlots((current) => {
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
          [mediaType === "image" ? "image_asset_id" : "audio_asset_id"]: assetId,
        }];
      }
      if (index < 0 || index >= current.length) return current;
      const next = [...current];
      next[index] = {
        ...(next[index] ?? {}),
        [mediaType === "image" ? "image_asset_id" : "audio_asset_id"]: assetId,
      };
      return next;
    });
  };

  const clearSlotMedia = (index: number, mediaType: "image" | "audio") => {
    setDraftSlots((current) => {
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
    setDraftSlots((current) => {
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

  const dropOnSlot = (event: DragEvent, targetIndex: number) => {
    event.preventDefault();
    const payload = readDragPayload(event);
    setDragOverIndex(null);
    if (!payload) return;
    if (payload.type === "slot") {
      moveSlot(payload.sourceIndex, targetIndex);
      return;
    }
    setSlotMedia(targetIndex, payload.mediaType, payload.assetId);
  };

  const applyDraft = async () => {
    if (targetShotId == null) return;
    setMessage(null);
    try {
      await onApplySlots(
        targetShotId,
        draftRefs,
      );
      setMessage(
        `Applied ${draftRefs.length} slot${draftRefs.length === 1 ? "" : "s"} to Shot ${targetShotId}.`,
      );
    } catch (error) {
      setMessage((error as Error).message);
    }
  };

  return (
    <section
      aria-label="Memory bank"
      className="shrink-0 border-y border-border/55 bg-foreground/[0.018]"
    >
      <input
        ref={imageInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/gif"
        className="hidden"
        onChange={(event) => {
          void uploadFile("image", event.target.files?.[0]);
          event.target.value = "";
        }}
      />
      <input
        ref={audioInputRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={(event) => {
          void uploadFile("audio", event.target.files?.[0]);
          event.target.value = "";
        }}
      />

      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="grid size-7 shrink-0 place-items-center rounded-md bg-foreground/[0.06] text-foreground/55">
            <Database className="size-3.5" aria-hidden />
          </span>
          <div className="min-w-0">
            <h3 className="text-xs font-semibold text-foreground/80">
              Memory Workspace
            </h3>
            <p className="text-[10px] tabular-nums text-muted-foreground">
              {identityCount} identities · {automaticCount} auto · {localCount} local · up to 7 slots
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div
            role="group"
            aria-label="Filter memory workspace"
            className="inline-flex rounded-md bg-foreground/[0.045] p-0.5"
          >
            {FILTERS.map((item) => (
              <button
                key={item.id}
                type="button"
                aria-pressed={filter === item.id}
                onClick={() => {
                  setFilter(item.id);
                  setLightboxIndex(null);
                }}
                className={cn(
                  "h-6 rounded px-2 text-[10px] font-medium transition-colors",
                  filter === item.id
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground/75",
                )}
              >
                {item.label}
              </button>
            ))}
          </div>
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              uploadTargetRef.current = null;
              imageInputRef.current?.click();
            }}
            className="inline-flex h-7 items-center gap-1.5 rounded-md border border-border bg-background px-2.5 text-[10px] font-medium text-foreground/75 hover:bg-muted disabled:opacity-50"
          >
            <Upload className="size-3" aria-hidden />
            Upload image
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              uploadTargetRef.current = null;
              audioInputRef.current?.click();
            }}
            className="inline-flex h-7 items-center gap-1.5 rounded-md border border-border bg-background px-2.5 text-[10px] font-medium text-foreground/75 hover:bg-muted disabled:opacity-50"
          >
            <Music2 className="size-3" aria-hidden />
            Upload audio
          </button>
        </div>
      </div>

      {visible.length > 0 ? (
        <div className="flex gap-2 overflow-x-auto px-4 pb-3 scrollbar-thin">
          {visible.map((asset) => (
            <MemoryAssetCard
              key={asset.asset_id}
              asset={asset}
              selected={draftSlots.some(
                (slot) => slot.image_asset_id === asset.asset_id,
              )}
              slotLimitReached={draftSlots.length >= MAX_SLOTS}
              showAdd={showSlots}
              busy={busy}
              onPreview={() =>
                asset.image?.url
                  ? setLightboxIndex(previewIndexByAssetId.get(asset.asset_id) ?? 0)
                  : undefined
              }
              onAdd={() => asset.image?.url ? addImageToDraft(asset.asset_id) : undefined}
              onReplaceImage={() => {
                uploadTargetRef.current = asset.asset_id;
                imageInputRef.current?.click();
              }}
              onReplaceAudio={() => {
                uploadTargetRef.current = asset.asset_id;
                audioInputRef.current?.click();
              }}
              onRemoveAudio={async () => {
                setMessage(null);
                try {
                  await onSaveAsset({
                    asset_id: asset.asset_id,
                    remove_audio: true,
                  });
                  setMessage("Audio removed.");
                } catch (error) {
                  setMessage((error as Error).message);
                }
              }}
              onDelete={async () => {
                setMessage(null);
                try {
                  await onDeleteAsset(asset.asset_id);
                  setDraftSlots((current) => current.flatMap((slot) => {
                    const next = { ...slot };
                    if (next.image_asset_id === asset.asset_id) {
                      delete next.image_asset_id;
                    }
                    if (next.audio_asset_id === asset.asset_id) {
                      delete next.audio_asset_id;
                    }
                    return next.image_asset_id || next.audio_asset_id ? [next] : [];
                  }));
                  setMessage("Asset removed from the workspace.");
                } catch (error) {
                  setMessage((error as Error).message);
                }
              }}
              onSaveProfile={async ({ profileText, referenceType, referenceLabel }) => {
                setMessage(null);
                try {
                  await onSaveAsset({
                    asset_id: asset.asset_id,
                    profile_text: profileText,
                    reference_type: referenceType || null,
                    reference_label: referenceLabel,
                    identity_ids:
                      referenceType === "character" && referenceLabel
                        ? [referenceLabel]
                        : [],
                  });
                  setMessage("Asset reference and profile saved for generation.");
                } catch (error) {
                  setMessage((error as Error).message);
                }
              }}
            />
          ))}
        </div>
      ) : (
        <div className="flex h-20 items-center justify-center gap-2 px-4 pb-3 text-xs text-muted-foreground">
          <Images className="size-4" aria-hidden />
          No {filter === "all" ? "" : `${filter} `}assets yet
        </div>
      )}

      {showSlots ? (
      <div className="border-t border-border/55 px-4 py-3">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <label
              htmlFor="memory-target-shot"
              className="text-[10px] font-semibold text-foreground/70"
            >
              Assemble for
            </label>
            <select
              id="memory-target-shot"
              value={targetShotId ?? ""}
              onChange={(event) => setTargetShotId(Number(event.target.value))}
              className="h-7 rounded-md border border-border bg-background px-2 text-[10px] text-foreground"
            >
              {shots.map((shot) => (
                <option key={shot.shot_id} value={shot.shot_id}>
                  Shot {shot.shot_id}
                </option>
              ))}
            </select>
            <span className="text-[10px] text-muted-foreground">
              {occupiedSlotCount}/{MAX_SLOTS} positions · {imageSlotCount} ready
            </span>
            {!targetShot?.memory_slots_configured && recommendedRefs.length > 0 ? (
              <span className="rounded bg-sky-500/10 px-1.5 py-0.5 text-[9px] font-medium text-sky-700 dark:text-sky-300">
                Agent recommendation · review before applying
              </span>
            ) : null}
          </div>
          <button
            type="button"
            disabled={busy || targetShotId == null || !dirty}
            onClick={() => void applyDraft()}
            className="inline-flex h-7 items-center gap-1.5 rounded-md bg-foreground px-3 text-[10px] font-medium text-background disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Save className="size-3" aria-hidden />
            Apply to Shot {targetShotId ?? "–"}
          </button>
        </div>

        <p className="mb-2 text-[10px] text-muted-foreground">
          Drag assets in to add them. Drag a slot to reorder.
        </p>
        <div className="overflow-x-auto pb-1 scrollbar-thin">
          <ol
            aria-label="Memory slots"
            className="flex min-h-36 items-stretch gap-2"
          >
          {draftSlots.map((slot, index) => {
            const imageAsset = slot.image_asset_id
              ? assetById.get(slot.image_asset_id)
              : undefined;
            const audioAsset = slot.audio_asset_id
              ? assetById.get(slot.audio_asset_id)
              : undefined;
            return (
              <li
                key={`${slot.image_asset_id ?? "_"}:${slot.audio_asset_id ?? "_"}:${index}`}
                data-testid={`memory-slot-${index + 1}`}
                aria-label={`Memory slot ${index + 1}`}
                draggable
                onDragStart={(event) => {
                  event.dataTransfer.effectAllowed = "move";
                  event.dataTransfer.setData(MEMORY_SLOT_DRAG_TYPE, String(index));
                }}
                onDragEnd={() => setDragOverIndex(null)}
                onDragOver={(event) => {
                  event.preventDefault();
                  event.dataTransfer.dropEffect = Array.from(
                    event.dataTransfer.types,
                  ).includes(MEMORY_SLOT_DRAG_TYPE)
                    ? "move"
                    : "copy";
                  setDragOverIndex(index);
                }}
                onDragLeave={() => {
                  setDragOverIndex((current) => current === index ? null : current);
                }}
                onDrop={(event) => dropOnSlot(event, index)}
                className={cn(
                  "group/slot relative w-28 shrink-0 cursor-grab overflow-hidden rounded-lg border bg-background transition-all active:cursor-grabbing",
                  dragOverIndex === index
                    ? "translate-y-[-2px] border-sky-500 bg-sky-500/[0.06] ring-2 ring-sky-500/20"
                    : "border-border/70",
                )}
              >
                <div className="absolute left-1 top-1 z-10 flex items-center gap-0.5">
                  <span className="grid size-5 place-items-center rounded bg-black/60 text-[9px] font-semibold text-white tabular-nums">
                    {index + 1}
                  </span>
                  <span
                    aria-hidden
                    className="grid size-5 place-items-center rounded bg-black/60 text-white"
                  >
                    <GripVertical className="size-3" />
                  </span>
                </div>
                <button
                  type="button"
                  aria-label={`Clear memory slot ${index + 1}`}
                  title="Clear slot"
                  onClick={() => setDraftSlots((current) =>
                    current.filter((_, itemIndex) => itemIndex !== index),
                  )}
                  className="absolute right-1 top-1 z-10 grid size-5 place-items-center rounded bg-black/60 text-white opacity-0 transition-opacity group-hover/slot:opacity-100 focus:opacity-100"
                >
                  <X className="size-3" aria-hidden />
                </button>

                <div className="relative aspect-[4/3] border-b border-dashed border-border/70 bg-muted/50">
                  {imageAsset?.image?.url ? (
                    <>
                      <button
                        type="button"
                        aria-label={`Preview slot ${index + 1} image`}
                        onClick={() => setLightboxIndex(
                          previewIndexByAssetId.get(imageAsset.asset_id) ?? 0,
                        )}
                        className="block size-full overflow-hidden"
                      >
                        <img
                          src={imageAsset.image.url}
                          alt={imageAsset.display_name}
                          className="size-full object-cover transition-transform hover:scale-[1.03]"
                        />
                      </button>
                      <button
                        type="button"
                        aria-label={`Remove image from slot ${index + 1}`}
                        title="Remove image"
                        onClick={() => clearSlotMedia(index, "image")}
                        className="absolute bottom-1 right-1 grid size-5 place-items-center rounded bg-black/60 text-white opacity-0 transition-opacity group-hover/slot:opacity-100 focus:opacity-100"
                      >
                        <X className="size-3" aria-hidden />
                      </button>
                    </>
                  ) : (
                    <div className="grid size-full place-items-center px-2 text-center text-[9px] text-muted-foreground">
                      <span>
                        <span className="mx-auto mb-1 grid size-7 place-items-center rounded-md border border-dashed border-current/50">
                          <Images className="size-3.5" aria-hidden />
                        </span>
                        Drop image
                      </span>
                    </div>
                  )}
                </div>

                <div className="relative flex h-14 items-center border-t-2 border-foreground/15 px-1.5">
                  {audioAsset?.audio?.url ? (
                    <>
                      <MemoryAudioWaveform
                        src={audioAsset.audio.url}
                        label={`Slot ${index + 1}: ${audioAsset.display_name}`}
                        compact
                      />
                      <button
                        type="button"
                        aria-label={`Remove audio from slot ${index + 1}`}
                        title="Remove audio"
                        onClick={() => clearSlotMedia(index, "audio")}
                        className="absolute right-1 top-1 grid size-4 place-items-center rounded bg-background/85 text-muted-foreground opacity-0 transition-opacity group-hover/slot:opacity-100 focus:opacity-100"
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
          {draftSlots.length < MAX_SLOTS ? (
            <li
              data-testid="memory-slot-add"
              aria-label="Add memory slot"
              onDragOver={(event) => {
                event.preventDefault();
                event.dataTransfer.dropEffect = Array.from(
                  event.dataTransfer.types,
                ).includes(MEMORY_SLOT_DRAG_TYPE)
                  ? "move"
                  : "copy";
                setDragOverIndex(draftSlots.length);
              }}
              onDragLeave={() => {
                setDragOverIndex((current) =>
                  current === draftSlots.length ? null : current,
                );
              }}
              onDrop={(event) => dropOnSlot(event, draftSlots.length)}
              className={cn(
                "grid w-28 shrink-0 place-items-center rounded-lg border border-dashed text-center transition-all",
                dragOverIndex === draftSlots.length
                  ? "scale-[1.02] border-sky-500 bg-sky-500/[0.08] text-sky-600 ring-2 ring-sky-500/20"
                  : "border-border/80 bg-foreground/[0.015] text-muted-foreground",
              )}
            >
              <span className="px-3 text-[9px]">
                <span className="mx-auto mb-1.5 grid size-8 place-items-center rounded-full border border-current/40">
                  <Plus className="size-4" aria-hidden />
                </span>
                Drop to add
              </span>
            </li>
          ) : null}
          </ol>
        </div>
        {message ? (
          <p role="status" className="mt-2 text-[10px] text-muted-foreground">
            {message}
          </p>
        ) : null}
      </div>
      ) : null}

      <ImageLightbox
        images={previewImages}
        index={lightboxIndex}
        onIndexChange={setLightboxIndex}
        onOpenChange={(open) => {
          if (!open) setLightboxIndex(null);
        }}
      />
    </section>
  );
}

function MemoryAssetCard({
  asset,
  selected,
  slotLimitReached,
  showAdd,
  busy,
  onPreview,
  onAdd,
  onReplaceImage,
  onReplaceAudio,
  onRemoveAudio,
  onDelete,
  onSaveProfile,
}: {
  asset: MemoryWorkspaceAsset;
  selected: boolean;
  slotLimitReached: boolean;
  showAdd: boolean;
  busy: boolean;
  onPreview: () => void;
  onAdd: () => void;
  onReplaceImage: () => void;
  onReplaceAudio: () => void;
  onRemoveAudio: () => Promise<void>;
  onDelete: () => Promise<void>;
  onSaveProfile: (details: {
    profileText: string;
    referenceType: MemoryReferenceType | "";
    referenceLabel: string;
  }) => Promise<void>;
}) {
  const local = asset.source === "local";
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [profileText, setProfileText] = useState(asset.profile_text ?? "");
  const [referenceType, setReferenceType] = useState<MemoryReferenceType | "">(
    asset.reference_type ?? "",
  );
  const [referenceLabel, setReferenceLabel] = useState(asset.reference_label ?? "");
  useEffect(() => setProfileText(asset.profile_text ?? ""), [asset.profile_text]);
  useEffect(() => setReferenceType(asset.reference_type ?? ""), [asset.reference_type]);
  useEffect(() => setReferenceLabel(asset.reference_label ?? ""), [asset.reference_label]);
  const detailsDirty =
    profileText.trim() !== (asset.profile_text ?? "").trim()
    || referenceType !== (asset.reference_type ?? "")
    || referenceLabel.trim() !== (asset.reference_label ?? "").trim();
  return (
    <article
      draggable={!busy && Boolean(asset.image?.url || asset.audio?.url)}
      onDragStart={(event) => {
        writeMemoryAssetDrag(event.dataTransfer, {
          assetId: asset.asset_id,
          mediaType: asset.image?.url ? "image" : "audio",
        });
      }}
      title="Drag this asset into a Shot memory slot"
      className="grid w-56 shrink-0 cursor-grab grid-cols-[7rem_1fr] self-start overflow-hidden rounded-lg border border-border/70 bg-background active:cursor-grabbing"
    >
      <button
        type="button"
        onClick={onPreview}
        disabled={!asset.image?.url}
        aria-label={asset.image?.url
          ? `Preview ${asset.display_name} memory image`
          : `${asset.display_name} audio asset`}
        className="relative block aspect-square cursor-grab overflow-hidden bg-muted text-left active:cursor-grabbing disabled:cursor-default"
      >
        {asset.image?.url ? (
          <img
            draggable={false}
            src={asset.image.url}
            alt={`${asset.display_name} Memory`}
            className="size-full object-cover"
          />
        ) : (
          <span className="grid size-full place-items-center text-muted-foreground">
            <Music2 className="size-8" aria-hidden />
          </span>
        )}
        <span
          className={cn(
            "pointer-events-none absolute bottom-1 left-1 inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[9px] font-medium text-white",
            local ? "bg-violet-700/90" : "bg-black/70",
          )}
        >
          {local ? (
            <Upload className="size-2.5" aria-hidden />
          ) : asset.kind === "character" ? (
            <UserRound className="size-2.5" aria-hidden />
          ) : (
            <Images className="size-2.5" aria-hidden />
          )}
          {local ? "Local" : "Auto"}
        </span>
      </button>
      <div className="flex min-w-0 flex-col gap-2 p-2">
        <div className="flex min-w-0 items-start gap-1">
          <div
            className="min-w-0 flex-1 truncate text-[11px] font-semibold text-foreground/80"
            title={asset.display_name}
          >
            {asset.display_name}
          </div>
          <button
            type="button"
            aria-expanded={detailsOpen}
            aria-label={`${detailsOpen ? "Hide" : "Show"} ${asset.display_name} details`}
            title="Asset details"
            onClick={() => setDetailsOpen((open) => !open)}
            className={cn(
              "grid size-6 shrink-0 place-items-center rounded border border-border text-muted-foreground transition-colors hover:text-foreground",
              detailsOpen && "bg-muted text-foreground",
            )}
          >
            <Info className="size-3" aria-hidden />
          </button>
        </div>
        {asset.audio?.url ? (
          <MemoryAudioWaveform
            src={asset.audio.url}
            label={asset.display_name}
            draggable
            onDragStart={(event) => {
              event.stopPropagation();
              writeMemoryAssetDrag(event.dataTransfer, {
                assetId: asset.asset_id,
                mediaType: "audio",
              });
            }}
          />
        ) : (
          <span className="flex h-7 items-center gap-1 rounded border border-dashed border-border px-1.5 text-[9px] text-muted-foreground">
            <Music2 className="size-3" aria-hidden />
            No audio
          </span>
        )}
        {showAdd ? (
        <div className="mt-auto">
          <button
            type="button"
            disabled={selected || slotLimitReached || busy || !asset.image?.url}
            onClick={onAdd}
            className="inline-flex h-7 w-full items-center justify-center gap-1 rounded bg-foreground px-2 text-[9px] font-medium text-background disabled:opacity-35"
          >
            <Plus className="size-2.5" />
            {selected ? "Added" : asset.image?.url ? "Add slot" : "Audio only"}
          </button>
        </div>
        ) : null}
      </div>

      {detailsOpen ? (
        <div className="col-span-2 border-t border-border/60 bg-foreground/[0.015] p-2">
          <div className="mb-1.5 flex items-center gap-1.5 text-[9px] text-muted-foreground">
            <span>{local ? "Local asset" : "Automatic asset"}</span>
            {asset.source_shot_id ? (
              <span>· Shot {String(asset.source_shot_id).padStart(3, "0")}</span>
            ) : null}
          </div>
          <div className="mb-1.5 grid grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] gap-1.5">
            <select
              aria-label={`Assign ${asset.display_name} reference type`}
              value={referenceType}
              onChange={(event) =>
                setReferenceType(event.target.value as MemoryReferenceType | "")
              }
              className="h-7 min-w-0 rounded border border-border bg-background px-1.5 text-[9px] text-foreground"
            >
              {REFERENCE_TYPES.map((option) => (
                <option key={option.value || "unassigned"} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <input
              aria-label={`Edit ${asset.display_name} reference label`}
              value={referenceLabel}
              onChange={(event) => setReferenceLabel(event.target.value)}
              placeholder="角色_A / 雨夜街道"
              maxLength={80}
              className="h-7 min-w-0 rounded border border-border bg-background px-1.5 text-[9px] text-foreground placeholder:text-muted-foreground/60"
            />
          </div>
          <textarea
            aria-label={`Edit ${asset.display_name} profile`}
            value={profileText}
            onChange={(event) => setProfileText(event.target.value)}
            placeholder="Describe identity, appearance, scene, motion, or audio cues…"
            rows={3}
            className="min-h-14 w-full resize-y rounded border border-border bg-background px-1.5 py-1 text-[9px] leading-3 text-foreground placeholder:text-muted-foreground/60"
          />
          <div className="mt-1.5 flex flex-wrap gap-1">
            <button
              type="button"
              disabled={busy || !detailsDirty}
              aria-label={`Save ${asset.display_name} profile`}
              onClick={() => void onSaveProfile({
                profileText: profileText.trim(),
                referenceType,
                referenceLabel: referenceLabel.trim(),
              })}
              className="inline-flex h-6 items-center gap-1 rounded border border-border px-2 text-[9px] text-muted-foreground hover:text-foreground disabled:opacity-35"
            >
              <Save className="size-2.5" />
              Save reference
            </button>
          {local ? (
            <>
              <button
                type="button"
                disabled={busy}
                aria-label={`Replace ${asset.display_name} image`}
                title="Replace image"
                onClick={onReplaceImage}
                className="grid size-6 place-items-center rounded border border-border text-muted-foreground hover:text-foreground disabled:opacity-35"
              >
                <RefreshCw className="size-2.5" />
              </button>
              <button
                type="button"
                disabled={busy}
                aria-label={`${asset.audio?.url ? "Replace" : "Add"} ${asset.display_name} audio`}
                title={asset.audio?.url ? "Replace audio" : "Add audio"}
                onClick={onReplaceAudio}
                className="grid size-6 place-items-center rounded border border-border text-muted-foreground hover:text-foreground disabled:opacity-35"
              >
                <Music2 className="size-2.5" />
              </button>
              {asset.audio?.url && asset.image?.url ? (
                <button
                  type="button"
                  disabled={busy}
                  aria-label={`Remove ${asset.display_name} audio`}
                  title="Remove audio"
                  onClick={() => void onRemoveAudio()}
                  className="grid size-6 place-items-center rounded border border-border text-muted-foreground hover:text-destructive disabled:opacity-35"
                >
                  <Trash2 className="size-2.5" />
                </button>
              ) : null}
              <button
                type="button"
                disabled={busy}
                aria-label={`Delete ${asset.display_name}`}
                title="Delete asset"
                onClick={() => void onDelete()}
                className="grid size-6 place-items-center rounded border border-border text-muted-foreground hover:text-destructive disabled:opacity-35"
              >
                <Trash2 className="size-2.5" />
              </button>
            </>
          ) : null}
          </div>
        </div>
      ) : null}
    </article>
  );
}

const WAVEFORM_BARS = [
  34, 58, 42, 78, 52, 86, 44, 66, 92, 48, 72, 38, 84, 56, 96, 62, 76, 46,
  88, 54, 68, 40, 82, 50,
];

export function MemoryAudioWaveform({
  src,
  label,
  compact = false,
  draggable = false,
  onDragStart,
}: {
  src: string;
  label: string;
  compact?: boolean;
  draggable?: boolean;
  onDragStart?: (event: DragEvent<HTMLDivElement>) => void;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);

  const togglePlayback = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      void audio.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
    } else {
      audio.pause();
      setPlaying(false);
    }
  };

  return (
    <div
      draggable={draggable}
      onDragStart={onDragStart}
      title={draggable ? "Drag this audio into a slot" : undefined}
      className={cn(
        "min-w-0 flex-1 rounded border border-border/65 bg-foreground/[0.025]",
        draggable && "cursor-grab active:cursor-grabbing",
      )}
    >
      <button
        type="button"
        aria-label={`${playing ? "Pause" : "Play"} ${label} audio`}
        onClick={togglePlayback}
        className={cn(
          "flex w-full items-center gap-1 overflow-hidden px-1 text-muted-foreground hover:text-foreground",
          compact ? "h-8" : "h-7",
        )}
      >
        {playing ? (
          <Pause className="size-2.5 shrink-0" aria-hidden />
        ) : (
          <Play className="size-2.5 shrink-0" aria-hidden />
        )}
        <span
          aria-hidden
          className={cn(
            "flex flex-1 items-center justify-between gap-px",
            compact ? "h-5" : "h-4",
          )}
        >
          {WAVEFORM_BARS.map((height, index) => (
            <span
              key={index}
              className={cn(
                "w-px min-w-px rounded-full bg-current opacity-65",
                playing && "animate-pulse",
              )}
              style={{ height: `${height}%` }}
            />
          ))}
        </span>
      </button>
      <audio
        ref={audioRef}
        src={src}
        preload="none"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />
    </div>
  );
}
