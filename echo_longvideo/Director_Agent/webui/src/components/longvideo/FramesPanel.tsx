import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";
import {
  AlertCircle,
  ArrowRight,
  Check,
  Circle,
  Clapperboard,
  Film,
  HelpCircle,
  Loader2,
  Minus,
  Play,
  Plus,
  RefreshCcw,
  Send,
  Sparkles,
  X,
  ThumbsUp,
  SquarePen,
  ChevronDown,
  FileText,
  Copy,
  // Pencil,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ImageLightbox } from "@/components/ImageLightbox";
import { ShotVideoPlayer } from "@/components/longvideo/ShotVideoPlayer";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  memoryDisplayName,
  type GenerationMemory,
  type UIImage,
} from "@/lib/types";
import { cn } from "@/lib/utils";

/** Shot1 first-frame display (uploaded from Composer; read-only here). */
export type Shot1FirstFrameControls = {
  displayUrl: string | null;
  videoSize: { width: number; height: number };
};

// export type FrameStatus = "idle" | "generating" | "done" | "error";
export type FrameStatus =
  | "planned"
  | "prompt_ready"
  | "queued"
  | "generated"
  | "error"
  | "review_pass"
  | "review_fail"
  | "approved"
  | "revised_prompt_ready";

export interface Frame {
  id: string;
  shotId?: number;
  cut?: boolean;
  caption?: string;
  numFrames?: number;
  segmentText: string;
  prompt: string;
  status: FrameStatus;
  videoUrl?: string;
  error?: string;
  durationSec?: number;
  referenceShotIds?: number[];
  referenceNote?: string;
  canGenerate?: boolean;
  dependencyMessage?: string;
  hintMessage?: string;
  hasActions?: boolean;
  reviewNotes?: string;
  accepted?: boolean;
  generationMemories?: GenerationMemory[];
  continuousEnabled?: boolean;
}

interface FramesPanelProps {
  frames: Frame[];
  memoryBank?: ReactNode;
  renderMemorySlots?: (frameId: string) => ReactNode;
  batchGenerating?: boolean;
  composeDisabled?: boolean;
  referencesReady?: boolean;
  onGenerate: (frameId: string) => void;
  onGenerateAll: () => void;
  onUpdatePrompt: (frameId: string, prompt: string) => void;
  onUpdateDuration: (frameId: string, durationSec: number) => void;
  onRetry: (frameId: string) => void;
  onAccept: (frameId: string) => void;
  onAcceptAll: () => void;
  acceptAllBusy?: boolean;
  onRevise: (frameId: string, feedback: string) => void | Promise<void>;
  onSetContinuousMode?: (frameId: string, enabled: boolean) => void;
  busyFrameId?: string | null;
  onCompose: () => void;
  /** Notifies when a shot prompt textarea enters/leaves edit mode (focus session). */
  onPromptEditingChange?: (editing: boolean) => void;
  /** Shot1 first-frame reference image (display only). */
  shot1FirstFrame?: Shot1FirstFrameControls | null;
}

const injectCSS = `
@keyframes shot-shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
@keyframes shot-pulse-ring {
  0%, 100% { opacity: 0.25; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(1.18); }
}
@keyframes shot-scan-line {
  0% { top: -2px; }
  100% { top: calc(100% + 2px); }
}
`;

const isReadyForMerge = (status: FrameStatus) =>
  status === "generated" || status === "review_pass" || status === "approved";

const isApproved = (status: FrameStatus) => status === "approved";

const isDone = (status: FrameStatus) => isReadyForMerge(status);
const isActive = (status: FrameStatus) =>
  status === "queued" ||
  status === "review_fail" ||
  status === "revised_prompt_ready";
const isError = (status: FrameStatus) => status === "error";
const isIdle = (status: FrameStatus) =>
  status === "planned" || status === "prompt_ready";
/** Shot 已开始/完成生成后锁定首尾衔接，不可再改。 */
const isContinuousLocked = (status: FrameStatus) => !isIdle(status);

export function FramesPanel({
  frames,
  memoryBank,
  renderMemorySlots,
  batchGenerating = false,
  composeDisabled = false,
  referencesReady = true,
  onGenerate,
  onGenerateAll,
  onUpdatePrompt,
  onUpdateDuration,
  onRetry,
  onAccept,
  onAcceptAll,
  acceptAllBusy = false,
  onRevise,
  onSetContinuousMode,
  busyFrameId = null,
  onCompose,
  onPromptEditingChange,
  shot1FirstFrame = null,
}: FramesPanelProps) {
  const doneCount = frames.filter((f) => isReadyForMerge(f.status)).length;
  const errorCount = frames.filter((f) => isError(f.status)).length;
  const generatingCount = frames.filter((f) => isActive(f.status)).length;
  // 所有分镜都生成完成，并且有视频 url
  const allDone =
    frames.length > 0 &&
    // frames.every((f) => isReadyForMerge(f.status) && f.videoUrl);
    frames.every((f) => isReadyForMerge(f.status));
  const hasUnacceptedShots = frames.some(
    // (f) => f.hasActions && isReadyForMerge(f.status) && f.status !== "approved",
    (f) => isReadyForMerge(f.status) && f.status !== "approved",
  );
  const canCompose = allDone && !hasUnacceptedShots;
  const anyGenerating = generatingCount > 0;
  const hasGeneratableFrame = frames.some(
    (frame) => isIdle(frame.status) && frame.canGenerate !== false,
  );
  const hasContent = frames.length > 0;
  const progressPct = hasContent
    ? Math.round((doneCount / frames.length) * 100)
    : 0;

  if (!hasContent || !referencesReady) {
    return (
      <div className="flex h-full min-h-0 flex-col">
        {memoryBank ? <div className="shrink-0">{memoryBank}</div> : null}
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center px-8 text-center">
        <div className="relative mb-6">
          <div className="absolute -inset-4 rounded-2xl bg-foreground/[0.02]" />
          <div className="relative grid h-14 w-14 place-items-center rounded-2xl bg-foreground/[0.04] ring-1 ring-inset ring-foreground/[0.06]">
            <Clapperboard className="h-6 w-6 text-foreground/40" />
          </div>
        </div>
        <p className="text-[15px] font-semibold text-foreground/90">
          Developing shots...
        </p>
        <p className="mt-2 max-w-xs text-sm leading-relaxed text-muted-foreground">
          The director is shaping each shot for production.
        </p>
        </div>
      </div>
    );
  }

  // if (!referencesReady) {
  //   return (
  //     <div className="flex h-full flex-col items-center justify-center px-8 text-center">
  //       <Loader2 className="mb-4 h-8 w-8 animate-spin text-foreground/35" />
  //       <p className="text-[15px] font-semibold text-foreground/90">
  //         正在规划参考镜头
  //       </p>
  //       <p className="mt-2 max-w-xs text-sm leading-relaxed text-muted-foreground">
  //         Agent 正在为每个分镜选择参考镜头，完成后即可手动生成。
  //       </p>
  //     </div>
  //   );
  // }

  return (
    <div className="flex h-full flex-col">
      <style>{injectCSS}</style>

      {/* ── status bar ── */}
      <div className="shrink-0 border-b border-border/40">
        <div className="flex items-center justify-between px-4 py-2.5">
          <div className="flex items-center gap-3">
            <span className="text-[12px] font-medium tabular-nums text-foreground/70">
              {frames.filter((f) => isApproved(f.status)).length}/
              {frames.length}
              <span className="ml-1 font-normal text-muted-foreground">
                Approved
              </span>
            </span>
            {/* <span className="text-[12px] font-medium tabular-nums text-foreground/70">
              {doneCount}/{frames.length}
              <span className="ml-1 font-normal text-muted-foreground">
                Complete
              </span>
            </span> */}
            {anyGenerating && (
              <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground/70">
                <Loader2 className="h-3 w-3 animate-spin" />
                {generatingCount} generating
              </span>
            )}
            {errorCount > 0 && !anyGenerating && (
              <span className="flex items-center gap-1 text-[11px] text-foreground/40">
                <AlertCircle className="h-3 w-3" />
                {errorCount} failed
              </span>
            )}
          </div>
          {/* {!allDone && (
            <Button
              size="sm"
              onClick={onGenerateAll}
              disabled={batchGenerating || !hasGeneratableFrame}
              className={cn(
                "h-7 gap-1.5 rounded-lg px-3 text-[11px] font-medium",
                "bg-foreground text-background hover:bg-foreground/90",
              )}
            >
              {batchGenerating ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Sparkles className="h-3 w-3" />
              )}
              Generate All
            </Button>
          )} */}
          {allDone && hasUnacceptedShots && (
            <Button
              size="sm"
              onClick={onAcceptAll}
              disabled={acceptAllBusy}
              className={cn(
                "h-7 gap-1.5 rounded-lg px-3 text-[11px] font-medium",
                "bg-foreground text-background hover:bg-foreground/90",
              )}
            >
              {acceptAllBusy ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Check className="h-3 w-3" />
              )}
              Approve All Shots
            </Button>
          )}
        </div>

        {(anyGenerating || (doneCount > 0 && !allDone)) && (
          <div className="px-4 pb-2.5">
            <div className="h-[3px] overflow-hidden rounded-full bg-foreground/[0.06]">
              <div
                className="h-full rounded-full bg-foreground/20 transition-all duration-700 ease-out"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* ── timeline ── */}
      <div className="min-h-0 flex-1 overflow-y-auto py-4 pr-4 pl-0 scrollbar-thin">
        {memoryBank ? (
          <div className="sticky top-0 z-30 mb-3 bg-background/95 pb-1 backdrop-blur-sm">
            {memoryBank}
          </div>
        ) : null}
        <div className="relative">
          {/* continuous vertical line behind all badges */}
          <div
            className="absolute left-7 top-0 bottom-0 w-px bg-border/40"
            style={{ zIndex: 0 }}
          />

          {
            frames.reduce<{ startSec: number; elements: React.ReactNode[] }>(
              (acc, frame, idx) => {
                const dur = frame.durationSec ?? 5;
                acc.elements.push(
                  <div key={frame.id} className="relative flex">
                    {/* ── left rail: badge ── */}
                    <div className="relative z-10 flex w-14 shrink-0 justify-center pt-3">
                      <div
                        className={cn(
                          "grid h-7 w-7 place-items-center rounded-full text-[11px] font-semibold tabular-nums transition-all duration-500",
                          "bg-background ring-1",
                          isDone(frame.status) &&
                            "text-foreground/70 ring-foreground/20",
                          isActive(frame.status) &&
                            "text-foreground/50 ring-foreground/15",
                          isError(frame.status) &&
                            "text-foreground/40 ring-foreground/25",
                          isIdle(frame.status) &&
                            "text-muted-foreground/40 ring-border/60",
                          frame.accepted === true && "bg-foreground/10",
                        )}
                      >
                        {isDone(frame.status) ? (
                          <Check className="h-3 w-3" strokeWidth={2.5} />
                        ) : isActive(frame.status) ? (
                          <Loader2 className="h-3 w-3 animate-spin" />
                        ) : isError(frame.status) ? (
                          <X className="h-3 w-3" strokeWidth={2.5} />
                        ) : (
                          idx + 1
                        )}
                      </div>
                    </div>

                    {/* ── card ── */}
                    <div className="min-w-0 flex-1 pb-4">
                      <ShotCard
                        frame={frame}
                        index={idx}
                        memorySlots={renderMemorySlots?.(frame.id)}
                        startSec={acc.startSec}
                        durationSec={dur}
                        disableGenerate={batchGenerating}
                        onGenerate={() => onGenerate(frame.id)}
                        onUpdatePrompt={(p) => onUpdatePrompt(frame.id, p)}
                        onUpdateDuration={(d) => onUpdateDuration(frame.id, d)}
                        onRetry={() => onRetry(frame.id)}
                        onAccept={() => onAccept(frame.id)}
                        onRevise={(feedback) => onRevise(frame.id, feedback)}
                        onSetContinuousMode={
                          onSetContinuousMode
                            ? (enabled) =>
                                onSetContinuousMode(frame.id, enabled)
                            : undefined
                        }
                        nextContinuous={
                          idx + 1 < frames.length && onSetContinuousMode
                            ? {
                                enabled:
                                  frames[idx + 1].continuousEnabled ?? false,
                                disabled: isContinuousLocked(
                                  frames[idx + 1].status,
                                ),
                                onToggle: (enabled: boolean) =>
                                  onSetContinuousMode(
                                    frames[idx + 1].id,
                                    enabled,
                                  ),
                              }
                            : null
                        }
                        reviewBusy={busyFrameId === frame.id}
                        onPromptEditingChange={onPromptEditingChange}
                        firstFrame={
                          !renderMemorySlots && idx === 0 && shot1FirstFrame
                            ? shot1FirstFrame
                            : null
                        }
                      />
                    </div>
                  </div>,
                );
                acc.startSec += dur;
                return acc;
              },
              { startSec: 0, elements: [] },
            ).elements
          }
        </div>
      </div>

      {/* ── bottom: 下一步 (consistent with step 1 & 2) ── */}
      <div className="shrink-0 border-t border-border/50 px-4 py-3">
        <Button
          onClick={onCompose}
          disabled={!canCompose || composeDisabled}
          className={cn(
            "w-full rounded-lg text-[13px] font-medium",
            "bg-foreground text-background",
            "hover:bg-foreground/90 active:scale-[0.99]",
            "h-9 transition-all duration-200",
            "disabled:opacity-40",
          )}
        >
          Next Step
          <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────── */

function ShotCard({
  frame,
  index,
  startSec = 0,
  durationSec = 5,
  disableGenerate = false,
  onGenerate,
  onUpdatePrompt,
  onUpdateDuration,
  onRetry,
  onAccept,
  onRevise,
  onSetContinuousMode,
  // nextContinuous,
  reviewBusy = false,
  onPromptEditingChange,
  firstFrame = null,
  memorySlots,
}: {
  frame: Frame;
  index: number;
  startSec?: number;
  durationSec?: number;
  disableGenerate?: boolean;
  onGenerate: () => void;
  onUpdatePrompt: (prompt: string) => void;
  onUpdateDuration: (durationSec: number) => void;
  onRetry: () => void;
  onAccept: () => void;
  onRevise: (feedback: string) => void | Promise<void>;
  onSetContinuousMode?: (enabled: boolean) => void;
  /** Next shot's continuous mode toggle data, for display in the memory box. */
  nextContinuous?: {
    enabled: boolean;
    disabled?: boolean;
    onToggle: (enabled: boolean) => void;
  } | null;
  reviewBusy?: boolean;
  onPromptEditingChange?: (editing: boolean) => void;
  firstFrame?: Shot1FirstFrameControls | null;
  memorySlots?: ReactNode;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(frame.prompt || frame.segmentText);
  const [dirty, setDirty] = useState(false);
  const [durInput, setDurInput] = useState(String(durationSec));
  const [revising, setRevising] = useState(false);
  const [reviseFeedback, setReviseFeedback] = useState("");
  const [revisionSubmitted, setRevisionSubmitted] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const reviseRef = useRef<HTMLTextAreaElement>(null);

  const [textOpen, setTextOpen] = useState(false);
  const [isTruncated, setIsTruncated] = useState(false);
  const [detailOpen, setDetailOpen] = useState(false);
  const [captionCopied, setCaptionCopied] = useState(false);
  const [memoryLightboxIndex, setMemoryLightboxIndex] = useState<number | null>(
    null,
  );
  const clampedMeasureRef = useRef<HTMLParagraphElement>(null);
  const fullMeasureRef = useRef<HTMLParagraphElement>(null);

  const firstFrameDisplayUrl = firstFrame?.displayUrl || null;
  const [firstFrameLightboxOpen, setFirstFrameLightboxOpen] = useState(false);

  const segmentText = frame.segmentText || frame.prompt;

  const memoryPreviewImages = useMemo<UIImage[]>(
    () =>
      (frame.generationMemories ?? [])
        .filter((memory) => Boolean(memory.image?.url))
        .map((memory) => ({
          url: memory.image.url,
          name: memory.image.name ?? memoryDisplayName(memory),
        })),
    [frame.generationMemories],
  );

  const memoryPreviewIndexById = useMemo(() => {
    const map = new Map<string, number>();
    let i = 0;
    for (const memory of frame.generationMemories ?? []) {
      if (!memory.image?.url) continue;
      map.set(memory.id, i);
      i += 1;
    }
    return map;
  }, [frame.generationMemories]);

  useLayoutEffect(() => {
    if (editing) return;

    const clamped = clampedMeasureRef.current;
    const full = fullMeasureRef.current;
    if (!clamped || !full) return;

    const check = () => {
      const truncated = full.scrollHeight > clamped.clientHeight;
      setIsTruncated(truncated);
      if (!truncated) setTextOpen(false);
    };

    check();
    const ro = new ResizeObserver(check);
    ro.observe(clamped);
    ro.observe(full);
    return () => ro.disconnect();
  }, [segmentText, editing, textOpen]);

  const canReviewShot =
    frame.status === "generated" || frame.status === "review_pass";
  const displayedVideoUrl = frame.videoUrl;
  const serverRevisionPending =
    frame.status === "review_fail" || frame.status === "queued";
  const revisionLocked = revisionSubmitted || serverRevisionPending;
  const showActions =
    // frame.hasActions === true && canReviewShot && !revisionLocked;
    canReviewShot && !revisionLocked;

  useEffect(() => {
    if (!editing) setDraft(frame.prompt || frame.segmentText);
  }, [frame.prompt, frame.segmentText, editing]);
  useEffect(() => setDurInput(String(durationSec)), [durationSec]);
  useEffect(() => {
    if (editing && taRef.current) {
      const el = taRef.current;
      el.focus();
      el.style.height = "auto";
      el.style.height = `${el.scrollHeight}px`;
    }
  }, [editing]);

  useEffect(() => {
    onPromptEditingChange?.(editing);
    return () => {
      if (editing) onPromptEditingChange?.(false);
    };
  }, [editing, onPromptEditingChange]);

  useEffect(() => {
    if (revisionSubmitted && !canReviewShot) {
      setRevisionSubmitted(false);
    }
  }, [canReviewShot, revisionSubmitted]);

  useEffect(() => {
    if (revisionLocked) {
      setRevising(false);
    }
  }, [revisionLocked]);

  useEffect(() => {
    if (revising && reviseRef.current) {
      reviseRef.current.focus();
    }
  }, [revising]);

  const submitRevise = useCallback(async () => {
    const trimmed = reviseFeedback.trim();
    if (!trimmed) {
      setLocalError("Describe the requested changes first.");
      return;
    }
    setLocalError(null);
    setRevisionSubmitted(true);
    setRevising(false);
    try {
      await onRevise(trimmed);
      setReviseFeedback("");
    } catch (error) {
      setRevisionSubmitted(false);
      setRevising(true);
      setLocalError(
        error instanceof Error ? error.message : "Revision could not be submitted.",
      );
    }
  }, [onRevise, reviseFeedback]);

  const save = useCallback(() => {
    const trimmed = draft.trim();
    onUpdatePrompt(trimmed);
    setEditing(false);
    if (trimmed !== frame.prompt && isDone(frame.status)) setDirty(true);
  }, [draft, frame.prompt, frame.status, onUpdatePrompt]);

  const cancel = useCallback(() => {
    setDraft(frame.prompt || frame.segmentText);
    setEditing(false);
  }, [frame.prompt, frame.segmentText]);

  useEffect(() => {
    if (!editing) return;
    const onKey = (e: KeyboardEvent) => {
      if (document.activeElement !== taRef.current) return;
      if (e.isComposing || e.keyCode === 229) return;
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        cancel();
        return;
      }
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        save();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [editing, cancel, save]);

  const commitDuration = useCallback(
    (raw: string) => {
      const n = Math.max(
        1,
        Math.min(10, Math.round(Number(raw) || durationSec)),
      );
      setDurInput(String(n));
      if (n !== durationSec) onUpdateDuration(n);
    },
    [durationSec, onUpdateDuration],
  );

  const showRegen = dirty && isDone(frame.status);

  const fmtTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };
  const endSec = startSec + durationSec;
  const timeLabel = `${fmtTime(startSec)} – ${fmtTime(endSec)}`;

  return (
    <div
      role={revising ? "group" : undefined}
      aria-label={revising ? `Revise Shot ${index + 1}` : undefined}
    >
      {memorySlots}
      <article
        className={cn(
          "group/shot overflow-hidden rounded-2xl transition-all duration-300",
          "bg-gradient-to-b from-card/80 to-card/40",
          "border border-border/40",
          "shadow-[0_1px_3px_0_rgba(0,0,0,0.03)]",
          isActive(frame.status) &&
            "border-foreground/12 shadow-[0_2px_12px_0_rgba(0,0,0,0.06)]",
          isDone(frame.status) && "border-foreground/8",
          isError(frame.status) && "border-foreground/15",
        )}
      >
      {/* ── header ── */}
      <div className="flex items-center gap-2.5 px-3.5 py-2.5">
        <span className="text-[10px] font-bold uppercase tracking-[0.12em] text-muted-foreground/45">
          Shot {index + 1}
        </span>
        <span className="rounded bg-foreground/[0.04] px-1.5 py-0.5 text-[9px] tabular-nums text-muted-foreground/40">
          {timeLabel}
        </span>
        {isIdle(frame.status) || revising ? (
          <div className="flex items-center gap-px rounded-md bg-foreground/[0.04]">
            <button
              type="button"
              disabled={durationSec <= 1}
              onClick={() => commitDuration(String(durationSec - 1))}
              className={cn(
                "grid h-5 w-5 place-items-center rounded-l-md transition-colors",
                "text-muted-foreground/40 hover:bg-foreground/[0.06] hover:text-foreground/60",
                "disabled:opacity-25 disabled:pointer-events-none",
              )}
            >
              <Minus className="h-2.5 w-2.5" strokeWidth={2.5} />
            </button>
            <div className="flex items-baseline gap-[1px] px-1">
              <input
                type="text"
                inputMode="numeric"
                value={durInput}
                onChange={(e) =>
                  setDurInput(e.target.value.replace(/[^\d]/g, ""))
                }
                onBlur={(e) => commitDuration(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter")
                    commitDuration((e.target as HTMLInputElement).value);
                }}
                className={cn(
                  "w-[2ch] bg-transparent text-center text-[9px] font-medium tabular-nums text-muted-foreground/55",
                  "focus:outline-none focus:text-foreground/70",
                )}
              />
              <span className="text-[8px] text-muted-foreground/30">s</span>
            </div>
            <button
              type="button"
              disabled={durationSec >= 10}
              onClick={() => commitDuration(String(durationSec + 1))}
              className={cn(
                "grid h-5 w-5 place-items-center rounded-r-md transition-colors",
                "text-muted-foreground/40 hover:bg-foreground/[0.06] hover:text-foreground/60",
                "disabled:opacity-25 disabled:pointer-events-none",
              )}
            >
              <Plus className="h-2.5 w-2.5" strokeWidth={2.5} />
            </button>
          </div>
        ) : (
          <div className="flex h-5 items-center gap-px rounded-md bg-foreground/[0.04]">
            <div className="flex items-baseline gap-[1px] px-1">
              <input
                type="text"
                inputMode="numeric"
                value={durInput}
                readOnly={true}
                disabled={true}
                className={cn(
                  "w-[2ch] bg-transparent text-center text-[9px] font-medium tabular-nums text-muted-foreground/55",
                  "focus:outline-none focus:text-foreground/70",
                )}
              />
              <span className="text-[8px] text-muted-foreground/30">s</span>
            </div>
          </div>
        )}
        <div className="ml-auto flex items-center gap-2">
          {frame.caption && frame.caption.trim() && (
            <button
              type="button"
              onClick={() => setDetailOpen(true)}
              title="View generation request"
              className={cn(
                "inline-flex items-center gap-1 rounded-full px-2 py-0.5",
                "border border-foreground/10 bg-foreground/[0.03]",
                "text-[10px] font-medium text-muted-foreground/55",
                "transition-all hover:bg-foreground/[0.06] hover:text-foreground/70",
              )}
            >
              <FileText className="h-2.5 w-2.5" />
              Request
            </button>
          )}
          {frame.status === "approved" && (
            <span className="rounded-full bg-foreground/[0.06] px-2.5 py-0.5 text-[10px] font-medium text-foreground/55">
              Approved
            </span>
          )}
          {isDone(frame.status) && !dirty && (
            <span className="rounded-full bg-foreground/[0.06] px-2.5 py-0.5 text-[10px] font-medium text-foreground/55">
              Complete
            </span>
          )}
          {isActive(frame.status) && !disableGenerate && (
            <span className="inline-flex items-center gap-1 rounded-full bg-foreground/[0.05] px-2.5 py-0.5 text-[10px] font-medium text-foreground/45">
              <Loader2 className="h-2.5 w-2.5 animate-spin" />
              Generating
            </span>
          )}
          {isActive(frame.status) && disableGenerate && (
            <span className="rounded-full bg-foreground/[0.05] px-2.5 py-0.5 text-[10px] font-medium text-foreground/40">
              Generating
            </span>
          )}
          {isError(frame.status) && (
            <button
              type="button"
              onClick={onRetry}
              className={cn(
                "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5",
                "border border-foreground/10 bg-foreground/[0.03]",
                "text-[10px] font-medium text-foreground/50",
                "transition-all hover:bg-foreground/[0.06] hover:text-foreground/70",
              )}
            >
              <RefreshCcw className="h-2.5 w-2.5" />
              Retry
            </button>
          )}
          {isIdle(frame.status) && (
            <button
              type="button"
              onClick={onGenerate}
              disabled={frame.canGenerate === false || disableGenerate}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-3 py-1",
                "border border-border/50 bg-background/80",
                "text-[11px] font-medium text-muted-foreground/70",
                "transition-all hover:border-foreground/15 hover:text-foreground/70",
                "disabled:cursor-not-allowed disabled:opacity-40",
              )}
            >
              <Play className="h-3 w-3" />
              Generate
            </button>
          )}
          {showRegen && (
            <button
              type="button"
              onClick={() => {
                setDirty(false);
                onGenerate();
              }}
              className={cn(
                "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5",
                "border border-foreground/12 bg-foreground/[0.03]",
                "text-[10px] font-medium text-foreground/50",
                "transition-all hover:bg-foreground/[0.06]",
              )}
            >
              <RefreshCcw className="h-2.5 w-2.5" />
              Regenerate
            </button>
          )}
          {index > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="select-none text-[10px] font-medium text-muted-foreground/50">
                Continuous
              </span>
              <Switch
                checked={frame.continuousEnabled ?? false}
                disabled={isContinuousLocked(frame.status)}
                onCheckedChange={(enabled) => onSetContinuousMode?.(enabled)}
                aria-label="Continuous first frame"
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
                    Use the previous shot's final frame as this shot's opening frame.
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          )}
        </div>
      </div>

      {isIdle(frame.status) && frame.hintMessage ? (
        <div
          role="status"
          className="mx-3.5 mb-2 rounded-lg border border-amber-500/15 bg-amber-500/[0.06] px-2.5 py-2 text-[11px] leading-5 text-amber-700 dark:text-amber-300"
        >
          {frame.hintMessage}
        </div>
      ) : null}

      {isIdle(frame.status) && frame.dependencyMessage ? (
        <div
          role="status"
          className="mx-3.5 mb-2 rounded-lg border border-amber-500/15 bg-amber-500/[0.06] px-2.5 py-2 text-[11px] leading-5 text-amber-700 dark:text-amber-300"
        >
          {frame.dependencyMessage}
        </div>
      ) : null}

      {/* {frame.referenceShotIds !== undefined ? (
        <div className="mx-3.5 mb-2 rounded-lg bg-foreground/[0.03] px-2.5 py-2 text-[11px] leading-5 text-muted-foreground">
          <p>
            Reference shots:
            {frame.referenceShotIds.length > 0
              ? frame.referenceShotIds.join("、")
              : "None"}
          </p>
          {frame.referenceNote ? (
            <p className="mt-1 text-foreground/55">{frame.referenceNote}</p>
          ) : null}
          {frame.dependencyMessage ? (
            <p className="mt-1 text-amber-700 dark:text-amber-300">
              {frame.dependencyMessage}
            </p>
          ) : null}
        </div>
      ) : null} */}

      {/* ── script preview (+ shot1 first-frame uploader) ── */}
      <div className="px-3.5">
        <div className="mb-2">
          {(isTruncated || textOpen) && !editing && (
            <button
              type="button"
              onClick={() => setTextOpen(!textOpen)}
              className={cn(
                "flex w-full items-center gap-1.5 rounded-lg py-1 text-left text-[11px] transition-colors",
                "text-muted-foreground/45 hover:text-foreground/55",
              )}
            >
              <ChevronDown
                className={cn(
                  "h-3 w-3 shrink-0 transition-transform duration-200",
                  textOpen && "rotate-180",
                )}
              />
              {textOpen ? "Collapse" : "Expand"}
              <span className="ml-1 text-[10px] text-muted-foreground/25">
                {frame.segmentText.length} chars
              </span>
            </button>
          )}

          {!editing && (
            <div
              className={cn(
                firstFrameDisplayUrl && "flex items-center gap-3 pt-1.5 pr-1.5",
              )}
            >
              {firstFrameDisplayUrl ? (
                <div className="relative h-16 w-16 shrink-0">
                  <div className="relative h-full w-full overflow-hidden rounded-xl border border-foreground/10 bg-foreground/[0.03]">
                    <button
                      type="button"
                      className="h-full w-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-foreground/20"
                      onClick={() => setFirstFrameLightboxOpen(true)}
                      aria-label="Preview first-frame reference"
                    >
                      <img
                        src={firstFrameDisplayUrl}
                        alt=""
                        className="h-full w-full object-cover"
                        draggable={false}
                      />
                    </button>
                  </div>
                  <ImageLightbox
                    images={[
                      {
                        url: firstFrameDisplayUrl,
                        name: "first-frame.jpg",
                      },
                    ]}
                    index={firstFrameLightboxOpen ? 0 : null}
                    onIndexChange={() => {}}
                    onOpenChange={(open) => {
                      if (!open) setFirstFrameLightboxOpen(false);
                    }}
                  />
                </div>
              ) : null}
              <div className="relative min-w-0 flex-1">
                <div
                  className="pointer-events-none invisible absolute inset-x-0 top-0 -z-10 w-full"
                  aria-hidden
                >
                  <p
                    ref={clampedMeasureRef}
                    className="mt-0.5 line-clamp-3 text-[13px] leading-[1.7]"
                  >
                    {segmentText}
                  </p>
                  <p
                    ref={fullMeasureRef}
                    className="mt-0.5 text-[13px] leading-[1.7]"
                  >
                    {segmentText}
                  </p>
                </div>

                {!textOpen && (
                  <p
                    className={cn(
                      "mt-0.5 text-[13px] leading-[1.7] text-foreground/40",
                      isTruncated && "line-clamp-3",
                    )}
                  >
                    {segmentText}
                  </p>
                )}
                {textOpen && (
                  <p className="mt-0.5 text-[13px] leading-[1.7] text-foreground/40">
                    {segmentText}
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── preview ── */}
      <div className="px-3.5 pb-2.5">
        {frame.generationMemories && frame.generationMemories.length > 0 ? (
          <div className="mb-2.5 rounded-xl border border-foreground/8 bg-foreground/[0.015] px-2.5 py-2">
            <div className="mb-1.5 text-[10px] font-medium text-foreground/45">
              Memories used for this shot
            </div>
            <div className="flex gap-2 overflow-x-auto pb-0.5">
              {frame.generationMemories.map((memory) => {
                const label = memoryDisplayName(memory);
                const previewIndex = memoryPreviewIndexById.get(memory.id);
                return (
                  <div
                    key={memory.id}
                    className="w-24 shrink-0 overflow-hidden rounded-lg border border-foreground/8 bg-background/70"
                  >
                    <button
                      type="button"
                      onClick={() => {
                        if (previewIndex != null) {
                          setMemoryLightboxIndex(previewIndex);
                        }
                      }}
                      disabled={previewIndex == null}
                      aria-label={`Preview ${label} memory image`}
                      className="block w-full text-left disabled:cursor-default"
                    >
                      <img
                        src={memory.image.url}
                        alt={`${label} Memory`}
                        className="h-14 w-full object-cover"
                      />
                    </button>
                    <div className="px-1.5 py-1">
                      <div
                        className="truncate text-[9px] font-semibold text-foreground/60"
                        title={label}
                      >
                        {label}
                      </div>
                      {memory.audio?.url ? (
                        <audio
                          src={memory.audio.url}
                          controls
                          preload="none"
                          aria-label={`Play ${label} memory audio`}
                          className="mt-1 h-5 w-full"
                        />
                      ) : null}
                    </div>
                  </div>
                );
              })}
            </div>
            <ImageLightbox
              images={memoryPreviewImages}
              index={memoryLightboxIndex}
              onIndexChange={setMemoryLightboxIndex}
              onOpenChange={(open) => {
                if (!open) setMemoryLightboxIndex(null);
              }}
            />
            {/* {nextContinuous ? (
              <div className="mt-2 flex items-center gap-1.5 border-t border-border/30 pt-2">
                <span className="text-[10px] text-muted-foreground/50">
                  Shot {index + 2} continuous first frame
                </span>
                <Switch
                  checked={nextContinuous.enabled}
                  disabled={nextContinuous.disabled}
                  onCheckedChange={nextContinuous.onToggle}
                  aria-label={`Enable continuous first frame for Shot ${index + 2}`}
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
                      Use the previous shot's final frame as the next opening frame.
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            ) : null} */}
          </div>
        ) : null}
        {isDone(frame.status) && displayedVideoUrl ? (
          <ShotVideoPlayer src={displayedVideoUrl} />
        ) : isDone(frame.status) && !frame.videoUrl ? (
          <div className="overflow-hidden rounded-xl border border-foreground/8 bg-foreground/[0.02]">
            <div className="flex aspect-video w-full items-center justify-center">
              <div className="flex flex-col items-center gap-2.5">
                <div className="grid h-12 w-12 place-items-center rounded-full bg-foreground/[0.05] ring-1 ring-inset ring-foreground/[0.08]">
                  <Check
                    className="h-5 w-5 text-foreground/35"
                    strokeWidth={2}
                  />
                </div>
                <span className="text-[11px] font-medium text-foreground/40">
                  Generation complete
                </span>
              </div>
            </div>
          </div>
        ) : isActive(frame.status) ? (
          <div className="relative overflow-hidden rounded-xl border border-foreground/8 bg-foreground/[0.01] shadow-sm">
            <div className="relative flex aspect-video w-full items-center justify-center">
              <div
                className="absolute inset-0"
                style={{
                  background:
                    "linear-gradient(90deg, transparent 25%, var(--foreground) 50%, transparent 75%)",
                  backgroundSize: "200% 100%",
                  opacity: 0.03,
                  animation: "shot-shimmer 2.5s ease-in-out infinite",
                }}
              />
              <div className="absolute inset-0 bg-[length:28px_28px] bg-[linear-gradient(to_right,var(--border)_1px,transparent_1px),linear-gradient(to_bottom,var(--border)_1px,transparent_1px)] opacity-[0.06]" />
              <div
                className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-foreground/15 to-transparent"
                style={{ animation: "shot-scan-line 3s linear infinite" }}
              />
              <div className="relative flex flex-col items-center gap-3">
                <div className="relative">
                  <div
                    className="absolute -inset-4 rounded-full border border-foreground/[0.06]"
                    style={{
                      animation: "shot-pulse-ring 2.5s ease-in-out infinite",
                    }}
                  />
                  <div
                    className="absolute -inset-7 rounded-full border border-foreground/[0.03]"
                    style={{
                      animation:
                        "shot-pulse-ring 2.5s ease-in-out infinite 0.4s",
                    }}
                  />
                  <Loader2 className="relative h-6 w-6 animate-spin text-foreground/30" />
                </div>
                <div className="text-center">
                  <p className="text-[11px] font-medium text-foreground/40">
                    Generating video
                  </p>
                  <p className="mt-0.5 text-[9px] text-muted-foreground/30">
                    Shot {index + 1} · Please wait
                  </p>
                </div>
              </div>
            </div>
          </div>
        ) : isError(frame.status) ? (
          <div className="overflow-hidden rounded-xl border border-foreground/10 bg-foreground/[0.015]">
            <div className="flex aspect-video w-full items-center justify-center">
              <div className="flex flex-col items-center gap-2.5 text-center">
                <div className="grid h-12 w-12 place-items-center rounded-full bg-foreground/[0.04] ring-1 ring-inset ring-foreground/[0.08]">
                  <Circle className="h-5 w-5 text-foreground/25" />
                </div>
                <div>
                  <p className="text-[11px] font-medium text-foreground/45">
                    Generation failed
                  </p>
                  <p className="mt-0.5 max-w-[220px] text-[10px] leading-relaxed text-muted-foreground/40">
                    {frame.error || "Check the request and try again."}
                  </p>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="overflow-hidden rounded-xl border border-border/25 bg-foreground/[0.01]">
            <div className="flex aspect-video w-full items-center justify-center">
              <div className="flex flex-col items-center gap-2">
                <div className="grid h-10 w-10 place-items-center rounded-full bg-foreground/[0.03] ring-1 ring-inset ring-foreground/[0.05]">
                  <Film className="h-4 w-4 text-foreground/12" />
                </div>
                <span className="text-[13px] text-muted-foreground/25">
                  Use the action above to start production.
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {frame.reviewNotes ? (
        <div className="mx-3.5 mb-2 rounded-lg bg-foreground/[0.03] px-2.5 py-2 text-[11px] leading-5 text-muted-foreground">
          {frame.reviewNotes}
        </div>
      ) : null}
      {/* {frame.status === "approved" ? (
        <div className="mx-3.5 mb-2 rounded-lg border border-emerald-500/20 bg-emerald-500/8 px-2.5 py-2 text-[11px] text-emerald-800 dark:text-emerald-200">
          This shot is approved and will be included in the final cut.
        </div>
      ) : null} */}
      {/* ── accept / revise actions ── */}
      {showActions && isDone(frame.status) ? (
        <div className="px-3.5 pb-2.5">
          {!revising ? (
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => void onAccept()}
                disabled={reviewBusy}
                className={cn(
                  "flex h-9 flex-1 items-center justify-center gap-1.5 rounded-lg py-2",
                  "text-[12px] font-medium text-background",
                  "bg-foreground transition-all duration-200",
                  "hover:bg-foreground/90",
                  "active:scale-[0.99]",
                  "disabled:pointer-events-none disabled:opacity-40",
                )}
              >
                <ThumbsUp className="h-3.5 w-3.5" />
                Approve
              </button>
              <button
                type="button"
                onClick={() => {
                  setRevising(true);
                  setLocalError(null);
                }}
                disabled={reviewBusy}
                className={cn(
                  "flex flex-1 items-center justify-center gap-1.5 rounded-lg py-2",
                  "border border-foreground/10 bg-foreground/[0.03]",
                  "text-[12px] font-medium text-foreground/60",
                  "transition-all duration-200",
                  "hover:border-foreground/20 hover:bg-foreground/[0.07] hover:text-foreground/80",
                  "active:scale-[0.98]",
                  "disabled:pointer-events-none disabled:opacity-40",
                )}
              >
                <SquarePen className="h-3.5 w-3.5" />
                Revise
              </button>
            </div>
          ) : (
            <div className="rounded-lg border border-foreground/10 bg-foreground/[0.02] p-2.5">
              {memorySlots ? (
                <p className="mb-2 text-[10px] leading-4 text-muted-foreground/55">
                  Adjust the Memory inputs above if needed. The visible draft is
                  applied before this revision is generated.
                </p>
              ) : null}
              <textarea
                ref={reviseRef}
                value={reviseFeedback}
                onChange={(e) => setReviseFeedback(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && e.metaKey && reviseFeedback.trim()) {
                    e.preventDefault();
                    submitRevise();
                  }
                  if (e.key === "Escape") {
                    setReviseFeedback("");
                    setRevising(false);
                    setLocalError(null);
                  }
                }}
                placeholder="Describe what should change..."
                rows={2}
                className={cn(
                  "w-full resize-none rounded-md bg-background/80 px-2.5 py-2",
                  "border border-foreground/8",
                  "text-[13px] leading-relaxed text-foreground/70",
                  "placeholder:text-muted-foreground/30",
                  "focus:border-foreground/15 focus:outline-none",
                  "transition-colors duration-150",
                )}
              />
              {localError ? (
                <p className="mt-1.5 text-[10px] text-destructive">
                  {localError}
                </p>
              ) : null}
              <div className="mt-2 flex items-center justify-between">
                <span className="text-[9px] text-muted-foreground/30">
                  Cmd+Enter submit · Esc cancel
                </span>
                <div className="flex gap-1.5">
                  <button
                    type="button"
                    onClick={() => {
                      setReviseFeedback("");
                      setRevising(false);
                      setLocalError(null);
                    }}
                    disabled={reviewBusy}
                    className="rounded-md px-2.5 py-1 text-[11px] text-muted-foreground/50 transition-colors hover:bg-foreground/[0.05] hover:text-foreground/60 disabled:opacity-40"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    disabled={!reviseFeedback.trim() || reviewBusy}
                    onClick={submitRevise}
                    className={cn(
                      "flex items-center gap-1 rounded-md px-3 py-1",
                      "bg-foreground text-[11px] font-medium text-background",
                      "transition-all duration-150",
                      "hover:bg-foreground/90 active:scale-[0.97]",
                      "disabled:opacity-30 disabled:pointer-events-none",
                    )}
                  >
                    <Send className="h-3 w-3" />
                    Submit
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      ) : null}

      <Sheet open={detailOpen} onOpenChange={setDetailOpen}>
        <SheetContent
          side="left"
          className="w-full gap-0 overflow-y-auto sm:max-w-md"
        >
          <SheetHeader>
            <SheetTitle>Shot {index + 1} · Generation Request</SheetTitle>
          </SheetHeader>
          <div className="mt-4 space-y-4 text-sm">
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-[13px]">
              <dt className="text-muted-foreground/60">shot_id</dt>
              <dd className="tabular-nums text-foreground/80">
                {frame.shotId ?? "—"}
              </dd>
              <dt className="text-muted-foreground/60">Internal cut</dt>
              <dd className="text-foreground/80">
                {frame.cut === undefined ? "—" : frame.cut ? "Yes" : "No"}
              </dd>
              <dt className="text-muted-foreground/60">Duration</dt>
              <dd className="tabular-nums text-foreground/80">
                {durationSec}s
                {frame.numFrames ? ` (${frame.numFrames} frames)` : ""}
              </dd>
              <dt className="text-muted-foreground/60">Reference frames</dt>
              <dd className="tabular-nums text-foreground/80">
                {frame.referenceShotIds && frame.referenceShotIds.length > 0
                  ? frame.referenceShotIds.map((id) => `#${id}`).join("、")
                  : "None"}
              </dd>
            </dl>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[12px] font-medium text-muted-foreground/70">
                  Full caption sent to the generation service
                </span>
                <button
                  type="button"
                  onClick={() => {
                    void navigator.clipboard
                      ?.writeText(frame.caption ?? "")
                      .then(() => {
                        setCaptionCopied(true);
                        setTimeout(() => setCaptionCopied(false), 1500);
                      });
                  }}
                  className={cn(
                    "inline-flex items-center gap-1 rounded-md px-2 py-1",
                    "border border-foreground/10 bg-foreground/[0.03]",
                    "text-[11px] font-medium text-muted-foreground/60",
                    "transition-all hover:bg-foreground/[0.06] hover:text-foreground/80",
                  )}
                >
                  <Copy className="h-3 w-3" />
                  {captionCopied ? "Copied" : "Copy"}
                </button>
              </div>
              <pre className="max-h-[60vh] overflow-y-auto whitespace-pre-wrap break-words rounded-xl border border-border/40 bg-foreground/[0.02] p-3 font-sans text-[13px] leading-relaxed text-foreground/85 selection:bg-foreground/15">
                {frame.caption}
              </pre>
            </div>
          </div>
        </SheetContent>
      </Sheet>
      </article>
    </div>
  );
}
