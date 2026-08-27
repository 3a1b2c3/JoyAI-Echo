import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronLeft, ChevronRight, LoaderCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { GenerationAlertDialog } from "@/components/ui/GenerationAlertDialog";
import { DEFAULT_VIDEO_SIZE } from "@/components/thread/AspectRatioPicker";
import { GENERATION_BUSY_ALERT_ENABLED } from "@/config/features";
import { useShotGenerationAlerts } from "@/hooks/useShotGenerationAlerts";
import { useWorkplaceContext } from "@/providers/WorkplaceProvider";
import type { WorkplaceData, WorkplaceShot } from "@/lib/types";
import { collectConfirmedMemoryEntries } from "@/lib/memory-bank-history";
import {
  canGenerateShot,
  isMemoryRecommendationReady,
  missingReferenceGenerations,
  previousShotApprovalMessage,
  referenceDependencyMessage,
  shotOneWorkflowHint,
  shotReferenceIds,
} from "@/lib/workplace/generation";
import {
  beatSummaryForShot,
  buildBeatRevisionFeedback,
} from "@/lib/workplace/revision";
import type { CommitAction } from "@/hooks/workplace-editor/types";
import { beatPath } from "@/hooks/workplace-editor/types";
import { serverValueForPath } from "@/hooks/workplace-editor/reducer";
import type { UseWorkplaceEditorResult } from "@/hooks/workplace-editor/useWorkplaceEditor";
import { cn } from "@/lib/utils";
import { ScriptEditor } from "../longvideo/ScriptEditor";
import { StoryboardScriptEditor } from "../longvideo/StoryboardScriptEditor";
import {
  FramesPanel,
  FrameStatus,
  type Shot1FirstFrameControls,
} from "../longvideo/FramesPanel";
import { RenderOverlay, type RenderStatus } from "../longvideo/RenderOverlay";
import { ComposePanel } from "../longvideo/ComposePanel";
import { StepConfirmDialog } from "../longvideo/StepConfirmDialog";
import type { StepId } from "../longvideo/StepHeader";
import { MemoryBankBoard } from "./MemoryBankBoard";
import {
  ShotMemorySlots,
  type ShotMemorySlotsHandle,
} from "./ShotMemorySlots";

/** After start-generation, wait this long for agent prep before offering retry. */
const SHOT_GENERATING_PREP_TIMEOUT_MS = 5 * 60 * 1000;

type WorkplaceTab = "story" | "shots";

type IrreversibleAction =
  | "auto_generate"
  | Extract<
      CommitAction["type"],
      "confirm_story" | "start_generation" | "start_merge"
    >
  | "generate_all";

/** 不可逆工作流动作 → StepConfirmDialog 目标步骤文案 */
const ACTION_TO_STEP: Record<IrreversibleAction, StepId> = {
  confirm_story: 2,
  start_generation: 3,
  generate_all: 3,
  auto_generate: 4,
  start_merge: 4,
};

interface WorkplacePanelProps {
  sessionKey: string | null;
  activeTab: WorkplaceTab;
  onTabChange: (tab: WorkplaceTab) => void;
  onNewChat?: () => Promise<string | null>;
  /** True when the left chat has messages but the story is still empty. */
  showThinking?: boolean;
  /** True while the left chat turn is still running; 02 workflow buttons wait. */
  chatBusy?: boolean;
}

function hasUnsavedBeatChanges(
  workplace: WorkplaceData,
  editor: Pick<UseWorkplaceEditorResult, "segments">,
): boolean {
  return editor.segments.some((segment) => {
    const baseline = serverValueForPath(workplace, beatPath(segment.shotId));
    return segment.text.trim() !== baseline.trim();
  });
}

function stageLabel(stage: string | null | undefined): string {
  switch (stage) {
    case "story_discussion":
      return "Briefing";
    case "story_confirmed":
      return "Story approval";
    case "shot_planning":
      return "Shot planning";
    case "shot_generating":
      return "Shot generation";
    case "shot_reviewing":
      return "Shot review";
    case "shot_revising":
      return "Shot revision";
    case "awaiting_memory_review":
      return "Memory review";
    case "awaiting_memory_build":
      return "Memory build";
    case "merging":
      return "Final assembly";
    case "done":
      return "Complete";
    default:
      return "Preparing";
  }
}

function workflowStepIndex(stage: string | null | undefined): number {
  switch (stage) {
    case "story_discussion":
    case "story_confirmed":
      return 0;
    case "shot_planning":
      return 1;
    case "shot_generating":
    case "shot_reviewing":
    case "shot_revising":
    case "awaiting_memory_review":
    case "awaiting_memory_build":
      return 2;
    case "merging":
      return 3;
    case "done":
      return 4;
    default:
      return 0;
  }
}

function failedAutoGenerateShot(workplace: WorkplaceData | null) {
  return workplace?.shots.find((shot) => shot.status === "error") ?? null;
}

function composeRenderStatus(workplace: WorkplaceData | null): RenderStatus {
  if (!workplace) return "idle";
  if (
    workplace.stage === "done" ||
    Boolean(workplace.final_output_url) ||
    Boolean(workplace.final_video)
  ) {
    return "done";
  }
  if (
    workplace.stage === "error" ||
    workplace.stage === "failed" ||
    failedAutoGenerateShot(workplace)
  ) {
    return "error";
  }
  if (workplace.stage === "awaiting_memory_review") {
    return "idle";
  }
  if (workplace.auto_generate || workplace.stage === "merging") {
    return "rendering";
  }
  return "idle";
}

function composeRenderError(workplace: WorkplaceData | null): string | undefined {
  const failed = failedAutoGenerateShot(workplace);
  const message =
    (typeof workplace?.generation_error === "string" &&
      workplace.generation_error) ||
    failed?.generation_error ||
    "";
  return message.trim() || undefined;
}

function resolveWorkflowIndex(workplace: WorkplaceData | null): number {
  const progress = workplace?.progress;
  if (progress === "done") return 4;
  // 一键成片不要跟 progress 02/03 走进逐镜打磨；重新生成过程中也停在合成页。
  if (workplace?.auto_generate) {
    if (
      workplace.stage === "done" ||
      workplace.final_output_url ||
      workplace.final_video
    ) {
      return 4;
    }
    return 3;
  }
  // 逐镜打磨：Memory 审核 / 单镜失败仍停在 03，避免 progress 短暂报 02 打回分镜脚本。
  if (
    workplace?.stage === "awaiting_memory_review" ||
    workplace?.stage === "awaiting_memory_build" ||
    workplace?.stage === "failed" ||
    workplace?.stage === "shot_generating" ||
    workplace?.stage === "shot_reviewing" ||
    workplace?.stage === "shot_revising"
  ) {
    return 2;
  }
  if (progress === "01") return 0;
  if (progress === "02") return 1;
  if (progress === "03") return 2;
  if (progress === "04") return 3;
  return workflowStepIndex(workplace?.stage);
}

function shotIdFromKey(shots: WorkplaceShot[], shotKey: string): number | null {
  const shot = shots.find((item) => item.shot_key === shotKey);
  return shot?.shot_id ?? null;
}

const WORKFLOW_STEPS = [
  { id: 1, label: "Story" },
  { id: 2, label: "Shot Plan" },
  { id: 3, label: "Generate" },
  { id: 4, label: "Final Cut" },
  // { id: 5, label: "完成输出" },
] as const;

/** 只读阶段条：步骤由 agent stage 驱动，视觉对齐 StepHeader。 */
function WorkflowStageCard({ workplace }: { workplace: WorkplaceData | null }) {
  const stage = workplace?.stage ?? null;
  const activeIndex = resolveWorkflowIndex(workplace);
  const currentStep = activeIndex + 1;
  const totalShots = workplace?.shots.length ?? 0;
  const approvedShots =
    workplace?.shots.filter((shot) => shot.status === "approved").length ?? 0;
  const progress =
    totalShots > 0 ? { done: approvedShots, total: totalShots } : null;
  const progressPct =
    progress && progress.total > 0
      ? Math.round((progress.done / progress.total) * 100)
      : null;
  const showShotProgress = activeIndex >= 2 && totalShots > 0;

  return (
    <div
      className="relative shrink-0 h-[100px]"
      aria-label={`Current stage: ${stageLabel(stage)}`}
    >
      <div className="flex h-full items-center gap-1 px-3 py-2.5">
        <Button
          variant="ghost"
          size="icon"
          disabled
          className="invisible h-7 w-7 rounded-lg"
          aria-hidden
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>

        <div className="flex flex-1 items-center justify-center gap-0">
          {WORKFLOW_STEPS.map((step, idx) => {
            const completed = step.id < currentStep;
            const active = step.id === currentStep;
            const reached = step.id <= currentStep;
            return (
              <div key={step.id} className="flex items-center">
                {idx > 0 && (
                  <div className="relative mx-1.5 w-6">
                    <div className="absolute inset-y-1/2 h-px w-full bg-border/50" />
                    <div
                      className={cn(
                        "absolute inset-y-1/2 h-px bg-foreground/20 transition-all duration-500 ease-out",
                        reached ? "w-full" : "w-0",
                      )}
                    />
                  </div>
                )}
                <div
                  className={cn(
                    "group relative flex max-w-[8.5rem] flex-col items-center gap-0.5 rounded-xl px-3 py-2 transition-all duration-300",
                    active &&
                      "bg-foreground/[0.05] ring-1 ring-inset ring-foreground/[0.07]",
                  )}
                >
                  <span
                    className={cn(
                      "text-[13px] tabular-nums transition-all duration-300",
                      active && "font-medium text-foreground/70",
                      completed && "font-medium text-foreground/40",
                      !active &&
                        !completed &&
                        reached &&
                        "font-medium text-foreground/35",
                      !reached &&
                        !active &&
                        !completed &&
                        "font-normal text-muted-foreground/30",
                    )}
                  >
                    {completed ? (
                      <Check className="h-3 w-3" strokeWidth={2.5} />
                    ) : (
                      String(step.id).padStart(2, "0")
                    )}
                  </span>
                  <span
                    className={cn(
                      "text-center text-[13px] transition-all duration-300",
                      active && "font-medium text-foreground/85",
                      completed && "font-normal text-foreground/45",
                      !active &&
                        !completed &&
                        reached &&
                        "font-normal text-foreground/40",
                      !reached &&
                        !active &&
                        !completed &&
                        "font-normal text-muted-foreground/35",
                    )}
                  >
                    {step.label}
                  </span>

                  {active && showShotProgress && progressPct !== null ? (
                    <span className="mt-0.5 text-[13px] tabular-nums text-foreground/40">
                      {progressPct}%
                    </span>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>

        {/* {progress ? (
          <span className="mr-1 shrink-0 text-[15px] tabular-nums text-muted-foreground/70">
            {progress.done}/{progress.total}
          </span>
        ) : (
          <span className="invisible mr-1 shrink-0 text-[15px]" aria-hidden>
            0/0
          </span>
        )} */}

        <Button
          variant="ghost"
          size="icon"
          disabled
          className="invisible h-7 w-7 rounded-lg"
          aria-hidden
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>

      {progress && progress.total > 0 ? (
        <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-border/40">
          <div
            className="h-full bg-foreground/30 transition-all duration-700 ease-out"
            style={{ width: `${(progress.done / progress.total) * 100}%` }}
          />
        </div>
      ) : (
        <div className="h-px w-full bg-border/60" />
      )}
    </div>
  );
}

const waitingCSS = `
@keyframes slate-fade-in {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes status-pulse {
  0%, 100% { opacity: 0.55; }
  50% { opacity: 1; }
}
@keyframes bar-shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(250%); }
}
`;
function StoryWorkspace({
  story,
  storyEmpty,
  readOnly = false,
  onChange,
  onFocus,
  onBlur,
  onApplyNext,
  applyNextDisabled = false,
  applyNextLabel = "Next Step",
  onSave,
  saveDisabled,
  saving = false,
  showThinking = false,
}: {
  story: string;
  storyEmpty: boolean;
  readOnly?: boolean;
  onChange: (text: string) => void;
  onFocus?: () => void;
  onBlur?: () => void;
  onApplyNext: () => void;
  applyNextDisabled?: boolean;
  applyNextLabel?: string;
  onSave?: () => void | Promise<void>;
  saveDisabled?: boolean;
  saving?: boolean;
  showThinking?: boolean;
}) {
  if (!story.trim() && storyEmpty) {
    return (
      <div className="flex h-full flex-col items-center justify-center px-10">
        <style>{waitingCSS}</style>

        {/* Clapperboard card */}
        <div
          className="w-full max-w-[320px] rounded-xl border border-border/70 bg-card/40 backdrop-blur-sm"
          style={{ animation: "slate-fade-in 0.5s ease-out both" }}
        >
          {/* Slate top bar */}
          <div className="flex items-center border-b border-border/50 px-5 py-3">
            <span className="text-[13px] font-semibold uppercase tracking-[0.18em] text-foreground/50">
              Production
            </span>
          </div>

          {/* Fields */}
          <div className="space-y-3 px-5 py-4">
            <div className="flex items-baseline gap-3">
              <span className="w-16 shrink-0 text-[12px] font-medium uppercase tracking-[0.14em] text-foreground/40">
                Scene
              </span>
              <div className="h-px flex-1 bg-border/60" />
            </div>
            <div className="flex items-baseline gap-3">
              <span className="w-16 shrink-0 text-[12px] font-medium uppercase tracking-[0.14em] text-foreground/40">
                Take
              </span>
              <div className="h-px flex-1 bg-border/60" />
            </div>
            <div className="flex items-baseline gap-3">
              <span className="w-16 shrink-0 text-[12px] font-medium uppercase tracking-[0.14em] text-foreground/40">
                Director
              </span>
              <div className="h-px flex-1 bg-border/60" />
            </div>
          </div>

          {/* Status section */}
          {/* 如果 messages 里面有消息了，则展示构思中 */}
          {showThinking ? (
            <div className="border-t border-border/50 px-5 py-4">
              <div className="flex items-center justify-center">
                <span
                  className="text-[13px] font-medium text-foreground/70"
                  style={{
                    animation: "status-pulse 2.4s ease-in-out infinite",
                  }}
                >
                  Developing story...
                </span>
              </div>

              {/* Progress bar */}
              <div className="relative mt-3 h-[3px] overflow-hidden rounded-full bg-border/50">
                <div
                  className="absolute inset-y-0 left-0 w-[28%] rounded-full bg-foreground/20"
                  style={{ animation: "bar-shimmer 2.2s ease-in-out infinite" }}
                />
              </div>
            </div>
          ) : null}
        </div>

        {/* Subtitle text */}
        <p
          className="mt-6 text-center text-[14px] leading-relaxed text-muted-foreground/60"
          style={{ animation: "slate-fade-in 0.5s ease-out 0.2s both" }}
        >
          Describe your idea in the conversation. The script will appear here.
        </p>
      </div>
    );
  }
  return (
    <ScriptEditor
      value={story}
      onChange={onChange}
      onFocus={onFocus}
      onBlur={onBlur}
      readOnly={readOnly}
      onApplyNext={onApplyNext}
      applyNextDisabled={applyNextDisabled}
      applyNextLabel={applyNextLabel}
      onSave={onSave}
      saveDisabled={saveDisabled}
      saving={saving}
    />
  );
}

// function ShotCard({
//   shot,
//   busy,
//   onAccept,
//   onRevise,
// }: {
//   shot: WorkplaceShot;
//   busy: boolean;
//   onAccept: () => Promise<void>;
//   onRevise: (feedback: string) => Promise<void>;
// }) {
//   const [editing, setEditing] = useState(false);
//   const [feedback, setFeedback] = useState("");
//   const [localError, setLocalError] = useState<string | null>(null);
//   const [revisionSubmitted, setRevisionSubmitted] = useState(false);
//   const canReviewShot = ["generated", "review_pass"].includes(shot.status);
//   const serverRevisionPending = ["review_fail", "queued"].includes(shot.status);
//   const revisionLocked = revisionSubmitted || serverRevisionPending;
//   const showActions = shot.has_actions && canReviewShot && !revisionLocked;

//   useEffect(() => {
//     if (revisionSubmitted && !canReviewShot) {
//       setRevisionSubmitted(false);
//     }
//   }, [canReviewShot, revisionSubmitted]);

//   useEffect(() => {
//     if (revisionLocked) {
//       setEditing(false);
//     }
//   }, [revisionLocked]);

//   const timelineLabel = useMemo(
//     () =>
//       `${shot.timeline.label} · ${formatDuration(shot.timeline.duration_seconds)}`,
//     [shot.timeline.duration_seconds, shot.timeline.label],
//   );

//   return (
//     <div className="group relative pl-10">
//       <div className="absolute left-[0.8rem] top-4 h-full w-px bg-gradient-to-b from-border via-border/70 to-transparent" />
//       <div className="absolute left-0 top-3 flex h-7 w-7 items-center justify-center rounded-full border border-border/70 bg-background shadow-sm">
//         <span className="text-[11px] font-semibold text-foreground/82">
//           {shot.shot_id}
//         </span>
//       </div>

//       <article
//         className={cn(
//           "animate-in fade-in-0 slide-in-from-top-2 rounded-[1.6rem] border border-border/70 bg-background/92 p-4 shadow-[0_22px_65px_-48px_rgba(15,23,42,0.7)] backdrop-blur",
//           "transition-all duration-300 hover:-translate-y-0.5 hover:shadow-[0_30px_80px_-46px_rgba(15,23,42,0.82)]",
//         )}
//       >
//         <div className="mb-3 flex items-start justify-between gap-3">
//           <div className="min-w-0">
//             <p className="text-xs uppercase tracking-[0.24em] text-muted-foreground">
//               Shot {shot.shot_id}
//             </p>
//             <p className="mt-1 text-sm leading-6 text-foreground/92">
//               {shot.summary || "等待 agent 补充分镜说明"}
//             </p>
//           </div>
//           <span className="rounded-full border border-border/70 bg-muted/45 px-2.5 py-1 text-[11px] font-medium text-muted-foreground">
//             {timelineLabel}
//           </span>
//         </div>

//         {shot.video?.url ? (
//           <div className="overflow-hidden rounded-[1.2rem] border border-border/70 bg-black/90">
//             <video
//               src={shot.video.url}
//               controls
//               preload="metadata"
//               className="aspect-video w-full bg-black object-cover"
//             />
//           </div>
//         ) : isShotWaitingForVideo(shot) ? (
//           <ShotLoadingPreview />
//         ) : null}

//         {showActions ? (
//           <div className="mt-4 rounded-[1.2rem] border border-border/70 bg-muted/20 p-3">
//             <div className="flex flex-wrap items-center gap-2">
//               <Button
//                 size="sm"
//                 onClick={() => void onAccept()}
//                 disabled={busy}
//                 className="rounded-full"
//               >
//                 <Check className="mr-1.5 h-4 w-4" />
//                 接受
//               </Button>
//               <Button
//                 size="sm"
//                 variant="outline"
//                 onClick={() => {
//                   setEditing((value) => !value);
//                   setLocalError(null);
//                 }}
//                 disabled={busy}
//                 className="rounded-full"
//               >
//                 修改
//               </Button>
//             </div>

//             {editing ? (
//               <div className="mt-3 space-y-2">
//                 <Textarea
//                   value={feedback}
//                   onChange={(event) => setFeedback(event.target.value)}
//                   placeholder="填写需要修改的镜头节奏、画面、人物动作或台词意见"
//                   className="min-h-[88px] resize-y rounded-2xl"
//                 />
//                 {localError ? (
//                   <p className="text-xs text-destructive">{localError}</p>
//                 ) : null}
//                 <div className="flex items-center gap-2">
//                   <Button
//                     size="sm"
//                     variant="outline"
//                     disabled={busy}
//                     onClick={() => {
//                       setEditing(false);
//                       setFeedback("");
//                       setLocalError(null);
//                     }}
//                     className="rounded-full"
//                   >
//                     取消
//                   </Button>
//                   <Button
//                     size="sm"
//                     disabled={busy}
//                     onClick={() => {
//                       const trimmed = feedback.trim();
//                       if (!trimmed) {
//                         setLocalError("请先填写修改意见");
//                         return;
//                       }
//                       setLocalError(null);
//                       setRevisionSubmitted(true);
//                       setEditing(false);
//                       setFeedback("");
//                       void onRevise(trimmed).catch(() => {
//                         setRevisionSubmitted(false);
//                       });
//                     }}
//                     className="rounded-full"
//                   >
//                     提交修改
//                   </Button>
//                 </div>
//               </div>
//             ) : null}
//           </div>
//         ) : null}

//         {shot.status === "approved" ? (
//           <div className="mt-4 rounded-[1.2rem] border border-emerald-500/20 bg-emerald-500/8 px-3 py-2 text-sm text-emerald-800 dark:text-emerald-200">
//             这个 shot 已经被接受，后续会参与最终合成。
//           </div>
//         ) : null}

//         {shot.review_notes ? (
//           <div className="mt-3 rounded-2xl bg-muted/35 px-3 py-2 text-sm text-muted-foreground">
//             {shot.review_notes}
//           </div>
//         ) : null}

//         <div className="mt-4 flex items-end justify-between gap-3">
//           <span
//             className={cn(
//               "inline-flex rounded-full px-2.5 py-1 text-[11px] font-medium ring-1 ring-inset",
//               statusTone(shot.status),
//             )}
//           >
//             {statusLabel(shot.status)}
//           </span>
//           <span className="text-[11px] text-muted-foreground">
//             {shot.cut ? "新镜头切点" : "承接上一镜"}
//           </span>
//         </div>
//       </article>
//     </div>
//   );
// }

// function ShotsWorkspace({
//   shots,
//   finalVideo,
//   generationStarted,
//   busyShotId,
//   onAccept,
//   onRevise,
// }: {
//   shots: WorkplaceShot[];
//   finalVideo?: WireMediaRef | null;
//   generationStarted: boolean;
//   busyShotId: number | null;
//   onAccept: (shotId: number) => Promise<void>;
//   onRevise: (shotId: number, feedback: string) => Promise<void>;
// }) {
//   if (shots.length === 0) {
//     if (!generationStarted) return null;
//     return <ShotLoadingPreview />;
//   }

//   return (
//     <div className="animate-in fade-in-0 slide-in-from-top-3 space-y-4 pb-6 duration-500">
//       {shots.map((shot) => (
//         <ShotCard
//           key={shot.shot_key}
//           shot={shot}
//           busy={busyShotId === shot.shot_id}
//           onAccept={() => onAccept(shot.shot_id)}
//           onRevise={(feedback) => onRevise(shot.shot_id, feedback)}
//         />
//       ))}
//       {finalVideo?.url ? <FinalVideoCard video={finalVideo} /> : null}
//     </div>
//   );
// }

export function WorkplacePanel({
  sessionKey,
  activeTab: _activeTab,
  onTabChange: _onTabChange,
  onNewChat,
  showThinking = false,
  chatBusy = false,
}: WorkplacePanelProps) {
  const {
    workplace,
    loading,
    error,
    workflowBusy,
    splitMergeBusy,
    mutatingShotId,
    accept,
    acceptAll,
    acceptAllBusy,
    revise,
    generateOneShot,
    continuousGenerateOneShot,
    setContinuousMode,
    generateAll,
    promptOverrides,
    updateShotDuration,
    regenerate,
    abortGeneration,
    startAutoGenerate,
    editor,
    setAuxTextEditing,
    updateEchoLike,
    recordEchoDownloadPrompt,
    memoryWorkspaceBusy,
    saveMemoryAsset,
    createShotMemoryAsset,
    deleteMemoryAsset,
    saveShotMemorySlots,
  } = useWorkplaceContext();

  const [showOverlay, setShowOverlay] = useState(false);
  // 不可逆工作流动作统一经 StepConfirmDialog 门控，避免误触直接 commit
  const [pendingConfirm, setPendingConfirm] =
    useState<IrreversibleAction | null>(null);
  const [shotCountAlertOpen, setShotCountAlertOpen] = useState(false);

  const [shotEditAlertOpen, setShotEditAlertOpen] = useState(false);
  const [generationCongestedOpen, setGenerationCongestedOpen] = useState(false);
  const generationCongestedShownRef = useRef(false);
  const shotMemorySlotsRefs = useRef(
    new Map<number, ShotMemorySlotsHandle>(),
  );
  const generationAlerts = useShotGenerationAlerts(workplace, sessionKey);

  const shot1VideoSize = useMemo(() => {
    const width = Number(workplace?.goal?.width);
    const height = Number(workplace?.goal?.height);
    if (
      Number.isFinite(width) &&
      Number.isFinite(height) &&
      width > 0 &&
      height > 0
    ) {
      return { width, height };
    }
    return DEFAULT_VIDEO_SIZE;
  }, [workplace?.goal?.height, workplace?.goal?.width]);

  const step = resolveWorkflowIndex(workplace);
  const shots = workplace?.shots ?? [];
  const memoryWorkspaceAssets = useMemo(
    () =>
      workplace?.memory_workspace_assets ??
      collectConfirmedMemoryEntries(workplace).map((entry, index) => ({
        asset_id: `legacy-${entry.memory_id}-${entry.source_shot_id}-${entry.frame_index}-${index}`,
        display_name: entry.display_name ?? entry.memory_id,
        source: "automatic" as const,
        kind: entry.kind,
        memory_id: entry.memory_id,
        source_shot_id: entry.source_shot_id,
        frame_index: entry.frame_index,
        image: entry.image,
        audio: entry.audio,
      })),
    [workplace],
  );

  const frames = useMemo(
    () =>
      workplace?.shots.map((shot) => {
        const memoryRecommendationReady = workplace
          ? isMemoryRecommendationReady(workplace, shot)
          : false;
        const memoryNeedsApply =
          shot.shot_id > 1 &&
          workplace?.stage === "awaiting_memory_build" &&
          memoryRecommendationReady &&
          shot.memory_slots_configured !== true;
        const referenceShotIds = shotReferenceIds(shot);
        const missing = workplace
          ? missingReferenceGenerations(workplace, shot)
          : [];
        const approvalMessage = workplace
          ? previousShotApprovalMessage(workplace, shot)
          : null;
        const shotOneHint =
          shot.shot_id === 1 && workplace
            ? shotOneWorkflowHint(workplace)
            : null;
        return {
          id: shot.shot_key,
          shotId: shot.shot_id,
          cut: shot.cut,
          caption: shot.caption ?? "",
          numFrames: shot.num_frames ?? undefined,
          segmentText: editor.get(`beat:${shot.shot_id}`) ?? shot.summary ?? "",
          prompt: promptOverrides[shot.shot_id] ?? shot.summary ?? "",
          status: shot.status as FrameStatus,
          videoUrl: shot.video?.url ?? "",
          error: shot.generation_error || undefined,
          durationSec: shot.timeline.duration_seconds ?? 5,
          referenceShotIds,
          referenceNote: shot.reference_selection_note ?? "",
          canGenerate: workplace ? canGenerateShot(workplace, shot) : false,
          dependencyMessage:
            approvalMessage ??
            (!memoryRecommendationReady
              ? "Memory will unlock after the Agent finishes its recommendation."
              : memoryNeedsApply
                ? "Review and apply the recommended Memory before generating."
                : null) ??
            (missing.length > 0
              ? referenceDependencyMessage(shot, missing)
              : undefined),
          hintMessage: shotOneHint ?? undefined,
          hasActions: shot.has_actions,
          reviewNotes: shot.review_notes || undefined,
          accepted: shot.accepted,
          generationMemories: shot.generation_memories ?? [],
          continuousEnabled: shot.continuous_enabled ?? false,
        };
      }) ?? [],
    [editor, promptOverrides, workplace],
  );

  const shotGenerationBusy = loading || workflowBusy || mutatingShotId !== null;

  const busyFrameId = useMemo(() => {
    if (mutatingShotId === null) return null;
    return (
      shots.find((shot) => shot.shot_id === mutatingShotId)?.shot_key ?? null
    );
  }, [mutatingShotId, shots]);

  const handleConfirmStory = useCallback(() => {
    if (editor.storyDirty) {
      setShotEditAlertOpen(true);
      return;
    }
    if (!workplace?.goal?.shot_count) {
      setShotCountAlertOpen(true);
      return;
    }
    setPendingConfirm("confirm_story");
  }, [editor.storyDirty, workplace?.goal?.shot_count]);

  /** 分镜脚本阶段「进入逐镜打磨」→ start-generation。 */
  const handleStoryboardNext = useCallback(() => {
    if (workplace && hasUnsavedBeatChanges(workplace, editor)) {
      setShotEditAlertOpen(true);
      return;
    }
    setPendingConfirm("start_generation");
  }, [workplace, editor.segments]);

  /** 分镜脚本阶段「确认并一键成片」→ workflow/auto-generate。 */
  const handleStoryboardAutoGenerate = useCallback(() => {
    if (workplace && hasUnsavedBeatChanges(workplace, editor)) {
      setShotEditAlertOpen(true);
      return;
    }
    setPendingConfirm("auto_generate");
  }, [workplace, editor.segments]);

  /** FramesPanel「全部生成」→ generate-all（同步提交 Echo 任务）。 */
  const handleGenerateAll = useCallback(() => {
    void generateAll();
  }, [generateAll]);

  /** FramesPanel「全部接收」→ accept-all（批量确认分镜）。 */
  const handleAcceptAll = useCallback(() => {
    void acceptAll();
  }, [acceptAll]);

  const handleStartMerge = useCallback(() => {
    setPendingConfirm("start_merge");
  }, []);

  const handleStepConfirm = useCallback(() => {
    if (!pendingConfirm) return;
    const action = pendingConfirm;
    setPendingConfirm(null);
    switch (action) {
      case "confirm_story":
        void editor.commit({ type: "confirm_story" });
        break;
      case "start_generation":
        void editor.commit({ type: "start_generation" });
        break;
      case "auto_generate":
        void startAutoGenerate();
        break;
      case "generate_all":
        void generateAll();
        break;
      case "start_merge":
        void editor.commit({ type: "start_merge" });
        setShowOverlay(true);
        break;
    }
  }, [
    editor,
    generateAll,
    pendingConfirm,
    startAutoGenerate,
  ]);

  const handleStepCancel = useCallback(() => {
    setPendingConfirm(null);
  }, []);

  /** FramesPanel「下一步」：触发合成并展示全屏 RenderOverlay。 */
  const handleCompose = useCallback(() => {
    handleStartMerge();
  }, [handleStartMerge]);

  const handleDismissOverlay = useCallback(() => {
    setShowOverlay(false);
  }, []);

  useEffect(() => {
    setShowOverlay(false);
    setPendingConfirm(null);
    setShotCountAlertOpen(false);
    setShotEditAlertOpen(false);
    setGenerationCongestedOpen(false);
    generationCongestedShownRef.current = false;
  }, [sessionKey]);

  /**
   * start-generation 后 agent 写 caption / 设 references 若超过 5 分钟仍未就绪，
   * 提示服务器拥挤并允许回退到 shot_planning 重试。已进入可生成态则不计时。
   */
  useEffect(() => {
    if (
      !GENERATION_BUSY_ALERT_ENABLED ||
      workplace?.stage !== "shot_generating" ||
      workplace.references_ready
    ) {
      if (workplace?.stage !== "shot_generating") {
        generationCongestedShownRef.current = false;
      }
      return;
    }
    if (generationCongestedShownRef.current) return;

    const startedRaw = workplace.shot_generating_started_at;
    const startedMs = startedRaw ? Date.parse(startedRaw) : Number.NaN;
    const anchor = Number.isFinite(startedMs) ? startedMs : Date.now();
    const remaining = Math.max(
      0,
      SHOT_GENERATING_PREP_TIMEOUT_MS - (Date.now() - anchor),
    );

    const timer = window.setTimeout(() => {
      generationCongestedShownRef.current = true;
      setGenerationCongestedOpen(true);
    }, remaining);

    return () => window.clearTimeout(timer);
  }, [
    workplace?.stage,
    workplace?.references_ready,
    workplace?.shot_generating_started_at,
    sessionKey,
  ]);

  const handleGenerationCongestedRetry = useCallback(() => {
    setGenerationCongestedOpen(false);
    void abortGeneration().catch(() => {
      // error 已写入 workplace.error；保持弹窗关闭避免重复打扰
    });
  }, [abortGeneration]);

  const mergeRenderStatus = useMemo((): RenderStatus => {
    const status = composeRenderStatus(workplace);
    if (status !== "idle") return status;
    if (showOverlay) return "rendering";
    return "idle";
  }, [showOverlay, workplace]);

  /** 成片后回到分镜编辑：清空 final_video，stage → shot_planning */
  const handleRegenerate = useCallback(() => {
    void regenerate().then(() => setShowOverlay(false));
  }, [regenerate]);

  /** RenderOverlay onRetry：成片完成走 regenerate，合成失败仍 startMerge */
  const handleMergeOverlayRetry = useCallback(() => {
    if (mergeRenderStatus === "done") {
      handleRegenerate();
      return;
    }
    handleStartMerge();
  }, [mergeRenderStatus, handleRegenerate, handleStartMerge]);

  /** ComposePanel onRetry：与 RenderOverlay 分支逻辑一致 */
  const handleComposeRetry = useCallback(() => {
    if (workplace?.stage === "done") {
      handleRegenerate();
      return;
    }
    const failed = failedAutoGenerateShot(workplace);
    if (workplace?.auto_generate && failed) {
      void generateOneShot(failed.shot_id);
      return;
    }
    handleStartMerge();
  }, [
    generateOneShot,
    handleRegenerate,
    handleStartMerge,
    workplace,
  ]);

  const handleSplitAt = useCallback(
    (segmentId: string, cursorPos: number, segmentText: string) => {
      const segment = editor.segments.find((item) => item.id === segmentId);
      if (!segment) return;
      const before = segmentText.slice(0, cursorPos).trimEnd();
      const after = segmentText.slice(cursorPos).trimStart();
      if (!before || !after) return;
      void editor.commit({
        type: "split_shot",
        shotId: segment.shotId,
        beforeText: before,
        afterText: after,
      });
    },
    [editor],
  );

  const handleMergeUp = useCallback(
    (lowerSegmentId: string, mergedText: string) => {
      const segment = editor.segments.find(
        (item) => item.id === lowerSegmentId,
      );
      if (!segment) return;
      void editor.commit({
        type: "merge_shot",
        shotId: segment.shotId,
        mergedText,
      });
    },
    [editor],
  );

  const handleDeleteSegment = useCallback(
    (segmentId: string) => {
      const segment = editor.segments.find((item) => item.id === segmentId);
      if (!segment) return;
      if (
        editor.beatsReadOnly ||
        splitMergeBusy ||
        editor.segments.length <= 1
      ) {
        return;
      }
      void editor.commit({ type: "delete_shot", shotId: segment.shotId });
    },
    [editor, splitMergeBusy],
  );

  const handleRetryFrame = useCallback(
    (frameId: string) => {
      const shotId = shotIdFromKey(shots, frameId);
      if (shotId === null) return;
      void revise(shotId, "Please regenerate.");
    },
    [revise, shots],
  );

  const handleAcceptFrame = useCallback(
    (frameId: string) => {
      const shotId = shotIdFromKey(shots, frameId);
      if (shotId === null) return;
      void accept(shotId);
    },
    [accept, shots],
  );

  const handleReviseFrame = useCallback(
    async (frameId: string, feedback: string) => {
      const shotId = shotIdFromKey(shots, frameId);
      if (shotId === null) return;
      await shotMemorySlotsRefs.current.get(shotId)?.applyPending();
      await revise(shotId, feedback);
    },
    [revise, shots],
  );

  const handleGenerateShot = useCallback(
    (frameId: string) => {
      if (!workplace) return;
      const shot = workplace.shots.find((item) => item.shot_key === frameId);
      if (!shot) return;
      const missing = missingReferenceGenerations(workplace, shot);
      if (missing.length > 0) return;

      if (shot.continuous_enabled && shot.shot_id > 1) {
        void continuousGenerateOneShot(shot.shot_id);
        return;
      }

      void generateOneShot(shot.shot_id);
    },
    [
      continuousGenerateOneShot,
      generateOneShot,
      workplace,
    ],
  );

  const shot1FirstFrameControls = useMemo<Shot1FirstFrameControls | null>(() => {
    const displayUrl = workplace?.reference_image?.url ?? null;
    if (!displayUrl) return null;
    return {
      displayUrl,
      videoSize: shot1VideoSize,
    };
  }, [shot1VideoSize, workplace?.reference_image?.url]);

  const handleSetContinuousMode = useCallback(
    (frameId: string, enabled: boolean) => {
      if (!workplace) return;
      const shot = workplace.shots.find((item) => item.shot_key === frameId);
      if (!shot) return;
      void setContinuousMode(shot.shot_id, enabled);
    },
    [setContinuousMode, workplace],
  );

  const handleUpdatePrompt = useCallback(
    (frameId: string, prompt: string) => {
      if (!workplace) return;
      const shotId = shotIdFromKey(shots, frameId);
      if (shotId === null) return;
      const trimmed = prompt.trim();
      if (!trimmed) return;
      const shot = workplace.shots.find((item) => item.shot_id === shotId);
      const oldSummary =
        beatSummaryForShot(workplace, shotId) || shot?.summary?.trim() || "";
      if (trimmed === oldSummary) return;
      const feedback = buildBeatRevisionFeedback(shotId, oldSummary, trimmed);
      void revise(shotId, feedback);
    },
    [revise, shots, workplace],
  );

  const handleUpdateDuration = useCallback(
    (frameId: string, durationSec: number) => {
      const shotId = shotIdFromKey(shots, frameId);
      if (shotId === null) return;
      const shot = shots.find((item) => item.shot_id === shotId);
      const current = shot?.timeline.duration_seconds;
      if (current === durationSec) return;
      void updateShotDuration(shotId, durationSec);
    },
    [shots, updateShotDuration],
  );

  const storyValue = editor.get("story");
  const storyBusy = workflowBusy || editor.committing;
  const generationBusy = storyBusy || splitMergeBusy || workflowBusy || chatBusy;

  // confirm-story 后 stage 可能停在 story_confirmed，分镜尚未落盘
  const storyWaitingForShots =
    Boolean(workplace?.story_confirmed) && shots.length === 0;

  const storyApplyLabel = useMemo(() => {
    // if (storyWaitingForShots) {
    //   const progress = workplace?.shot_prompts_progress;
    //   if (progress && progress.total > 0 && progress.ready < progress.total) {
    //     return `生成分镜脚本中 (${progress.ready}/${progress.total})...`;
    //   }
    //   return "生成分镜脚本中...";
    // }
    if (workflowBusy) return "Approving story...";
    return "Next Step";
  }, [storyWaitingForShots, workflowBusy, workplace?.shot_prompts_progress]);

  const beatsApplyLabel = useMemo(() => {
    if (!workflowBusy || workplace?.shot_prompts_ready) return "Next Step";
    const progress = workplace?.shot_prompts_progress;
    if (progress && progress.total > 0 && progress.ready < progress.total) {
      return `Preparing shot prompts (${progress.ready}/${progress.total})...`;
    }
    return "Preparing shot prompts...";
  }, [
    workflowBusy,
    workplace?.shot_prompts_progress,
    workplace?.shot_prompts_ready,
  ]);

  const workflowContent = useMemo(() => {
    if (!sessionKey || !workplace) return null;

    switch (step) {
      case 0:
        return (
          <StoryWorkspace
            story={storyValue}
            storyEmpty={workplace.story_empty}
            showThinking={showThinking}
            readOnly={editor.storyReadOnly}
            onChange={(text) => editor.edit("story", text)}
            onFocus={() => editor.hold("story")}
            onBlur={() => editor.release("story")}
            onSave={() => void editor.saveStory()}
            saveDisabled={
              editor.storyReadOnly || !editor.storyDirty || editor.committing
            }
            saving={editor.committing}
            onApplyNext={handleConfirmStory}
            applyNextLabel={storyApplyLabel}
            applyNextDisabled={
              storyBusy ||
              // storyWaitingForShots ||
              !storyValue.trim()
            }
          />
        );
      case 1:
        return (
          <StoryboardScriptEditor
            segments={editor.segments.map(({ id, text }) => ({ id, text }))}
            readOnly={editor.beatsReadOnly}
            onSegmentChange={(segmentId, text) => {
              const segment = editor.segments.find(
                (item) => item.id === segmentId,
              );
              if (!segment) return;
              editor.edit(`beat:${segment.shotId}`, text);
            }}
            onDeleteSegment={handleDeleteSegment}
            onSegmentFocus={(segmentId) => {
              const segment = editor.segments.find(
                (item) => item.id === segmentId,
              );
              if (!segment) return;
              editor.hold(`beat:${segment.shotId}`);
            }}
            onSegmentBlur={(segmentId) => {
              const segment = editor.segments.find(
                (item) => item.id === segmentId,
              );
              if (!segment) return;
              editor.release(`beat:${segment.shotId}`);
            }}
            onSave={() => void editor.saveBeats()}
            saveDisabled={
              editor.beatsReadOnly || !editor.beatsDirty || editor.committing
            }
            saving={editor.committing}
            onApplyNext={handleStoryboardNext}
            applyNextLabel={
              workflowBusy && !workplace?.shot_prompts_ready
                ? beatsApplyLabel
                : "Enter Shot Workshop"
            }
            onAutoGenerate={handleStoryboardAutoGenerate}
            autoGenerateLabel="Approve & Auto Generate"
            onSplitAt={handleSplitAt}
            onMergeUp={handleMergeUp}
            applyNextDisabled={
              generationBusy || editor.segments.length === 0
            }
            splitMergeDisabled={
              splitMergeBusy || storyBusy || editor.beatsReadOnly
            }
          />
        );
      case 2:
        return (
          <FramesPanel
            frames={frames}
            memoryBank={
              !workplace.auto_generate ? (
                <MemoryBankBoard
                  assets={memoryWorkspaceAssets}
                  shots={shots}
                  busy={memoryWorkspaceBusy}
                  showSlots={false}
                  onSaveAsset={saveMemoryAsset}
                  onDeleteAsset={deleteMemoryAsset}
                  onApplySlots={saveShotMemorySlots}
                />
              ) : null
            }
            renderMemorySlots={
              !workplace.auto_generate
                ? (frameId) => {
                    const shotIndex = shots.findIndex(
                      (candidate) => candidate.shot_key === frameId,
                    );
                    const shot = shots[shotIndex];
                    const recommendationReady = shot
                      ? isMemoryRecommendationReady(workplace, shot)
                      : false;
                    const previousShot = shotIndex > 0
                      ? shots[shotIndex - 1]
                      : null;
                    const conditionImage = shot?.shot_id === 1
                      ? workplace.reference_image ?? null
                      : shot?.continuous_enabled && previousShot?.tail_frame_url
                        ? {
                            url: previousShot.tail_frame_url,
                            name: `Shot ${previousShot.shot_id} tail frame`,
                          }
                        : null;
                    if (shot && !recommendationReady) {
                      return (
                        <section
                          aria-label={`Memory recommendation pending for Shot ${shot.shot_id}`}
                          className="flex items-center gap-2 rounded-xl border border-border/60 bg-muted/20 px-3 py-2 text-xs text-muted-foreground"
                        >
                          <LoaderCircle className="size-3.5 animate-spin" aria-hidden />
                          <span>Memory is folded while the Agent prepares its recommendation.</span>
                        </section>
                      );
                    }
                    return shot ? (
                      <ShotMemorySlots
                        ref={(editor) => {
                          if (editor) {
                            shotMemorySlotsRefs.current.set(shot.shot_id, editor);
                          } else {
                            shotMemorySlotsRefs.current.delete(shot.shot_id);
                          }
                        }}
                        assets={memoryWorkspaceAssets}
                        shot={shot}
                        conditionImage={conditionImage}
                        busy={memoryWorkspaceBusy}
                        onApplySlots={saveShotMemorySlots}
                        onCreateAsset={createShotMemoryAsset}
                      />
                    ) : null;
                  }
                : undefined
            }
            // framesSummary={framesSummary}
            batchGenerating={shotGenerationBusy}
            composeDisabled={shotGenerationBusy}
            referencesReady={workplace.references_ready === true}
            onGenerate={handleGenerateShot}
            onGenerateAll={handleGenerateAll}
            onUpdatePrompt={handleUpdatePrompt}
            onUpdateDuration={handleUpdateDuration}
            onRetry={handleRetryFrame}
            onAccept={handleAcceptFrame}
            onAcceptAll={handleAcceptAll}
            acceptAllBusy={acceptAllBusy}
            onRevise={handleReviseFrame}
            onSetContinuousMode={handleSetContinuousMode}
            busyFrameId={busyFrameId}
            onCompose={handleCompose}
            onPromptEditingChange={setAuxTextEditing}
            shot1FirstFrame={shot1FirstFrameControls}
          />
        );
      // case 3:
      //   return (
      //     <RenderOverlay
      //       status="rendering"
      //       progress={0}
      //       onRetry={handleStartMerge}
      //       onDismiss={() => {}}
      //       hideBack={false}
      //     />
      //   );
      case 3:
      case 4:
        const renderStatus = composeRenderStatus(workplace);
        return (
          <>
            <ComposePanel
              renderStatus={renderStatus}
              composingTitle={
                workplace?.auto_generate && renderStatus === "rendering"
                  ? "Automatic production in progress"
                  : undefined
              }
              composingHint={
                workplace?.auto_generate && renderStatus === "rendering"
                  ? "Generating and assembling all shots automatically"
                  : undefined
              }
              error={composeRenderError(workplace)}
              videoUrl={workplace?.final_video?.url}
              sessionKey={sessionKey ?? undefined}
              downloadFileName={workplace?.final_video?.name}
              onRetry={handleComposeRetry}
              onNewChat={
                onNewChat
                  ? () => {
                      void onNewChat();
                    }
                  : undefined
              }
              playbackInOverlay={showOverlay && renderStatus === "done"}
              storyboardSegments={editor.segments.map(({ id, text }) => ({
                id,
                text,
              }))}
              echoRequestId={workplace?.echo_request_id}
              likeStatus={workplace?.like_status}
              onEchoLike={updateEchoLike}
              onEchoDownloadPrompt={recordEchoDownloadPrompt}
            />
          </>
        );
      default:
        return null;
    }
  }, [
    beatsApplyLabel,
    busyFrameId,
    editor,
    frames,
    memoryWorkspaceAssets,
    memoryWorkspaceBusy,
    generationBusy,
    handleAcceptFrame,
    handleConfirmStory,
    handleDeleteSegment,
    handleGenerateShot,
    handleSetContinuousMode,
    handleMergeUp,
    handleRetryFrame,
    handleReviseFrame,
    handleSplitAt,
    handleStoryboardNext,
    handleStoryboardAutoGenerate,
    handleGenerateAll,
    handleAcceptAll,
    handleCompose,
    handleComposeRetry,
    handleUpdatePrompt,
    handleUpdateDuration,
    onNewChat,
    showOverlay,
    acceptAllBusy,
    sessionKey,
    saveMemoryAsset,
    createShotMemoryAsset,
    deleteMemoryAsset,
    saveShotMemorySlots,
    shotGenerationBusy,
    shot1FirstFrameControls,
    showThinking,
    splitMergeBusy,
    step,
    storyApplyLabel,
    storyBusy,
    storyValue,
    storyWaitingForShots,
    workplace,
    updateEchoLike,
    recordEchoDownloadPrompt,
  ]);
  return (
    <section className="flex h-full min-h-0 flex-col bg-background">
      {sessionKey ? <WorkflowStageCard workplace={workplace} /> : null}

      <div className="min-h-0 flex-1 overflow-y-auto scrollbar-thin [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-muted-foreground/30 [&::-webkit-scrollbar-track]:bg-transparent">
        {!sessionKey ? (
          <div className="flex h-full min-h-[18rem] flex-col items-center justify-center rounded-3xl border border-dashed border-border/80 bg-muted/35 px-6 text-center">
            <p className="text-base font-medium text-foreground/88">
              Select a project
            </p>
            <p className="mt-2 text-sm leading-6 text-muted-foreground">
              The production workspace follows the active conversation.
            </p>
          </div>
        ) : null}
        {workflowContent}

        {showOverlay && mergeRenderStatus !== "idle" ? (
          <RenderOverlay
            status={mergeRenderStatus}
            videoUrl={workplace?.final_video?.url}
            sessionKey={sessionKey ?? undefined}
            downloadFileName={workplace?.final_video?.name}
            onRetry={handleMergeOverlayRetry}
            onDismiss={handleDismissOverlay}
            onNewChat={
              onNewChat
                ? () => {
                    void onNewChat();
                  }
                : undefined
            }
            hideBack={false}
            echoRequestId={workplace?.echo_request_id}
            likeStatus={workplace?.like_status}
            onEchoLike={updateEchoLike}
          />
        ) : null}

        {error ? (
          <p className="mt-4 text-xs text-destructive">{error}</p>
        ) : null}
      </div>

      <StepConfirmDialog
        open={pendingConfirm !== null}
        to={pendingConfirm ? ACTION_TO_STEP[pendingConfirm] : 2}
        onConfirm={handleStepConfirm}
        onCancel={handleStepCancel}
      />
      <StepConfirmDialog
        open={shotCountAlertOpen}
        alert={{
          title: "Shot count required",
          desc: "Confirm the number of shots before continuing.",
        }}
        onConfirm={() => setShotCountAlertOpen(false)}
        onCancel={() => setShotCountAlertOpen(false)}
      />
      <StepConfirmDialog
        open={shotEditAlertOpen}
        alert={{
          title: "Unsaved changes",
          desc: "Save the current edit before continuing.",
        }}
        onConfirm={() => setShotEditAlertOpen(false)}
        onCancel={() => setShotEditAlertOpen(false)}
      />
      <StepConfirmDialog
        open={GENERATION_BUSY_ALERT_ENABLED && generationCongestedOpen}
        alert={{
          title: "Generation service busy",
          desc: "The generation service is busy. Please retry.",
          actionLabel: "Retry",
        }}
        onConfirm={handleGenerationCongestedRetry}
        onCancel={handleGenerationCongestedRetry}
      />
      <GenerationAlertDialog
        open={
          generationAlerts.activeVariant === "error" ||
          (GENERATION_BUSY_ALERT_ENABLED &&
            generationAlerts.activeVariant === "congested")
        }
        variant={generationAlerts.activeVariant ?? "error"}
        onDismiss={generationAlerts.dismissActive}
      />
    </section>
  );
}
