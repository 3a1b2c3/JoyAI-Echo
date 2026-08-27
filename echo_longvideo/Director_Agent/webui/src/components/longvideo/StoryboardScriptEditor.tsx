import { useCallback, useRef } from "react";
import {
  ArrowRight,
  Download,
  Save,
  GripVertical,
  Layout,
  Merge,
  Scissors,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface StoryboardSegment {
  id: string;
  text: string;
}

export function downloadStoryboardSegments(segments: StoryboardSegment[]) {
  const content = segments
    .map((seg, i) => `[SHOT ${String(i + 1).padStart(2, "0")}]\n${seg.text}`)
    .join("\n\n");
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "shot-plan.txt";
  a.click();
  URL.revokeObjectURL(url);
}

interface StoryboardScriptEditorProps {
  segments: StoryboardSegment[];
  onSegmentChange: (segmentId: string, text: string) => void;
  onDeleteSegment: (segmentId: string) => void;
  onApplyNext: () => void;
  onAutoGenerate?: () => void;
  autoGenerateLabel?: string;
  onSplitAt?: (
    segmentId: string,
    cursorPos: number,
    segmentText: string,
  ) => void;
  onMergeUp?: (lowerSegmentId: string, mergedText: string) => void;
  onSegmentFocus?: (segmentId: string) => void;
  onSegmentBlur?: (segmentId: string) => void;
  applyNextDisabled?: boolean;
  applyNextLabel?: string;
  splitMergeDisabled?: boolean;
  readOnly?: boolean;
  onSave?: () => void | Promise<void>;
  saveDisabled?: boolean;
  saving?: boolean;
}

export function StoryboardScriptEditor({
  segments,
  onSegmentChange,
  onDeleteSegment,
  onApplyNext,
  onSplitAt,
  onMergeUp,
  onSegmentFocus,
  onSegmentBlur,
  applyNextDisabled = false,
  applyNextLabel = "Next Step",
  onAutoGenerate,
  autoGenerateLabel = "Approve & Auto Generate",
  splitMergeDisabled = false,
  readOnly = false,
  onSave,
  saveDisabled,
  saving = false,
}: StoryboardScriptEditorProps) {
  const updateText = useCallback(
    (id: string, text: string) => {
      onSegmentChange(id, text);
    },
    [onSegmentChange],
  );

  const splitAt = useCallback(
    (id: string, cursorPos: number) => {
      const seg = segments.find((s) => s.id === id);
      if (!seg) return;
      if (onSplitAt) {
        onSplitAt(id, cursorPos, seg.text);
        return;
      }
      const idx = segments.findIndex((s) => s.id === id);
      if (idx === -1) return;
      const before = seg.text.slice(0, cursorPos);
      const after = seg.text.slice(cursorPos);
      const newSeg: StoryboardSegment = {
        id: `seg-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        text: after.trimStart(),
      };
      const updated = [...segments];
      updated[idx] = { ...seg, text: before.trimEnd() };
      updated.splice(idx + 1, 0, newSeg);
      onSegmentChange(id, before.trimEnd());
      onSegmentChange(newSeg.id, newSeg.text);
    },
    [onSegmentChange, onSplitAt, segments],
  );

  const mergeUp = useCallback(
    (idx: number) => {
      if (idx <= 0) return;
      const lower = segments[idx];
      const upper = segments[idx - 1];
      const mergedText = `${upper.text.trimEnd()}\n\n${lower.text.trimStart()}`;
      if (onMergeUp) {
        onMergeUp(lower.id, mergedText);
        return;
      }
      const updated = [...segments];
      updated[idx - 1] = {
        ...updated[idx - 1],
        text: mergedText,
      };
      updated.splice(idx, 1);
      onSegmentChange(upper.id, mergedText);
    },
    [onMergeUp, onSegmentChange, segments],
  );

  const handleDownload = useCallback(() => {
    downloadStoryboardSegments(segments);
  }, [segments]);

  if (segments.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center px-6 text-center">
        <div className="relative mb-6">
          <div className="absolute -inset-4 rounded-2xl bg-foreground/[0.02]" />
          <div className="relative grid h-14 w-14 place-items-center rounded-2xl bg-foreground/[0.04] ring-1 ring-inset ring-foreground/[0.06]">
            <Layout className="h-6 w-6 text-foreground/40" />
          </div>
        </div>
        <p className="text-[15px] font-semibold text-foreground/90">
          Building shot plan
        </p>
        <p className="mt-2 max-w-xs text-sm leading-relaxed text-muted-foreground">
          The story is being divided into production-ready shots.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border/40 px-4 py-2">
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
          <Scissors className="h-3 w-3" />
          Shift+Enter Split
          <span className="mx-1 text-border/60">|</span>
          <Merge className="h-3 w-3" />
          Hover center to merge
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => void onSave?.()}
            disabled={saveDisabled ?? readOnly ?? saving}
            className={cn(
              "flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-muted-foreground transition-colors hover:bg-foreground/[0.06] hover:text-foreground",
              "disabled:pointer-events-none disabled:opacity-40",
            )}
          >
            <Save className="h-3 w-3" />
            {saving ? "Saving..." : "Save"}
          </button>
          <button
            type="button"
            onClick={handleDownload}
            className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-muted-foreground transition-colors hover:bg-foreground/[0.06] hover:text-foreground"
          >
            <Download className="h-3 w-3" />
            Download
          </button>
          <span className="rounded-md bg-foreground/[0.05] px-2 py-0.5 text-[11px] tabular-nums text-muted-foreground">
            {segments.length} shots
          </span>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3 scrollbar-thin">
        <div className="space-y-0">
          {segments.map((seg, idx) => (
            <SegmentBlock
              key={seg.id}
              segment={seg}
              index={idx}
              total={segments.length}
              readOnly={readOnly}
              splitMergeDisabled={splitMergeDisabled}
              onTextChange={(text) => updateText(seg.id, text)}
              onFocus={() => onSegmentFocus?.(seg.id)}
              onBlur={() => onSegmentBlur?.(seg.id)}
              onSplit={(cursorPos) => splitAt(seg.id, cursorPos)}
              onMergeUp={() => mergeUp(idx)}
              onDeleteSegment={(segmentId) => onDeleteSegment(segmentId)}
            />
          ))}
        </div>
      </div>

      {/* bottom action bar */}
      <div className="shrink-0 border-t border-border/50 px-4 py-3">
        {onAutoGenerate ? (
          <div className="flex  gap-2">
            <Button
              onClick={onAutoGenerate}
              variant="outline"
              disabled={segments.length === 0 || applyNextDisabled}
              className={cn(
                "w-full rounded-lg text-[13px] font-medium",
                // "bg-foreground text-background",
                // "hover:bg-foreground/90 active:scale-[0.99]",
                "h-9 transition-all duration-200",
                "disabled:opacity-40",
              )}
            >
              {autoGenerateLabel}
              <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
            </Button>
            <Button
              onClick={onApplyNext}
              disabled={segments.length === 0 || applyNextDisabled}
              variant="outline"
              className={cn(
                "w-full rounded-lg text-[13px] font-medium",
                "h-9 transition-all duration-200",
                "disabled:opacity-40",
              )}
            >
              {applyNextLabel}
              <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
            </Button>
          </div>
        ) : (
          <Button
            onClick={onApplyNext}
            disabled={segments.length === 0 || applyNextDisabled}
            className={cn(
              "w-full rounded-lg text-[13px] font-medium",
              "bg-foreground text-background",
              "hover:bg-foreground/90 active:scale-[0.99]",
              "h-9 transition-all duration-200",
              "disabled:opacity-40",
            )}
          >
            {applyNextLabel}
            <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
          </Button>
        )}
      </div>
    </div>
  );
}

function SegmentBlock({
  segment,
  index,
  total,
  readOnly = false,
  splitMergeDisabled = false,
  onTextChange,
  onFocus,
  onBlur,
  onSplit,
  onMergeUp,
  onDeleteSegment,
}: {
  segment: StoryboardSegment;
  index: number;
  total: number;
  readOnly?: boolean;
  splitMergeDisabled?: boolean;
  onTextChange: (text: string) => void;
  onFocus?: () => void;
  onBlur?: () => void;
  onSplit: (cursorPos: number) => void;
  onMergeUp: () => void;
  onDeleteSegment: (segmentId: string) => void;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (
      readOnly ||
      splitMergeDisabled ||
      e.key !== "Enter" ||
      !e.shiftKey ||
      e.nativeEvent.isComposing
    ) {
      return;
    }
    e.preventDefault();
    const pos = e.currentTarget.selectionStart;
    onSplit(pos);
  };

  return (
    <div className="group">
      {index > 0 && (
        <div className="relative flex items-center gap-2 py-1">
          <div className="h-px flex-1 bg-border/50" />
          <button
            type="button"
            onClick={onMergeUp}
            disabled={splitMergeDisabled}
            className="flex items-center gap-1 rounded-md px-2 py-0.5 text-[10px] text-muted-foreground opacity-0 transition-all duration-200 hover:bg-foreground/[0.06] hover:text-foreground group-hover:opacity-100 disabled:pointer-events-none disabled:opacity-30"
          >
            <Merge className="h-2.5 w-2.5" />
            Merge
          </button>
          <div className="h-px flex-1 bg-border/50" />
        </div>
      )}
      <div
        className={cn(
          "relative rounded-xl border border-border/40 bg-card/20 p-3 transition-all duration-200",
          "focus-within:border-foreground/15 focus-within:bg-card/40 focus-within:shadow-sm",
        )}
      >
        <div className="mb-1.5 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <GripVertical className="h-3 w-3 text-muted-foreground/30" />
            <span className="text-[10px] font-semibold tabular-nums tracking-wider text-muted-foreground/60">
              Shot {String(index + 1).padStart(2, "0")}/
              {String(total).padStart(2, "0")}
            </span>
          </div>
          {/* <span className="text-[10px] tabular-nums text-muted-foreground/40">
            {segment.text.length} chars
          </span> */}
          <span
            className="text-[12px] tabular-nums text-muted-foreground/40 cursor-pointer hover:text-muted-foreground transition-all duration-200"
            onClick={() => onDeleteSegment(segment.id)}
          >
            Delete
          </span>
        </div>
        <textarea
          ref={textareaRef}
          value={segment.text}
          readOnly={readOnly}
          onFocus={onFocus}
          onBlur={onBlur}
          onChange={(e) => {
            if (readOnly) return;
            onTextChange(e.target.value);
            e.target.style.height = "auto";
            e.target.style.height = `${e.target.scrollHeight}px`;
          }}
          onKeyDown={onKeyDown}
          rows={2}
          className={cn(
            "w-full resize-none bg-transparent text-[13px] leading-[1.85] text-foreground/88",
            "focus:outline-none",
            "min-h-[56px]",
          )}
        />
      </div>
    </div>
  );
}
