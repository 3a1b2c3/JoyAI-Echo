import { useCallback, useRef, useState } from "react";
import { ArrowRight, Download, Save } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

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

interface ScriptEditorProps {
  value: string;
  onChange: (text: string) => void;
  onApplyNext: () => void;
  applyNextDisabled?: boolean;
  /** 底部主按钮文案；父组件在 workflow 等待期可传入进度提示（对齐 StoryboardScriptEditor）。 */
  applyNextLabel?: string;
  readOnly?: boolean;
  onFocus?: () => void;
  onBlur?: () => void;
  onSave?: () => void | Promise<void>;
  saveDisabled?: boolean;
  saving?: boolean;
}

export function ScriptEditor({
  value,
  onChange,
  onApplyNext,
  applyNextDisabled = false,
  applyNextLabel = "Next Step",
  readOnly = false,
  onFocus,
  onBlur,
  onSave,
  saveDisabled,
  saving = false,
}: ScriptEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [manualEdit] = useState(false);

  const handleDownload = useCallback(() => {
    const blob = new Blob([value], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "story-script.txt";
    a.click();
    URL.revokeObjectURL(url);
  }, [value]);

  const hasContent = value.trim().length > 0;
  const showEditor = hasContent || manualEdit;

  if (!showEditor) {
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
          <div className="border-t border-border/50 px-5 py-4">
            <div className="flex items-center justify-center">
              <span
                className="text-[13px] font-medium text-foreground/70"
                style={{ animation: "status-pulse 2.4s ease-in-out infinite" }}
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

  const wordCount = value.length;

  return (
    <div className="flex h-full flex-col">
      {/* toolbar */}
      <div className="flex items-center justify-between border-b border-border/40 px-4 py-2">
        <span className="text-[11px] tabular-nums text-muted-foreground">
          {wordCount} characters
        </span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => {
              void onSave?.();
            }}
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
        </div>
      </div>

      {/* editor */}
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3 scrollbar-thin">
        <textarea
          style={{ height: "100%", resize: "none" }}
          ref={textareaRef}
          value={value}
          readOnly={readOnly}
          onFocus={onFocus}
          onBlur={onBlur}
          onChange={(e) => {
            if (readOnly) return;
            onChange(e.target.value);
          }}
          placeholder="Edit the story script here..."
          className={cn(
            "w-full resize-none bg-transparent",
            "text-[13.5px] leading-[1.9] text-foreground/88",
            "placeholder:text-muted-foreground/35",
            "focus:outline-none",
            "min-h-[200px] pb-4",
          )}
        />
      </div>

      {/* bottom action bar */}
      <div className="shrink-0 border-t border-border/50 px-4 py-3">
        <Button
          onClick={onApplyNext}
          disabled={!hasContent || applyNextDisabled}
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
      </div>
    </div>
  );
}
