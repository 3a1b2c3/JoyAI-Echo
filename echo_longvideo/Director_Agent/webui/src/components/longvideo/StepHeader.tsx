import { Check, ChevronLeft, ChevronRight } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export type StepId = 1 | 2 | 3 | 4;

const STEPS: Array<{ id: StepId; label: string }> = [
  { id: 1, label: "Story" },
  { id: 2, label: "Shot Plan" },
  { id: 3, label: "Generate" },
  { id: 4, label: "Final Cut" },
];

interface StepHeaderProps {
  currentStep: StepId;
  onPrev: () => void;
  onNext: () => void;
  progress?: { done: number; total: number } | null;
}

export function StepHeader({
  currentStep,
  onPrev,
  onNext,
  progress,
}: StepHeaderProps) {
  const canGoBack = false; // currentStep <= 2 && currentStep > 1;
  const progressPct =
    progress && progress.total > 0
      ? Math.round((progress.done / progress.total) * 100)
      : null;

  return (
    <div className="relative shrink-0">
      <div className="flex items-center gap-1 px-3 py-2.5">
        <Button
          variant="ghost"
          size="icon"
          disabled={!canGoBack}
          onClick={onPrev}
          className={cn(
            "h-7 w-7 rounded-lg text-muted-foreground transition-all duration-200 hover:bg-accent/60 hover:text-foreground",
            !canGoBack && "invisible",
          )}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>

        <div className="flex flex-1 items-center justify-center gap-0">
          {STEPS.map((step, idx) => {
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
                    "group relative flex flex-col items-center gap-0.5 rounded-xl px-4 py-2 transition-all duration-300",
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
                      "text-[13px] transition-all duration-300",
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

                  {active && progressPct !== null && (
                    <span className="mt-0.5 text-[13px] tabular-nums text-foreground/40">
                      {progressPct}%
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {progress && (
          <span className="mr-1 text-[15px] tabular-nums text-muted-foreground/70">
            {progress.done}/{progress.total}
          </span>
        )}

        <Button
          variant="ghost"
          size="icon"
          onClick={onNext}
          disabled={currentStep >= 4}
          className={cn(
            "h-7 w-7 rounded-lg text-muted-foreground transition-all duration-200 hover:bg-accent/60 hover:text-foreground",
            currentStep >= 4 && "invisible",
          )}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>

      {progress && progress.total > 0 && (
        <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-border/40">
          <div
            className="h-full bg-foreground/30 transition-all duration-700 ease-out"
            style={{ width: `${(progress.done / progress.total) * 100}%` }}
          />
        </div>
      )}

      {!progress && <div className="h-px w-full bg-border/60" />}
    </div>
  );
}
