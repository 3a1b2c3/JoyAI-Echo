import { Film, ScanLine } from "lucide-react";

import { cn } from "@/lib/utils";

export type WorkflowMode = "quick" | "director";

interface WorkflowSelectorProps {
  onSelect: (mode: WorkflowMode) => void;
}

const workflows: Array<{
  mode: WorkflowMode;
  title: string;
  description: string;
  icon: typeof Film;
}> = [
  {
    mode: "quick",
    title: "Quick Film",
    description: "Describe an idea and let Director complete the full workflow.",
    icon: Film,
  },
  {
    mode: "director",
    title: "Director Workshop",
    description: "Review the story, references, and shots step by step.",
    icon: ScanLine,
  },
];

export function WorkflowSelector({ onSelect }: WorkflowSelectorProps) {
  return (
    <div className="flex h-full w-full items-center justify-center px-6">
      <div className="flex w-full max-w-2xl flex-col items-center gap-8">
        <div className="text-center">
          <h1 className="text-4xl font-light tracking-tight text-foreground/90">
            Imagine it. Direct it.
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Choose how you want to create with Echo Director.
          </p>
        </div>
        <div className="grid w-full gap-4 sm:grid-cols-2">
          {workflows.map(({ mode, title, description, icon: Icon }) => (
            <button
              key={mode}
              type="button"
              onClick={() => onSelect(mode)}
              className={cn(
                "group flex min-h-52 flex-col items-center justify-center gap-5 rounded-2xl p-8 text-center",
                "border border-border/60 bg-card/40 transition-all duration-200",
                "hover:-translate-y-1 hover:border-foreground/20 hover:bg-card hover:shadow-xl",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              )}
            >
              <span className="grid h-12 w-12 place-items-center rounded-xl bg-foreground/[0.06] ring-1 ring-inset ring-foreground/[0.08]">
                <Icon className="h-5 w-5 text-foreground/60" />
              </span>
              <span>
                <span className="block text-base font-medium text-foreground/90">
                  {title}
                </span>
                <span className="mt-2 block text-xs leading-5 text-muted-foreground">
                  {description}
                </span>
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
