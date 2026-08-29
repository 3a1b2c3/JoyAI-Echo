import { useEffect, useState } from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { FlaskConical } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { PeSet } from "@/lib/pe-api";

interface Props {
  open: boolean;
  sets: PeSet[];
  onConfirm: (name: string) => void;
}

/**
 * Mandatory PE-set picker shown before a new conversation starts.
 *
 * Non-dismissable on purpose: the selected set is injected into the session's
 * system prompt for the whole conversation, so an A/B run must begin from an
 * explicit, conscious choice. Escape / outside-click are suppressed and there
 * is no close affordance — the only exit is confirming a selection.
 */
export function PeSelectModal({ open, sets, onConfirm }: Props) {
  const [selected, setSelected] = useState<string | null>(null);

  // Force a conscious choice each time the modal is (re)opened.
  useEffect(() => {
    if (open) setSelected(null);
  }, [open]);

  return (
    <DialogPrimitive.Root open={open}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm data-[state=open]:animate-in data-[state=open]:fade-in-0" />
        <DialogPrimitive.Content
          onEscapeKeyDown={(e) => e.preventDefault()}
          onPointerDownOutside={(e) => e.preventDefault()}
          onInteractOutside={(e) => e.preventDefault()}
          className="fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 sm:rounded-lg"
        >
          <div className="flex flex-col space-y-1.5">
            <DialogPrimitive.Title className="flex items-center gap-2 text-lg font-semibold leading-none tracking-tight">
              <FlaskConical className="h-4 w-4" /> Select Prompt Kit
            </DialogPrimitive.Title>
            <DialogPrimitive.Description className="text-sm text-muted-foreground">
              Choose a prompt kit for this project before the conversation begins.
            </DialogPrimitive.Description>
          </div>
          <div className="flex max-h-[50vh] flex-col gap-1.5 overflow-y-auto">
            {sets.map((s) => (
              <button
                key={s.name}
                type="button"
                onClick={() => setSelected(s.name)}
                className={cn(
                  "flex flex-col items-start gap-0.5 rounded-md border px-3 py-2 text-left text-sm transition-colors",
                  selected === s.name
                    ? "border-primary bg-primary/10"
                    : "border-border hover:bg-accent/40",
                )}
              >
                <span className="font-medium">{s.label || s.name}</span>
                {s.description ? (
                  <span className="text-xs text-muted-foreground">
                    {s.description}
                  </span>
                ) : null}
              </button>
            ))}
          </div>
          <div className="flex justify-end">
            <Button
              disabled={!selected}
              onClick={() => selected && onConfirm(selected)}
            >
              Start Project
            </Button>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
