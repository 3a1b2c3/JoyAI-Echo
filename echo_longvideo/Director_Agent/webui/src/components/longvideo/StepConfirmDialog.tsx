import { ArrowRight } from "lucide-react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

import type { StepId } from "./StepHeader";

const STEP_CONFIRM: Partial<Record<StepId, { title: string; desc: string }>> = {
  2: { title: "Build the shot plan?", desc: "The story will be split into editable shot segments." },
  3: { title: "Start shot generation?", desc: "Confirm the shot plan before generation begins." },
  4: { title: "Assemble the final cut?", desc: "All approved shots will be merged into one film." },
};

interface StepConfirmDialogProps {
  open: boolean;
  to?: StepId;
  /** 仅提示、不可继续时使用（单按钮） */
  alert?: { title: string; desc: string; actionLabel?: string };
  onConfirm: () => void;
  onCancel: () => void;
}

export function StepConfirmDialog({
  open,
  to,
  alert,
  onConfirm,
  onCancel,
}: StepConfirmDialogProps) {
  const copy = alert ?? STEP_CONFIRM[to ?? 2] ?? { title: "", desc: "" };
  const isAlert = Boolean(alert);
  const actionLabel = alert?.actionLabel ?? "Got It";

  return (
    <AlertDialog open={open} onOpenChange={(v) => !v && onCancel()}>
      <AlertDialogContent className="max-w-sm">
        <AlertDialogHeader>
          <AlertDialogTitle className="text-[15px]">
            {copy.title}
          </AlertDialogTitle>
          <AlertDialogDescription className="mt-1 text-[13px]">
            {copy.desc}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter className="mt-2">
          {isAlert ? (
            <AlertDialogAction
              onClick={onConfirm}
              className="h-8 rounded-lg px-3 text-[13px]"
            >
              {actionLabel}
            </AlertDialogAction>
          ) : (
            <>
              <AlertDialogCancel
                onClick={onCancel}
                className="h-8 rounded-lg px-3 text-[13px]"
              >
                Cancel
              </AlertDialogCancel>
              <AlertDialogAction
                onClick={onConfirm}
                className="h-8 rounded-lg px-3 text-[13px]"
              >
                Confirm
                <ArrowRight className="ml-1 h-3.5 w-3.5" />
              </AlertDialogAction>
            </>
          )}
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
