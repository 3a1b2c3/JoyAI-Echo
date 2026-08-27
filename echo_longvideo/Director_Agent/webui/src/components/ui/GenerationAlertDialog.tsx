import { AlertTriangle } from "lucide-react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

export type GenerationAlertVariant = "congested" | "error";

const COPY: Record<
  GenerationAlertVariant,
  { title: string; description: string }
> = {
  congested: {
    title: "Generation service is busy",
    description: "The generation service is at capacity. Please wait a moment.",
  },
  error: {
    title: "Generation error",
    description: "Generation failed. Please try again.",
  },
};

interface GenerationAlertDialogProps {
  open: boolean;
  variant: GenerationAlertVariant;
  onDismiss: () => void;
}

/** Single-action alert for shot generation congested / error states. */
export function GenerationAlertDialog({
  open,
  variant,
  onDismiss,
}: GenerationAlertDialogProps) {
  const copy = COPY[variant];

  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        if (!next) onDismiss();
      }}
    >
      <AlertDialogContent className="max-w-sm">
        <AlertDialogHeader>
          <AlertDialogTitle className="flex items-center gap-2 text-[15px]">
            <AlertTriangle
              className="h-5 w-5 shrink-0 text-destructive"
              aria-hidden
            />
            <span>{copy.title}</span>
          </AlertDialogTitle>
          <AlertDialogDescription className="sr-only">
            {copy.description}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter className="mt-2">
          <AlertDialogAction
            onClick={onDismiss}
            className="h-8 rounded-lg px-3 text-[13px]"
          >
            Got It
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
