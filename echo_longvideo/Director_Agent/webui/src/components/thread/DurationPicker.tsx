import { ChevronDown, Clock } from "lucide-react";
import { useTranslation } from "react-i18next";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

/** Short-video duration buckets. */
export const DURATION_OPTIONS = [5, 6, 7, 8, 9, 10];
export const DEFAULT_DURATION = 5;

/** Director automatic-generation duration buckets. */
export const LONG_VIDEO_DURATION_OPTIONS = [10, 20, 30, 60, 90, 120, 150, 180];
export const LONG_VIDEO_DEFAULT_DURATION = 30;

interface DurationPickerProps {
  value: number;
  onChange: (sec: number) => void;
  disabled?: boolean;
  size?: "hero" | "thread";
  options?: readonly number[];
}

export function DurationPicker({
  value,
  onChange,
  disabled,
  size = "thread",
  options = DURATION_OPTIONS,
}: DurationPickerProps) {
  const { t } = useTranslation();
  const isHero = size === "hero";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild disabled={disabled}>
        <span
          aria-label={t("thread.composer.durationPickerAria")}
          className={cn(
            "cursor-pointer",
            "group inline-flex min-w-0 items-center gap-1.5 rounded-full border px-2.5 py-1",
            "border-foreground/10 bg-foreground/[0.035] font-medium text-foreground/80",
            "whitespace-nowrap transition-colors hover:border-foreground/15 hover:bg-foreground/[0.04]",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-foreground/20",
            "disabled:cursor-not-allowed disabled:opacity-60",
            "data-[state=open]:border-foreground/15 data-[state=open]:bg-foreground/[0.04]",
            isHero ? "text-[11px]" : "text-[12px]",
          )}
        >
          <Clock className={cn(isHero ? "h-3 w-3" : "h-3 w-3")} aria-hidden />
          <span>{t("thread.composer.durationLabel", { sec: value })}</span>
          <ChevronDown
            className={cn(
              "h-2.5 w-2.5 opacity-60 transition-transform duration-200",
              "group-data-[state=open]:rotate-180",
            )}
            aria-hidden
          />
        </span>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side="top"
        align="start"
        sideOffset={6}
        className="max-h-[320px] min-w-[var(--radix-dropdown-menu-trigger-width)] overflow-y-auto rounded-xl p-1"
      >
        {options.map((sec) => (
          <DropdownMenuItem
            key={sec}
            onSelect={() => onChange(sec)}
            className={cn(
              "cursor-pointer",
              "justify-center rounded-lg px-3 py-1.5 font-mono text-[12px]",
              sec === value &&
                "bg-foreground/[0.06] font-medium text-foreground/85",
            )}
          >
            {sec}s
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
