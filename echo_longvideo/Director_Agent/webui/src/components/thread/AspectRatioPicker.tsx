import { ChevronDown, Ratio } from "lucide-react";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

export type VideoSize = { width: number; height: number };

export const VIDEO_SIZE_PRESETS = [
  { label: "16:9", width: 1280, height: 736 },
  { label: "1:1", width: 736, height: 736 },
  { label: "9:16", width: 736, height: 1280 },
] as const;

export const DEFAULT_VIDEO_SIZE: VideoSize = VIDEO_SIZE_PRESETS[0];

interface AspectRatioPickerProps {
  value: VideoSize;
  onChange: (value: VideoSize) => void;
  disabled?: boolean;
  size?: "hero" | "thread";
}

export function AspectRatioPicker({
  value,
  onChange,
  disabled,
  size = "thread",
}: AspectRatioPickerProps) {
  const isHero = size === "hero";
  const selected =
    VIDEO_SIZE_PRESETS.find(
      (preset) =>
        preset.width === value.width && preset.height === value.height,
    ) ?? VIDEO_SIZE_PRESETS[0];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild disabled={disabled}>
        <span
          aria-label="Select aspect ratio"
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
          <Ratio className="h-3 w-3" aria-hidden />
          <span>{selected.label}</span>
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
        className="min-w-[var(--radix-dropdown-menu-trigger-width)] rounded-xl p-1"
      >
        {VIDEO_SIZE_PRESETS.map((preset) => {
          const isSelected =
            preset.width === value.width && preset.height === value.height;
          return (
            <DropdownMenuItem
              key={preset.label}
              onSelect={() =>
                onChange({ width: preset.width, height: preset.height })
              }
              title={`${preset.width} × ${preset.height}`}
              className={cn(
                "cursor-pointer",
                "justify-center rounded-lg px-3 py-1.5 font-mono text-[12px]",
                isSelected &&
                  "bg-foreground/[0.06] font-medium text-foreground/85",
              )}
            >
              {preset.label}
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
