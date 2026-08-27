import { ChevronDown, Globe } from "lucide-react";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

export type StoryLanguage = "zh" | "en";

export const STORY_LANGUAGE_PRESETS: readonly StoryLanguage[] = [
  "zh",
  "en",
] as const;

export const DEFAULT_STORY_LANGUAGE: StoryLanguage = "zh";

interface LanguagePickerProps {
  value: StoryLanguage;
  onChange: (value: StoryLanguage) => void;
  disabled?: boolean;
  size?: "hero" | "thread";
}

export function LanguagePicker({
  value,
  onChange,
  disabled,
  size = "thread",
}: LanguagePickerProps) {
  const isHero = size === "hero";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild disabled={disabled}>
        <span
          aria-label="Film language"
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
          <Globe className="h-3 w-3" aria-hidden />
          <span>{value}</span>
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
        className="w-[var(--radix-dropdown-menu-trigger-width)] min-w-[var(--radix-dropdown-menu-trigger-width)] rounded-xl p-1 shadow-md"
      >
        <DropdownMenuLabel
          style={{
            wordBreak: "keep-all",
          }}
          className="px-2.5 py-1.5 text-[10px] font-semibold text-foreground"
        >
          Film Language
        </DropdownMenuLabel>
        {STORY_LANGUAGE_PRESETS.map((preset, index) => (
          <div key={preset}>
            {index > 0 ? <DropdownMenuSeparator className="my-0.5" /> : null}
            <DropdownMenuItem
              onSelect={() => onChange(preset)}
              className={cn(
                "cursor-pointer rounded-lg px-2.5 py-1.5 text-[12px]",
                preset === value &&
                  "bg-foreground/[0.06] font-medium text-foreground/85",
              )}
            >
              {preset}
            </DropdownMenuItem>
          </div>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
