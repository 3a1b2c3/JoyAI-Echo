import {
  forwardRef,
  useCallback,
  useMemo,
  type CompositionEventHandler,
  type FormEventHandler,
} from "react";

import { countMixedUnits, truncateToMaxUnits } from "@/lib/countText";
import { cn } from "@/lib/utils";

export interface CountableTextareaProps
  extends Omit<
    React.TextareaHTMLAttributes<HTMLTextAreaElement>,
    "onChange" | "value"
  > {
  value: string;
  onChange: (value: string) => void;
  /** Mixed unit limit; excess input is silently truncated. */
  maxUnits?: number;
  /** Whether to show the count label. Default true. */
  showCount?: boolean;
  /** Count label placement. Default "bottom-right". */
  countPosition?: "bottom-right" | "none";
  /** Custom count formatter. */
  formatCount?: (count: number) => string;
}

const MAX_HEIGHT_PX = 260;

export const CountableTextarea = forwardRef<
  HTMLTextAreaElement,
  CountableTextareaProps
>(function CountableTextarea(
  {
    value,
    onChange,
    maxUnits,
    showCount = true,
    countPosition = "bottom-right",
    formatCount,
    className,
    onInput,
    onCompositionEnd,
    ...props
  },
  ref,
) {
  const count = useMemo(() => countMixedUnits(value), [value]);

  const applyValue = useCallback(
    (next: string) => {
      const limited =
        maxUnits !== undefined ? truncateToMaxUnits(next, maxUnits) : next;
      onChange(limited);
    },
    [maxUnits, onChange],
  );

  const resize = useCallback((el: HTMLTextAreaElement) => {
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT_PX)}px`;
  }, []);

  const handleInput: FormEventHandler<HTMLTextAreaElement> = useCallback(
    (e) => {
      resize(e.currentTarget);
      onInput?.(e);
    },
    [onInput, resize],
  );

  const handleCompositionEnd: CompositionEventHandler<
    HTMLTextAreaElement
  > = useCallback(
    (e) => {
      if (maxUnits !== undefined) {
        applyValue(e.currentTarget.value);
      }
      onCompositionEnd?.(e);
    },
    [applyValue, maxUnits, onCompositionEnd],
  );

  const handleChange: React.ChangeEventHandler<HTMLTextAreaElement> =
    useCallback(
      (e) => {
        applyValue(e.target.value);
      },
      [applyValue],
    );

  const showCountLabel =
    showCount && countPosition === "bottom-right"
      ? formatCount
        ? formatCount(count)
        : `${count} characters`
      : null;

  return (
    <div className="relative">
      <textarea
        ref={ref}
        value={value}
        onChange={handleChange}
        onInput={handleInput}
        onCompositionEnd={handleCompositionEnd}
        className={cn(className, showCountLabel && "pr-14")}
        {...props}
      />
      {showCountLabel ? (
        <span
          className={cn(
            "pointer-events-none absolute bottom-2 right-3 select-none",
            "text-[11px] tabular-nums text-muted-foreground/60",
          )}
          aria-live="polite"
          aria-atomic="true"
        >
          {showCountLabel}
        </span>
      ) : null}
    </div>
  );
});

CountableTextarea.displayName = "CountableTextarea";
