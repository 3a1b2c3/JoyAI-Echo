import {
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useState,
} from "react";

import { cn } from "@/lib/utils";

const STORAGE_KEY = "nanobot-webui.split-ratio";
const DEFAULT_RATIO = 0.45;
const MIN_LEFT_PX = 360;
const MIN_RIGHT_PX = 420;

function readRatio(): number {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_RATIO;
    const n = Number(raw);
    return Number.isFinite(n)
      ? Math.min(0.7, Math.max(0.25, n))
      : DEFAULT_RATIO;
  } catch {
    return DEFAULT_RATIO;
  }
}

interface SplitPaneProps {
  left: ReactNode;
  right: ReactNode;
  className?: string;
}

export function SplitPane({ left, right, className }: SplitPaneProps) {
  const [ratio, setRatio] = useState(readRatio);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, String(ratio));
    } catch {}
  }, [ratio]);

  const onPointerDown = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(true);
    const container = (e.currentTarget as HTMLElement).parentElement!;
    const rect = container.getBoundingClientRect();

    const onMove = (me: PointerEvent) => {
      const x = me.clientX - rect.left;
      const total = rect.width;
      const leftPx = Math.max(MIN_LEFT_PX, Math.min(total - MIN_RIGHT_PX, x));
      setRatio(leftPx / total);
    };
    const onUp = () => {
      setDragging(false);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }, []);

  const pct = `${(ratio * 100).toFixed(2)}%`;

  return (
    <div
      className={cn(
        "relative flex h-full min-h-0 w-full overflow-hidden",
        className,
      )}
      style={dragging ? { userSelect: "none" } : undefined}
    >
      <div
        className="flex h-full min-w-0 flex-col overflow-hidden"
        style={{ width: right ? pct : "100%" }}
      >
        {left}
      </div>

      {/* divider */}
      {right ? (
        <div
          className="group/divider relative z-10 w-3 shrink-0 cursor-col-resize"
          onPointerDown={onPointerDown}
        >
          <div
            className={cn(
              "absolute inset-y-0 left-1/2 w-px -translate-x-1/2 transition-colors duration-200",
              dragging ? "bg-foreground/20" : "bg-border/70",
            )}
          />
          <div
            className={cn(
              "absolute left-1/2 top-1/2 h-10 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full transition-all duration-200",
              dragging
                ? "bg-foreground/30 shadow-sm"
                : "bg-border/80 group-hover/divider:bg-foreground/20 group-hover/divider:shadow-sm",
            )}
          />
        </div>
      ) : null}

      <div className="flex h-full min-w-0 flex-1 flex-col overflow-hidden">
        {right}
      </div>
    </div>
  );
}
