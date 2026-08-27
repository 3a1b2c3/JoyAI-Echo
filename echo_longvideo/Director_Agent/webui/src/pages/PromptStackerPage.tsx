import { useState, useMemo, useCallback, type PointerEvent as ReactPointerEvent } from "react";
import { usePromptStack } from "@/hooks/usePromptStack";
import { SessionList } from "@/components/promptstack/SessionList";
import { TraceTimeline } from "@/components/promptstack/TraceTimeline";
import { PartStack } from "@/components/promptstack/PartStack";
import { ResponseView } from "@/components/promptstack/ResponseView";
import { FullContextLog } from "@/components/promptstack/FullContextLog";

const SPLIT_STORAGE_KEY = "promptstack.split-height";
const MIN_TOP_HEIGHT = 200;
const MIN_BOTTOM_HEIGHT = 120;

function readSplitRatio(): number {
  try {
    const raw = window.localStorage.getItem(SPLIT_STORAGE_KEY);
    if (raw) {
      const val = Number(raw);
      if (Number.isFinite(val) && val > 0.2 && val < 0.9) return val;
    }
  } catch { /* ignore */ }
  return 0.55;
}

export function PromptStackerPage() {
  const { sessions, activeSession, setActiveSession, trace, loading, refresh } =
    usePromptStack();
  const [activeTurn, setActiveTurn] = useState<string | null>(null);
  const [splitRatio, setSplitRatio] = useState(readSplitRatio);

  const activeRecord = useMemo(() => {
    if (!activeTurn) return trace.length > 0 ? trace[trace.length - 1] : null;
    return trace.find((r) => r.id === activeTurn) ?? null;
  }, [trace, activeTurn]);

  const effectiveTurn = activeTurn ?? (trace.length > 0 ? trace[trace.length - 1].id : null);

  const startResize = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      event.preventDefault();
      const container = (event.currentTarget.parentElement as HTMLElement);
      const containerRect = container.getBoundingClientRect();
      const containerHeight = containerRect.height;

      const onMove = (e: PointerEvent) => {
        const relativeY = e.clientY - containerRect.top;
        const ratio = Math.max(
          MIN_TOP_HEIGHT / containerHeight,
          Math.min(1 - MIN_BOTTOM_HEIGHT / containerHeight, relativeY / containerHeight)
        );
        setSplitRatio(ratio);
      };
      const onUp = () => {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        try {
          window.localStorage.setItem(SPLIT_STORAGE_KEY, String(splitRatio));
        } catch { /* ignore */ }
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    },
    [splitRatio],
  );

  return (
    <div className="flex h-screen w-screen bg-background text-foreground">
      {/* Sidebar: sessions */}
      <aside className="w-64 shrink-0 border-r overflow-y-auto flex flex-col">
        <div className="flex items-center justify-between px-3 py-3 border-b">
          <h1 className="text-sm font-semibold">Prompt Stacker</h1>
          <div className="flex gap-1">
            <button
              onClick={refresh}
              className="rounded-md px-2 py-1 text-xs hover:bg-muted transition-colors"
              title="Refresh"
            >
              Refresh
            </button>
            <a
              href="/"
              className="rounded-md px-2 py-1 text-xs hover:bg-muted transition-colors"
            >
              Chat
            </a>
          </div>
        </div>
        <SessionList
          sessions={sessions}
          activeSession={activeSession}
          onSelect={(id) => {
            setActiveSession(id);
            setActiveTurn(null);
          }}
        />
      </aside>

      {/* Main content */}
      <main className="flex flex-1 flex-col min-w-0 overflow-hidden">
        {loading ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            Loading trace...
          </div>
        ) : trace.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            {activeSession
              ? "No turns recorded in this session"
              : "Select a session from the sidebar"}
          </div>
        ) : (
          <div className="flex flex-1 flex-col overflow-hidden relative">
            {/* Timeline */}
            <TraceTimeline
              trace={trace}
              activeTurn={effectiveTurn}
              onSelect={setActiveTurn}
            />

            {/* Resizable split area */}
            <div className="flex-1 flex flex-col overflow-hidden relative">
              {/* Upper: Part Stack + Response */}
              <div
                className="overflow-hidden flex"
                style={{ height: `${splitRatio * 100}%` }}
              >
                {activeRecord && (
                  <>
                    {/* Left: Part Stack */}
                    <div className="flex-1 overflow-y-auto p-4 border-r">
                      <div className="flex items-center gap-2 mb-3">
                        <h2 className="text-sm font-semibold">Prompt Parts</h2>
                        <span className="text-xs text-muted-foreground font-mono">
                          {activeRecord.model} &middot; iter #{activeRecord.iteration}
                        </span>
                      </div>
                      <PartStack parts={activeRecord.parts} />
                    </div>

                    {/* Right: Response */}
                    <div className="w-[45%] shrink-0 overflow-hidden flex flex-col">
                      <h2 className="text-sm font-semibold px-4 pt-4 pb-2 shrink-0">Model Response</h2>
                      <div className="flex-1 min-h-0 px-4 pb-2">
                        <ResponseView record={activeRecord} />
                      </div>
                    </div>
                  </>
                )}
              </div>

              {/* Drag handle */}
              <div
                className="h-2 shrink-0 cursor-row-resize flex items-center justify-center hover:bg-muted/50 transition-colors group"
                onPointerDown={startResize}
              >
                <div className="w-16 h-0.5 rounded-full bg-border group-hover:bg-foreground/30 transition-colors" />
              </div>

              {/* Lower: Full scrolling context log */}
              <div
                className="overflow-hidden border-t"
                style={{ height: `${(1 - splitRatio) * 100}%` }}
              >
                <FullContextLog trace={trace} activeTurn={effectiveTurn} />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
