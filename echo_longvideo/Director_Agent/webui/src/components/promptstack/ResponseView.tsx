import { useState, useCallback, type PointerEvent as ReactPointerEvent } from "react";
import type { TraceRecord } from "@/hooks/usePromptStack";

interface Props {
  record: TraceRecord;
}

const MSG_SPLIT_KEY = "promptstack.msg-split";
const MIN_TOP = 80;
const MIN_BOTTOM = 80;

function readMsgSplit(): number {
  try {
    const raw = window.localStorage.getItem(MSG_SPLIT_KEY);
    if (raw) {
      const val = Number(raw);
      if (Number.isFinite(val) && val > 0.1 && val < 0.9) return val;
    }
  } catch { /* ignore */ }
  return 0.4;
}

export function ResponseView({ record }: Props) {
  const [expandedMsgs, setExpandedMsgs] = useState<Set<number>>(new Set());
  const [msgSplit, setMsgSplit] = useState(readMsgSplit);
  const { response } = record;

  const toggleMsg = (idx: number) => {
    setExpandedMsgs((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const startMsgResize = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      event.preventDefault();
      const container = event.currentTarget.parentElement as HTMLElement;
      const rect = container.getBoundingClientRect();
      const h = rect.height;

      const onMove = (e: PointerEvent) => {
        const ratio = Math.max(
          MIN_TOP / h,
          Math.min(1 - MIN_BOTTOM / h, (e.clientY - rect.top) / h),
        );
        setMsgSplit(ratio);
      };
      const onUp = () => {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        try { window.localStorage.setItem(MSG_SPLIT_KEY, String(msgSplit)); } catch { /* */ }
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    },
    [msgSplit],
  );

  const roleColor = (role: string) => {
    switch (role) {
      case "assistant": return "text-green-600 dark:text-green-400";
      case "user": return "text-blue-600 dark:text-blue-400";
      case "system": return "text-purple-600 dark:text-purple-400";
      default: return "text-orange-600 dark:text-orange-400";
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Upper section: response summary — scrolls independently */}
      <div className="overflow-y-auto shrink-0" style={{ height: `${msgSplit * 100}%` }}>
        <div className="flex flex-col gap-3 p-1">
          {/* Usage */}
          {response.usage && (
            <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
              {Object.entries(response.usage).map(([key, val]) => (
                <span key={key} className="font-mono">
                  {key}: {val}
                </span>
              ))}
            </div>
          )}

          {/* Tool calls */}
          {response.tool_calls && response.tool_calls.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-xs font-medium text-muted-foreground">Tool Calls:</span>
              <div className="flex flex-wrap gap-1">
                {response.tool_calls.map((tc, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center rounded-md bg-orange-500/10 px-2 py-0.5 text-xs font-medium text-orange-700 dark:text-orange-300"
                  >
                    {tc.name}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Reasoning */}
          {response.reasoning_content && (
            <div className="flex flex-col gap-1">
              <span className="text-xs font-medium text-muted-foreground">Reasoning:</span>
              <pre className="text-xs whitespace-pre-wrap break-words font-mono bg-muted/30 rounded-md p-2 max-h-40 overflow-auto">
                {response.reasoning_content}
              </pre>
            </div>
          )}

          {/* Response content */}
          {response.content && (
            <div className="flex flex-col gap-1">
              <span className="text-xs font-medium text-muted-foreground">Response:</span>
              <pre className="text-xs whitespace-pre-wrap break-words font-mono bg-muted/30 rounded-md p-3 max-h-60 overflow-auto">
                {response.content}
              </pre>
            </div>
          )}
        </div>
      </div>

      {/* Drag handle */}
      <div
        className="h-2 shrink-0 cursor-row-resize flex items-center justify-center hover:bg-muted/50 transition-colors group"
        onPointerDown={startMsgResize}
      >
        <div className="w-12 h-0.5 rounded-full bg-border group-hover:bg-foreground/30 transition-colors" />
      </div>

      {/* Lower section: messages list — independently scrollable */}
      <div className="flex-1 overflow-hidden flex flex-col border-t min-h-0">
        <div className="px-2 py-1 text-xs font-medium text-muted-foreground bg-muted/30 border-b shrink-0">
          Messages ({record.messages_count} total)
        </div>
        <div className="flex-1 overflow-y-auto">
          <div className="flex flex-col gap-0.5 p-1">
            {record.messages.map((msg, i) => (
              <div key={i} className="border rounded-md overflow-hidden">
                <button
                  onClick={() => toggleMsg(i)}
                  className="flex w-full items-center gap-2 px-2 py-1 text-xs hover:bg-muted/50 transition-colors text-left"
                >
                  <span className={`font-medium shrink-0 w-14 ${roleColor(msg.role)}`}>
                    {msg.role}
                  </span>
                  {msg.tool_name && (
                    <span className="text-orange-500 font-mono shrink-0">[{msg.tool_name}]</span>
                  )}
                  {msg.tool_calls && msg.tool_calls.length > 0 && (
                    <span className="text-orange-500 font-mono shrink-0">
                      [{msg.tool_calls.map(tc => tc.name).join(", ")}]
                    </span>
                  )}
                  <span className="text-muted-foreground truncate flex-1">{msg.preview}</span>
                  <span className="text-muted-foreground font-mono shrink-0 text-[10px]">
                    {msg.char_count >= 1000
                      ? `${(msg.char_count / 1000).toFixed(1)}k`
                      : msg.char_count}
                  </span>
                  <span className="text-muted-foreground text-[10px]">
                    {expandedMsgs.has(i) ? "▼" : "▶"}
                  </span>
                </button>
                {expandedMsgs.has(i) && (
                  <div className="border-t bg-muted/20 p-2 max-h-80 overflow-auto">
                    <pre className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/80">
                      {msg.content}
                    </pre>
                    {msg.tool_calls && msg.tool_calls.length > 0 && (
                      <div className="mt-2 border-t pt-2">
                        <span className="text-[10px] text-muted-foreground font-medium">Tool Call Arguments:</span>
                        {msg.tool_calls.map((tc, j) => (
                          <pre key={j} className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/60 mt-1">
                            {tc.name}: {tc.arguments}
                          </pre>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
