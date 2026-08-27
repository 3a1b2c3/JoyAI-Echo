import { useState, useRef, useEffect } from "react";
import type { TraceRecord } from "@/hooks/usePromptStack";

interface Props {
  trace: TraceRecord[];
  activeTurn: string | null;
}

export function FullContextLog({ trace, activeTurn }: Props) {
  const [expandedTurns, setExpandedTurns] = useState<Set<string>>(new Set());
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  const activeRef = useRef<HTMLDivElement>(null);

  // Auto-expand active turn
  useEffect(() => {
    if (activeTurn && !expandedTurns.has(activeTurn)) {
      setExpandedTurns((prev) => new Set([...prev, activeTurn]));
    }
  }, [activeTurn]);

  // Auto-scroll to active turn
  useEffect(() => {
    if (activeRef.current) {
      activeRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [activeTurn]);

  const toggleTurn = (turnId: string) => {
    setExpandedTurns((prev) => {
      const next = new Set(prev);
      if (next.has(turnId)) next.delete(turnId);
      else next.add(turnId);
      return next;
    });
  };

  const toggleItem = (key: string) => {
    setExpandedItems((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const roleColor = (role: string) => {
    switch (role) {
      case "system": return "text-purple-600 dark:text-purple-400 border-l-purple-400";
      case "user": return "text-blue-600 dark:text-blue-400 border-l-blue-400";
      case "assistant": return "text-green-600 dark:text-green-400 border-l-green-400";
      case "assistant_response": return "text-emerald-600 dark:text-emerald-400 border-l-emerald-400";
      case "tool": return "text-orange-600 dark:text-orange-400 border-l-orange-400";
      default: return "text-gray-600 dark:text-gray-400 border-l-gray-400";
    }
  };

  if (trace.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
        Full context will appear here
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-2 border-b bg-muted/30 sticky top-0 z-10">
        <span className="text-xs font-medium text-muted-foreground">
          Full Context Log &middot; {trace.length} turns
        </span>
      </div>
      <div className="flex flex-col">
        {trace.map((record) => {
          const isActive = record.id === activeTurn;
          const isTurnExpanded = expandedTurns.has(record.id);
          const totalChars = record.messages.reduce((s, m) => s + m.char_count, 0);
          const respChars = record.response.content?.length ?? 0;

          return (
            <div key={record.id} ref={isActive ? activeRef : undefined}>
              {/* Turn header - always visible */}
              <button
                onClick={() => toggleTurn(record.id)}
                className={`flex w-full items-center gap-2 px-3 py-1.5 text-xs font-medium border-b hover:bg-muted/50 transition-colors ${
                  isActive ? "bg-primary/8 text-primary" : "bg-muted/30 text-muted-foreground"
                }`}
              >
                <span className="text-[10px]">{isTurnExpanded ? "▼" : "▶"}</span>
                <span>Turn #{record.iteration}</span>
                <span className="font-mono text-[10px] opacity-70">{record.model}</span>
                <span className="text-[10px] opacity-70">
                  {record.messages_count} msgs
                </span>
                <span className="text-[10px] opacity-70 font-mono">
                  {(totalChars / 1000).toFixed(1)}k→{(respChars / 1000).toFixed(1)}k
                </span>
                <span className="text-[10px] opacity-50 ml-auto">
                  {new Date(record.timestamp).toLocaleTimeString()}
                </span>
              </button>

              {/* Messages - shown only when turn is expanded */}
              {isTurnExpanded && (
                <div className="flex flex-col">
                  {record.messages.map((msg, i) => {
                    const itemKey = `${record.id}_msg_${i}`;
                    const isExpanded = expandedItems.has(itemKey);
                    return (
                      <div key={itemKey}>
                        <button
                          onClick={() => toggleItem(itemKey)}
                          className={`flex w-full items-center gap-1.5 px-3 py-0.5 text-xs text-left border-l-2 hover:bg-muted/30 transition-colors ${
                            roleColor(msg.role)
                          }`}
                        >
                          <span className="font-medium w-12 shrink-0 text-[10px]">
                            {msg.role}
                          </span>
                          {msg.tool_name && (
                            <span className="font-mono text-[10px] shrink-0 opacity-70">[{msg.tool_name}]</span>
                          )}
                          {msg.tool_calls && msg.tool_calls.length > 0 && (
                            <span className="font-mono text-[10px] shrink-0 opacity-70">
                              [{msg.tool_calls.map(tc => tc.name).join(",")}]
                            </span>
                          )}
                          <span className="truncate flex-1 text-muted-foreground">{msg.preview}</span>
                          <span className="font-mono text-[10px] text-muted-foreground shrink-0">
                            {msg.char_count >= 1000 ? `${(msg.char_count / 1000).toFixed(1)}k` : msg.char_count}
                          </span>
                        </button>
                        {isExpanded && (
                          <div className={`border-l-2 mx-3 mb-1 p-2 bg-muted/20 rounded-md max-h-60 overflow-auto ${roleColor(msg.role)}`}>
                            <pre className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/80">
                              {msg.content}
                            </pre>
                            {msg.tool_calls && msg.tool_calls.length > 0 && (
                              <div className="mt-2 pt-2 border-t border-border/50">
                                <span className="text-[10px] text-muted-foreground">Tool calls:</span>
                                {msg.tool_calls.map((tc, j) => (
                                  <pre key={j} className="text-[10px] whitespace-pre-wrap break-words font-mono text-foreground/60 mt-0.5">
                                    {tc.name}({tc.arguments?.slice(0, 500)}{(tc.arguments?.length ?? 0) > 500 ? "..." : ""})
                                  </pre>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}

                  {/* Model response as last entry */}
                  {record.response.content && (() => {
                    const respKey = `${record.id}_response`;
                    const isRespExpanded = expandedItems.has(respKey);
                    return (
                      <div>
                        <button
                          onClick={() => toggleItem(respKey)}
                          className={`flex w-full items-center gap-1.5 px-3 py-0.5 text-xs text-left border-l-2 hover:bg-muted/30 transition-colors ${roleColor("assistant_response")}`}
                        >
                          <span className="font-medium w-12 shrink-0 text-[10px] text-emerald-600 dark:text-emerald-400">
                            resp
                          </span>
                          {record.response.tool_calls && record.response.tool_calls.length > 0 && (
                            <span className="font-mono text-[10px] shrink-0 opacity-70">
                              [{record.response.tool_calls.map(tc => tc.name).join(",")}]
                            </span>
                          )}
                          <span className="truncate flex-1 text-muted-foreground">
                            {record.response.content.slice(0, 150)}
                          </span>
                          <span className="font-mono text-[10px] text-muted-foreground shrink-0">
                            {record.response.content.length >= 1000
                              ? `${(record.response.content.length / 1000).toFixed(1)}k`
                              : record.response.content.length}
                          </span>
                        </button>
                        {isRespExpanded && (
                          <div className={`border-l-2 mx-3 mb-1 p-2 bg-muted/20 rounded-md max-h-60 overflow-auto ${roleColor("assistant_response")}`}>
                            <pre className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/80">
                              {record.response.content}
                            </pre>
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
