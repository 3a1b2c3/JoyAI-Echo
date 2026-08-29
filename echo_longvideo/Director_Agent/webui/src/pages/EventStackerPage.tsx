import { useState, useMemo } from "react";
import { useEventStack, type EventRecord } from "@/hooks/useEventStack";

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const TYPE_STYLE: Record<string, { bg: string; text: string; label: string }> = {
  turn_start:     { bg: "bg-green-500/15",   text: "text-green-700 dark:text-green-400",     label: "START" },
  turn_end:       { bg: "bg-gray-500/15",    text: "text-gray-600 dark:text-gray-400",       label: "END" },
  context_group:  { bg: "bg-blue-500/15",    text: "text-blue-700 dark:text-blue-400",       label: "CONTEXT" },
  governance:     { bg: "bg-amber-500/15",   text: "text-amber-700 dark:text-amber-400",     label: "TRIM" },
  model_request:  { bg: "bg-purple-500/15",  text: "text-purple-700 dark:text-purple-400",   label: "CALL" },
  model_response: { bg: "bg-emerald-500/15", text: "text-emerald-700 dark:text-emerald-400", label: "REPLY" },
  tool_exec:      { bg: "bg-orange-500/15",  text: "text-orange-700 dark:text-orange-400",   label: "TOOL" },
  injection:      { bg: "bg-cyan-500/15",    text: "text-cyan-700 dark:text-cyan-400",       label: "INJECT" },
  retry:          { bg: "bg-red-500/15",     text: "text-red-700 dark:text-red-400",         label: "RETRY" },
  error:              { bg: "bg-red-500/20",     text: "text-red-700 dark:text-red-300",         label: "ERROR" },
  content_transform:  { bg: "bg-yellow-500/15", text: "text-yellow-700 dark:text-yellow-400",   label: "TRANSFORM" },
};
const DEFAULT_STYLE = { bg: "bg-gray-500/15", text: "text-gray-600 dark:text-gray-400", label: "EVT" };
function getStyle(type: string) { return TYPE_STYLE[type] ?? DEFAULT_STYLE; }

const PART_COLORS: Record<string, { bar: string; text: string }> = {
  identity:           { bar: "bg-blue-400",    text: "text-blue-600 dark:text-blue-400" },
  bootstrap:          { bar: "bg-indigo-400",  text: "text-indigo-600 dark:text-indigo-400" },
  memory:             { bar: "bg-purple-400",  text: "text-purple-600 dark:text-purple-400" },
  active_skills:      { bar: "bg-pink-400",    text: "text-pink-600 dark:text-pink-400" },
  skills_summary:     { bar: "bg-rose-400",    text: "text-rose-600 dark:text-rose-400" },
  recent_history:     { bar: "bg-amber-400",   text: "text-amber-600 dark:text-amber-400" },
  history:            { bar: "bg-orange-400",  text: "text-orange-600 dark:text-orange-400" },
  runtime_context:    { bar: "bg-gray-400",    text: "text-gray-600 dark:text-gray-400" },
  user_message:       { bar: "bg-green-400",   text: "text-green-600 dark:text-green-400" },
  system_instruction: { bar: "bg-red-400",     text: "text-red-600 dark:text-red-400" },
};
const DEFAULT_PART = { bar: "bg-slate-400", text: "text-slate-600 dark:text-slate-400" };
function getPartColor(label: string) { return PART_COLORS[label] ?? DEFAULT_PART; }

function formatChars(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

// ---------------------------------------------------------------------------
// Event grouping: merge consecutive context_part events into one group
// ---------------------------------------------------------------------------

interface SingleItem { kind: "single"; event: EventRecord }
interface ContextGroupItem { kind: "context_group"; events: EventRecord[]; totalChars: number; phase: string }
type ProcessedItem = SingleItem | ContextGroupItem;

const PHASE_LABELS: Record<string, string> = {
  turn_context: "Turn Context",
  token_estimation: "Token Estimation (Consolidator)",
};
function phaseLabel(phase: string): string {
  return PHASE_LABELS[phase] || phase;
}

function groupEvents(events: EventRecord[]): ProcessedItem[] {
  const result: ProcessedItem[] = [];
  let buf: EventRecord[] = [];
  let bufPhase = "";

  const flush = () => {
    if (buf.length > 0) {
      const totalChars = buf.reduce((s, e) => s + Number(e.data.char_count || 0), 0);
      result.push({ kind: "context_group", events: [...buf], totalChars, phase: bufPhase });
      buf = [];
      bufPhase = "";
    }
  };

  for (const event of events) {
    if (event.type === "context_part") {
      const phase = String(event.data.phase || "");
      if (buf.length > 0 && phase !== bufPhase) {
        flush();
      }
      if (buf.length === 0) bufPhase = phase;
      buf.push(event);
    } else {
      flush();
      result.push({ kind: "single", event });
    }
  }
  flush();
  return result;
}

// ---------------------------------------------------------------------------
// Context group component — collapsed proportion bar + expandable parts
// ---------------------------------------------------------------------------

function ContextGroupCard({
  group,
  expanded,
  onToggle,
}: {
  group: ContextGroupItem;
  expanded: boolean;
  onToggle: () => void;
}) {
  const [expandedParts, setExpandedParts] = useState<Set<number>>(new Set());
  const style = getStyle("context_group");

  const togglePart = (idx: number) => {
    setExpandedParts((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  return (
    <div>
      {/* Header row */}
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-2 px-2 py-0.5 text-xs text-left rounded hover:bg-muted/30 transition-colors cursor-pointer"
      >
        <span className={`inline-flex items-center justify-center rounded px-1.5 py-0 text-[10px] font-bold shrink-0 w-16 ${style.bg} ${style.text}`}>
          {style.label}
        </span>
        {group.phase && (
          <span className="text-[10px] font-medium text-blue-500 dark:text-blue-400 shrink-0">
            {phaseLabel(group.phase)}
          </span>
        )}
        <span className="text-muted-foreground text-[11px]">
          {group.events.length} parts &middot; {formatChars(group.totalChars)} total
        </span>
        {/* Inline mini proportion bar */}
        <span className="flex-1 flex h-2.5 rounded overflow-hidden bg-muted/40 min-w-[80px] max-w-[260px]">
          {group.events.map((e, i) => {
            const pct = group.totalChars > 0
              ? (Number(e.data.char_count || 0) / group.totalChars) * 100
              : 0;
            const pc = getPartColor(String(e.data.label || ""));
            return (
              <span
                key={i}
                className={`${pc.bar} h-full opacity-70`}
                style={{ width: `${pct}%` }}
                title={`${e.data.label}: ${formatChars(Number(e.data.char_count))} (${pct.toFixed(1)}%)`}
              />
            );
          })}
        </span>
        <span className="text-[10px] text-muted-foreground shrink-0">
          {expanded ? "▼" : "▶"}
        </span>
      </button>

      {/* Expanded: full proportion bar + part list */}
      {expanded && (
        <div className="ml-16 mr-2 mt-1 mb-2 p-2 bg-muted/20 border rounded-md">
          {/* Full proportion bar with labels */}
          <div className="flex h-5 rounded overflow-hidden mb-2">
            {group.events.map((e, i) => {
              const pct = group.totalChars > 0
                ? (Number(e.data.char_count || 0) / group.totalChars) * 100
                : 0;
              const pc = getPartColor(String(e.data.label || ""));
              return (
                <div
                  key={i}
                  className={`${pc.bar} h-full flex items-center justify-center opacity-80 border-r border-background/30 last:border-r-0`}
                  style={{ width: `${pct}%` }}
                  title={`${e.data.label}: ${formatChars(Number(e.data.char_count))} (${pct.toFixed(1)}%)`}
                >
                  {pct > 8 && (
                    <span className="text-[9px] font-bold text-white/90 truncate px-0.5">
                      {String(e.data.label || "").replace(/_/g, " ")}
                    </span>
                  )}
                </div>
              );
            })}
          </div>

          {/* Part list */}
          <div className="flex flex-col gap-0.5">
            {group.events.map((e, i) => {
              const label = String(e.data.label || "");
              const chars = Number(e.data.char_count || 0);
              const pct = group.totalChars > 0 ? (chars / group.totalChars) * 100 : 0;
              const pc = getPartColor(label);
              const isPartExpanded = expandedParts.has(i);

              return (
                <div key={i}>
                  <button
                    onClick={() => togglePart(i)}
                    className="flex w-full items-center gap-2 px-1 py-0.5 text-xs rounded hover:bg-muted/40 text-left"
                  >
                    <span className={`w-2 h-2 rounded-full shrink-0 ${pc.bar}`} />
                    <span className={`font-medium w-28 shrink-0 truncate ${pc.text}`}>
                      {label.replace(/_/g, " ")}
                    </span>
                    <span className="font-mono text-[10px] text-muted-foreground w-12 text-right shrink-0">
                      {formatChars(chars)}
                    </span>
                    <span className="font-mono text-[10px] text-muted-foreground w-12 text-right shrink-0">
                      {pct.toFixed(1)}%
                    </span>
                    {/* Mini bar */}
                    <span className="flex-1 h-1.5 rounded bg-muted/30 overflow-hidden">
                      <span className={`block h-full ${pc.bar} opacity-60`} style={{ width: `${pct}%` }} />
                    </span>
                    <span className="text-[10px] text-muted-foreground">{isPartExpanded ? "▼" : "▶"}</span>
                  </button>
                  {isPartExpanded && (
                    <div className="ml-6 mr-1 mt-0.5 mb-1 p-2 bg-muted/10 border rounded max-h-80 overflow-auto">
                      <pre className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/80">
                        {String(e.data.content || "")}
                      </pre>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Single event summary + detail (non-context events)
// ---------------------------------------------------------------------------

function EventSummary({ event }: { event: EventRecord }) {
  const d = event.data;
  switch (event.type) {
    case "turn_start":
      return <span className="text-muted-foreground">{String(d.model || "")}</span>;
    case "turn_end":
      return (
        <span className="text-muted-foreground">
          {String(d.stop_reason || "completed")}
          {d.usage && typeof d.usage === "object" ? (
            <span className="ml-2 font-mono text-[10px] opacity-70">
              {Object.entries(d.usage as Record<string, number>)
                .map(([k, v]) => `${k}:${v}`)
                .join(" ")}
            </span>
          ) : null}
        </span>
      );
    case "governance":
      return (
        <span className="text-muted-foreground font-mono text-[11px]">
          {String(d.input_messages ?? "?")} msgs &rarr; {String(d.output_messages ?? "?")} msgs
          {d.input_chars != null && d.output_chars != null && (
            <span className="ml-2">
              {formatChars(Number(d.input_chars))} &rarr; {formatChars(Number(d.output_chars))} chars
            </span>
          )}
        </span>
      );
    case "model_request":
      return (
        <span className="text-muted-foreground">
          <span className="font-mono text-[11px]">
            iter #{String(d.iteration ?? 0)} &middot; {String(d.messages_count ?? 0)} msgs &middot; {formatChars(Number(d.total_chars || 0))}
            &middot; {String(d.tools_count ?? "?")} tools
          </span>
          {d.model && <span className="ml-2 text-[10px] opacity-60">{String(d.model)}</span>}
          {d.type && <span className="ml-1 text-[10px] text-yellow-600 dark:text-yellow-400">({String(d.type)})</span>}
        </span>
      );
    case "model_response": {
      const content = String(d.content || "");
      const toolCalls = d.tool_calls as { name: string }[] | undefined;
      const usage = d.usage as Record<string, number> | undefined;
      return (
        <span className="text-muted-foreground">
          {toolCalls && toolCalls.length > 0 && (
            <span className="text-orange-600 dark:text-orange-400 font-mono text-[11px] mr-2">
              [{toolCalls.map(tc => tc.name).join(", ")}]
            </span>
          )}
          <span className="truncate">{content.slice(0, 120)}{content.length > 120 ? "..." : ""}</span>
          {usage && (
            <span className="ml-2 font-mono text-[10px] opacity-60">
              {Object.entries(usage).filter(([,v]) => v > 0).map(([k, v]) => `${k}:${v}`).join(" ")}
            </span>
          )}
        </span>
      );
    }
    case "tool_exec":
      return (
        <span className="text-muted-foreground">
          <span className="font-medium text-orange-600 dark:text-orange-400">{String(d.name || "")}</span>
          <span className="ml-2 font-mono text-[10px]">result: {formatChars(Number(d.result_chars || 0))}</span>
        </span>
      );
    case "injection":
      return (
        <span className="text-muted-foreground">
          {String(d.count || 0)} message(s) &middot; {String(d.phase || "")}
        </span>
      );
    case "retry":
      return (
        <span className="text-muted-foreground">
          {String(d.type || "unknown")} &middot; iter #{String(d.iteration ?? "?")}
        </span>
      );
    case "error":
      return (
        <span className="text-red-600 dark:text-red-400">
          {String(d.type || "Error")}: {String(d.message || "").slice(0, 150)}
        </span>
      );
    case "content_transform":
      return (
        <span className="text-muted-foreground">
          <span className="font-medium text-yellow-600 dark:text-yellow-400">{String(d.stage || "")}</span>
          <span className="ml-2 font-mono text-[11px]">
            {formatChars(Number(d.original_chars || 0))} &rarr; {formatChars(Number(d.transformed_chars || 0))} chars
          </span>
        </span>
      );
    default:
      return <span className="text-muted-foreground font-mono text-[10px]">{JSON.stringify(d).slice(0, 200)}</span>;
  }
}

function EventDetail({ event }: { event: EventRecord }) {
  const d = event.data;
  switch (event.type) {
    case "model_request": {
      const messages = d.messages as Array<Record<string, any>> | undefined;
      if (!messages) return <span className="text-xs text-muted-foreground">No message data</span>;
      return <ModelRequestDetail data={d} messages={messages} />;
    }
    case "model_response":
      return (
        <div className="flex flex-col gap-2">
          {d.reasoning_content && (
            <div>
              <span className="text-[10px] text-muted-foreground font-medium">Reasoning:</span>
              <pre className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/70 mt-0.5">
                {String(d.reasoning_content)}
              </pre>
            </div>
          )}
          <div>
            <span className="text-[10px] text-muted-foreground font-medium">Content:</span>
            <pre className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/80 mt-0.5">
              {String(d.content || "")}
            </pre>
          </div>
          {d.tool_calls && (d.tool_calls as unknown[]).length > 0 && (
            <div>
              <span className="text-[10px] text-muted-foreground font-medium">Tool Calls:</span>
              {(d.tool_calls as Array<{ name: string; arguments: string }>).map((tc, j) => (
                <pre key={j} className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/60 mt-0.5">
                  {tc.name}({tc.arguments})
                </pre>
              ))}
            </div>
          )}
        </div>
      );
    case "tool_exec":
      return (
        <div className="flex flex-col gap-2">
          <div>
            <span className="text-[10px] text-muted-foreground font-medium">Arguments:</span>
            <pre className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/70 mt-0.5 max-h-40 overflow-auto">
              {String(d.arguments || "")}
            </pre>
          </div>
          <div>
            <span className="text-[10px] text-muted-foreground font-medium">Result:</span>
            <pre className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/80 mt-0.5 max-h-60 overflow-auto">
              {String(d.result || "")}
            </pre>
          </div>
        </div>
      );
    case "content_transform":
      return (
        <div className="flex flex-col gap-2">
          <div className="flex gap-4 text-[10px] text-muted-foreground">
            <span>Stage: <span className="font-medium text-yellow-600 dark:text-yellow-400">{String(d.stage || "")}</span></span>
            <span>Original: {formatChars(Number(d.original_chars || 0))}</span>
            <span>Transformed: {formatChars(Number(d.transformed_chars || 0))}</span>
          </div>
          <div>
            <span className="text-[10px] text-muted-foreground font-medium">Transformed content:</span>
            <pre className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/80 mt-0.5 max-h-60 overflow-auto">
              {String(d.transformed || "")}
            </pre>
          </div>
        </div>
      );
    default:
      return (
        <pre className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/80">
          {JSON.stringify(d, null, 2)}
        </pre>
      );
  }
}

function extractContent(msg: Record<string, any>): string {
  const raw = msg.content;
  if (typeof raw === "string") return raw;
  if (Array.isArray(raw)) {
    return raw
      .map((b: any) => {
        if (typeof b === "string") return b;
        if (b?.type === "text") return b.text || "";
        if (b?.type === "image_url") return "[image]";
        return JSON.stringify(b);
      })
      .join("\n");
  }
  return raw ? JSON.stringify(raw) : "";
}

function MessageRow({ msg, idx }: { msg: Record<string, any>; idx: number }) {
  const [expanded, setExpanded] = useState(false);
  const role = String(msg.role || "unknown");
  const content = extractContent(msg);
  const toolCalls = msg.tool_calls as Array<{ id?: string; function?: { name: string; arguments: string }; name?: string; arguments?: string }> | undefined;
  const hasToolCalls = toolCalls && toolCalls.length > 0;
  const toolCallId = msg.tool_call_id as string | undefined;
  const charCount = content.length;
  const roleColor =
    role === "system" ? "text-purple-600 dark:text-purple-400" :
    role === "user" ? "text-blue-600 dark:text-blue-400" :
    role === "assistant" ? "text-green-600 dark:text-green-400" :
    "text-orange-600 dark:text-orange-400";

  const summary = hasToolCalls
    ? toolCalls!.map(tc => tc.function?.name || tc.name || "?").join(", ")
    : content.slice(0, 150);

  return (
    <div className="border rounded overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-2 px-2 py-0.5 text-xs hover:bg-muted/50 text-left"
      >
        <span className="text-[10px] text-muted-foreground w-4">{idx}</span>
        <span className={`font-medium w-12 shrink-0 ${roleColor}`}>{role}</span>
        {msg.name && <span className="text-orange-500 font-mono text-[10px] shrink-0">[{String(msg.name)}]</span>}
        {toolCallId && <span className="text-orange-500 font-mono text-[10px] shrink-0">tool_result</span>}
        {hasToolCalls ? (
          <span className="truncate flex-1">
            <span className="text-orange-600 dark:text-orange-400 font-mono text-[11px]">
              tool_call → [{summary}]
            </span>
          </span>
        ) : (
          <span className="truncate flex-1 text-muted-foreground">{summary}</span>
        )}
        <span className="font-mono text-[10px] text-muted-foreground shrink-0">{formatChars(charCount)}</span>
        <span className="text-[10px] text-muted-foreground">{expanded ? "▼" : "▶"}</span>
      </button>
      {expanded && (
        <div className="border-t bg-muted/20 p-2 max-h-[600px] overflow-auto flex flex-col gap-2">
          {content && (
            <pre className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/80">{content}</pre>
          )}
          {hasToolCalls && toolCalls!.map((tc, j) => {
            const name = tc.function?.name || tc.name || "unknown";
            const args = tc.function?.arguments || tc.arguments || "";
            return (
              <div key={j} className="border rounded p-1.5 bg-orange-500/5">
                <div className="text-[10px] text-orange-600 dark:text-orange-400 font-bold mb-0.5">
                  tool_call: {name}
                  {(tc.id || tc.function) && <span className="ml-2 font-normal text-muted-foreground">{tc.id || ""}</span>}
                </div>
                <pre className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/70">{args}</pre>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Model Request Detail — structured view of LLM input
// ---------------------------------------------------------------------------

interface MessageSection {
  label: string;
  color: string;
  messages: { msg: Record<string, any>; originalIdx: number }[];
  charCount: number;
}

function classifyMessages(messages: Array<Record<string, any>>): MessageSection[] {
  const sections: MessageSection[] = [];
  let i = 0;

  // 1. System prompt — always [0], contains identity+bootstrap+skills+memory merged
  if (i < messages.length && messages[i].role === "system") {
    const c = extractContent(messages[i]);
    sections.push({
      label: "System Prompt",
      color: "text-purple-600 dark:text-purple-400",
      messages: [{ msg: messages[i], originalIdx: 0 }],
      charCount: c.length,
    });
    i++;
  }

  // Scan remaining messages to classify
  const rest: { msg: Record<string, any>; originalIdx: number }[] = [];
  for (let j = i; j < messages.length; j++) {
    rest.push({ msg: messages[j], originalIdx: j });
  }

  // Find boundaries:
  // - Runtime Context = user message starting with "[Runtime Context"
  // - Tool round = assistant(tool_calls) + tool(result) pairs at the tail (iteration > 0 appended)
  // - History = past conversation turns
  // - Current user message = the real user input

  // Separate trailing runtime context
  const runtimeMsgs: typeof rest = [];
  while (
    rest.length > 0 &&
    rest[rest.length - 1].msg.role === "user" &&
    String(rest[rest.length - 1].msg.content || "").startsWith("[Runtime Context")
  ) {
    runtimeMsgs.unshift(rest.pop()!);
  }

  // Separate trailing tool round (from previous iteration: assistant+tool_calls then tool results)
  // These are appended by runner.py when iteration > 0
  const toolRoundMsgs: typeof rest = [];
  // Walk backwards: tool results first, then the assistant with tool_calls
  let cursor = rest.length - 1;
  while (cursor >= 0 && rest[cursor].msg.role === "tool") {
    cursor--;
  }
  if (
    cursor >= 0 &&
    rest[cursor].msg.role === "assistant" &&
    rest[cursor].msg.tool_calls?.length > 0
  ) {
    // Check if these are the tail — there should be user messages before this block
    const hasHistoryBefore = cursor > 0;
    if (hasHistoryBefore) {
      for (let j = cursor; j < rest.length; j++) {
        toolRoundMsgs.push(rest[j]);
      }
      rest.splice(cursor, rest.length - cursor);
    }
  }

  // Now rest = history conversation turns
  // Find the last real user message (not runtime context) as "Current Input"
  const currentMsgs: typeof rest = [];
  if (rest.length > 0 && rest[rest.length - 1].msg.role === "user") {
    currentMsgs.unshift(rest.pop()!);
  }

  // Everything left is History
  const historyMsgs = rest;
  const historyChars = historyMsgs.reduce((s, r) => s + extractContent(r.msg).length, 0);
  const currentChars = currentMsgs.reduce((s, r) => s + extractContent(r.msg).length, 0);
  const toolRoundChars = toolRoundMsgs.reduce((s, r) => s + extractContent(r.msg).length, 0);
  const runtimeChars = runtimeMsgs.reduce((s, r) => s + extractContent(r.msg).length, 0);

  if (historyMsgs.length > 0) {
    sections.push({ label: "History", color: "text-amber-600 dark:text-amber-400", messages: historyMsgs, charCount: historyChars });
  }
  if (currentMsgs.length > 0) {
    sections.push({ label: "User Message", color: "text-green-600 dark:text-green-400", messages: currentMsgs, charCount: currentChars });
  }
  if (toolRoundMsgs.length > 0) {
    sections.push({ label: "Tool Round (prev iteration)", color: "text-orange-600 dark:text-orange-400", messages: toolRoundMsgs, charCount: toolRoundChars });
  }
  if (runtimeMsgs.length > 0) {
    sections.push({ label: "Runtime Context", color: "text-gray-500 dark:text-gray-400", messages: runtimeMsgs, charCount: runtimeChars });
  }

  return sections;
}

function ToolDefRow({ tool }: { tool: Record<string, any> }) {
  const [expanded, setExpanded] = useState(false);
  const fn = tool.function || tool;
  const name = String(fn.name || "");
  const desc = String(fn.description || "");

  return (
    <div className="border rounded overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-2 px-2 py-0.5 text-[10px] hover:bg-muted/40 text-left"
      >
        <span className="font-mono font-bold text-foreground/80 shrink-0">{name}</span>
        <span className="text-muted-foreground truncate flex-1">{desc.slice(0, 80)}</span>
        <span className="text-muted-foreground shrink-0">{expanded ? "▼" : "▶"}</span>
      </button>
      {expanded && (
        <div className="border-t bg-muted/10 p-2 max-h-60 overflow-auto">
          <pre className="text-[10px] whitespace-pre-wrap break-words font-mono text-foreground/80">
            {JSON.stringify(tool, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

function ModelRequestDetail({ data, messages }: { data: Record<string, any>; messages: Array<Record<string, any>> }) {
  const sections = useMemo(() => classifyMessages(messages), [messages]);
  const [collapsedSections, setCollapsedSections] = useState<Set<string>>(() => new Set());
  const [showTools, setShowTools] = useState(false);

  const toggleSection = (label: string) => {
    setCollapsedSections((prev) => {
      const next = new Set(prev);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      return next;
    });
  };

  const totalChars = sections.reduce((s, sec) => s + sec.charCount, 0);
  const tools = data.tools as Array<{ name: string; description: string }> | undefined;
  const toolsCount = Number(data.tools_count ?? tools?.length ?? 0);

  return (
    <div className="flex flex-col gap-2">
      {/* Model params bar */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px] text-muted-foreground border-b pb-1.5">
        <span className="font-medium text-foreground text-xs">LLM Input</span>
        <span>{messages.length} messages</span>
        <span>{formatChars(totalChars)} chars</span>
        {data.temperature != null && <span>temp: {String(data.temperature)}</span>}
        {data.max_tokens != null && <span>max_tokens: {String(data.max_tokens)}</span>}
        {data.reasoning_effort != null && <span>reasoning: {String(data.reasoning_effort)}</span>}
        <button
          onClick={() => setShowTools(!showTools)}
          className="text-blue-500 hover:underline cursor-pointer"
        >
          {toolsCount} tools {showTools ? "▼" : "▶"}
        </button>
        {/* Proportion bar */}
        <span className="flex h-3 rounded overflow-hidden bg-muted/30 min-w-[120px] max-w-[300px] flex-1">
          {sections.map((sec) => {
            const pct = totalChars > 0 ? (sec.charCount / totalChars) * 100 : 0;
            const barColor =
              sec.label === "System Prompt" ? "bg-purple-400" :
              sec.label === "History" ? "bg-amber-400" :
              sec.label === "User Message" ? "bg-green-400" :
              sec.label.startsWith("Tool Round") ? "bg-orange-400" :
              "bg-gray-400";
            return (
              <span
                key={sec.label}
                className={`${barColor} h-full opacity-70`}
                style={{ width: `${pct}%` }}
                title={`${sec.label}: ${formatChars(sec.charCount)} (${pct.toFixed(1)}%)`}
              />
            );
          })}
        </span>
      </div>

      {/* Tools list (collapsible) */}
      {showTools && tools && tools.length > 0 && (
        <div className="border rounded p-2 bg-blue-500/5">
          <div className="text-[10px] font-medium text-blue-600 dark:text-blue-400 mb-1">
            Available Tools ({tools.length})
          </div>
          <div className="flex flex-col gap-0.5">
            {tools.map((t: any, j: number) => (
              <ToolDefRow key={j} tool={t} />
            ))}
          </div>
        </div>
      )}

      {/* Message sections */}
      {sections.map((sec) => {
        const isCollapsed = collapsedSections.has(sec.label);
        return (
          <div key={sec.label} className="border rounded">
            <button
              onClick={() => toggleSection(sec.label)}
              className="flex w-full items-center gap-2 px-2 py-1 text-xs hover:bg-muted/40 text-left"
            >
              <span className="text-[10px]">{isCollapsed ? "▶" : "▼"}</span>
              <span className={`font-bold ${sec.color}`}>{sec.label}</span>
              <span className="text-muted-foreground text-[10px]">
                {sec.messages.length} msg{sec.messages.length > 1 ? "s" : ""}
              </span>
              <span className="font-mono text-[10px] text-muted-foreground">{formatChars(sec.charCount)}</span>
              <span className="font-mono text-[10px] text-muted-foreground">
                ({totalChars > 0 ? ((sec.charCount / totalChars) * 100).toFixed(1) : 0}%)
              </span>
            </button>
            {!isCollapsed && (
              <div className="border-t px-1 py-0.5 flex flex-col gap-0.5">
                {sec.messages.map(({ msg, originalIdx }) => (
                  <MessageRow key={originalIdx} msg={msg} idx={originalIdx} />
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function hasExpandableContent(type: string): boolean {
  return ["model_request", "model_response", "tool_exec", "governance", "injection", "content_transform"].includes(type);
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function EventStackerPage() {
  const { sessions, activeSession, setActiveSession, events, loading, refresh } = useEventStack();
  const [expandedEvents, setExpandedEvents] = useState<Set<number>>(new Set());
  const [expandedContextGroups, setExpandedContextGroups] = useState<Set<string>>(new Set());
  const [collapsedTurns, setCollapsedTurns] = useState<Set<string>>(new Set());

  const turnGroups = useMemo(() => {
    const groups: { turnId: string; items: ProcessedItem[]; model: string; timestamp: string; eventCount: number }[] = [];
    // First pass: group by turn
    const rawGroups: { turnId: string; events: EventRecord[]; model: string; timestamp: string }[] = [];
    let current: (typeof rawGroups)[0] | null = null;
    for (const event of events) {
      if (!current || current.turnId !== event.turn_id) {
        current = {
          turnId: event.turn_id,
          events: [],
          model: event.type === "turn_start" ? String(event.data.model || "") : "",
          timestamp: event.timestamp,
        };
        rawGroups.push(current);
      }
      current.events.push(event);
      if (event.type === "turn_start" && !current.model) {
        current.model = String(event.data.model || "");
      }
    }
    // Second pass: group context_part events within each turn
    for (const rg of rawGroups) {
      groups.push({
        turnId: rg.turnId,
        items: groupEvents(rg.events),
        model: rg.model,
        timestamp: rg.timestamp,
        eventCount: rg.events.length,
      });
    }
    return groups;
  }, [events]);

  const toggleEvent = (seq: number) => {
    setExpandedEvents((prev) => {
      const next = new Set(prev);
      if (next.has(seq)) next.delete(seq);
      else next.add(seq);
      return next;
    });
  };

  const toggleContextGroup = (key: string) => {
    setExpandedContextGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const toggleTurn = (turnId: string) => {
    setCollapsedTurns((prev) => {
      const next = new Set(prev);
      if (next.has(turnId)) next.delete(turnId);
      else next.add(turnId);
      return next;
    });
  };

  return (
    <div className="flex h-screen w-screen bg-background text-foreground">
      {/* Sidebar */}
      <aside className="w-64 shrink-0 border-r overflow-y-auto flex flex-col">
        <div className="flex items-center justify-between px-3 py-3 border-b">
          <h1 className="text-sm font-semibold">Event Stacker</h1>
          <div className="flex gap-1">
            <button
              onClick={refresh}
              className="rounded-md px-2 py-1 text-xs hover:bg-muted transition-colors"
            >
              Refresh
            </button>
            <a href="/" className="rounded-md px-2 py-1 text-xs hover:bg-muted transition-colors">
              Chat
            </a>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto">
          {sessions.length === 0 ? (
            <div className="p-4 text-xs text-muted-foreground text-center">No sessions</div>
          ) : (
            sessions.map((s) => (
              <button
                key={s.id}
                onClick={() => setActiveSession(s.id)}
                className={`w-full text-left px-3 py-2 border-b text-xs hover:bg-muted/50 transition-colors ${
                  activeSession === s.id ? "bg-primary/8 border-l-2 border-l-primary" : ""
                }`}
              >
                <div className="font-medium truncate">{s.id}</div>
                <div className="text-[10px] text-muted-foreground mt-0.5 flex gap-2">
                  <span>{s.event_count} events</span>
                  <span>{(s.size_bytes / 1024).toFixed(1)}KB</span>
                </div>
                <div className="text-[10px] text-muted-foreground truncate">
                  {new Date(s.modified).toLocaleString()}
                </div>
              </button>
            ))
          )}
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {loading ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            Loading events...
          </div>
        ) : events.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            {activeSession ? "No events in this session" : "Select a session from the sidebar"}
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto">
            {turnGroups.map((group) => {
              const isCollapsed = collapsedTurns.has(group.turnId);
              return (
                <div key={group.turnId} className="border-b">
                  {/* Turn header */}
                  <button
                    onClick={() => toggleTurn(group.turnId)}
                    className="flex w-full items-center gap-2 px-4 py-2 text-xs font-medium bg-muted/40 hover:bg-muted/60 transition-colors sticky top-0 z-10"
                  >
                    <span className="text-[10px]">{isCollapsed ? "▶" : "▼"}</span>
                    <span className="text-foreground">{group.turnId.replace("_", " #")}</span>
                    <span className="font-mono text-[10px] text-muted-foreground">{group.model}</span>
                    <span className="text-[10px] text-muted-foreground">{group.eventCount} events</span>
                    <span className="text-[10px] text-muted-foreground ml-auto">
                      {new Date(group.timestamp).toLocaleTimeString()}
                    </span>
                  </button>

                  {/* Events timeline */}
                  {!isCollapsed && (
                    <div className="relative pl-8 pr-4 py-1">
                      <div className="absolute left-5 top-0 bottom-0 w-px bg-border" />

                      {group.items.map((item, itemIdx) => {
                        if (item.kind === "context_group") {
                          const cgKey = `${group.turnId}_cg_${itemIdx}`;
                          return (
                            <div key={cgKey} className="relative py-0.5">
                              <div className="absolute -left-3 top-1.5 w-2 h-2 rounded-full border-2 border-background bg-blue-500/15 ring-1 ring-border" />
                              <div className="ml-2">
                                <ContextGroupCard
                                  group={item}
                                  expanded={expandedContextGroups.has(cgKey)}
                                  onToggle={() => toggleContextGroup(cgKey)}
                                />
                              </div>
                            </div>
                          );
                        }

                        const event = item.event;
                        const style = getStyle(event.type);
                        const isExpanded = expandedEvents.has(event.seq);
                        const canExpand = hasExpandableContent(event.type);

                        return (
                          <div key={event.seq} className="relative py-0.5">
                            <div className={`absolute -left-3 top-1.5 w-2 h-2 rounded-full border-2 border-background ${style.bg} ring-1 ring-border`} />
                            <div className="ml-2">
                              <button
                                onClick={() => canExpand && toggleEvent(event.seq)}
                                className={`flex w-full items-center gap-2 px-2 py-0.5 text-xs text-left rounded hover:bg-muted/30 transition-colors ${
                                  canExpand ? "cursor-pointer" : "cursor-default"
                                }`}
                              >
                                <span className={`inline-flex items-center justify-center rounded px-1.5 py-0 text-[10px] font-bold shrink-0 w-16 ${style.bg} ${style.text}`}>
                                  {style.label}
                                </span>
                                <span className="flex-1 min-w-0 flex items-center gap-1 overflow-hidden">
                                  <EventSummary event={event} />
                                </span>
                                <span className="text-[10px] text-muted-foreground/50 shrink-0 font-mono">
                                  {new Date(event.timestamp).toLocaleTimeString(undefined, { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3 } as Intl.DateTimeFormatOptions)}
                                </span>
                                {canExpand && (
                                  <span className="text-[10px] text-muted-foreground shrink-0">
                                    {isExpanded ? "▼" : "▶"}
                                  </span>
                                )}
                              </button>

                              {isExpanded && (
                                <div className="ml-16 mr-2 mt-1 mb-2 p-2 bg-muted/20 border rounded-md max-h-96 overflow-auto">
                                  <EventDetail event={event} />
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}
