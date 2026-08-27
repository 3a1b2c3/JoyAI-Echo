import { useState } from "react";
import type { PromptPart } from "@/hooks/usePromptStack";

interface Props {
  parts: PromptPart[];
}

const COLORS = [
  "bg-blue-500",
  "bg-green-500",
  "bg-yellow-500",
  "bg-purple-500",
  "bg-pink-500",
  "bg-orange-500",
  "bg-teal-500",
  "bg-red-500",
];

export function PartStack({ parts }: Props) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const totalChars = parts.reduce((sum, p) => sum + p.char_count, 0);

  const toggle = (idx: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  return (
    <div className="flex flex-col gap-2">
      {/* Proportion bar */}
      <div className="flex h-6 w-full overflow-hidden rounded-md">
        {parts.map((part, i) => {
          const pct = totalChars > 0 ? (part.char_count / totalChars) * 100 : 0;
          if (pct < 0.5) return null;
          return (
            <div
              key={i}
              className={`${COLORS[i % COLORS.length]} opacity-70 flex items-center justify-center text-[9px] text-white font-medium overflow-hidden`}
              style={{ width: `${pct}%` }}
              title={`${part.label}: ${part.char_count} chars (${pct.toFixed(1)}%)`}
            >
              {pct > 8 ? part.label : ""}
            </div>
          );
        })}
      </div>

      {/* Part cards */}
      <div className="flex flex-col gap-1">
        {parts.map((part, i) => (
          <div key={i} className="border rounded-md overflow-hidden">
            <button
              onClick={() => toggle(i)}
              className="flex w-full items-center justify-between px-3 py-1.5 text-xs hover:bg-muted/50 transition-colors"
            >
              <div className="flex items-center gap-2">
                <span
                  className={`inline-block h-2.5 w-2.5 rounded-sm ${COLORS[i % COLORS.length]} opacity-70`}
                />
                <span className="font-medium">{part.label}</span>
              </div>
              <span className="text-muted-foreground font-mono">
                {part.char_count >= 1000
                  ? `${(part.char_count / 1000).toFixed(1)}k`
                  : part.char_count}
              </span>
            </button>
            {expanded.has(i) && (
              <div className="border-t bg-muted/30 p-3 max-h-80 overflow-auto">
                <pre className="text-xs whitespace-pre-wrap break-words font-mono text-foreground/80">
                  {part.content}
                </pre>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
