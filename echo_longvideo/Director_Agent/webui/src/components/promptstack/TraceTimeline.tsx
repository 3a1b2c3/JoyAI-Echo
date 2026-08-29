import type { TraceRecord } from "@/hooks/usePromptStack";

interface Props {
  trace: TraceRecord[];
  activeTurn: string | null;
  onSelect: (id: string) => void;
}

export function TraceTimeline({ trace, activeTurn, onSelect }: Props) {
  if (trace.length === 0) {
    return (
      <div className="flex items-center justify-center p-4 text-sm text-muted-foreground">
        Select a session to view traces
      </div>
    );
  }

  return (
    <div className="flex gap-1 overflow-x-auto p-2 border-b">
      {trace.map((record) => {
        const totalChars = record.parts.reduce((sum, p) => sum + p.char_count, 0);
        const usage = record.response.usage;
        return (
          <button
            key={record.id}
            onClick={() => onSelect(record.id)}
            className={`flex flex-col items-center gap-0.5 rounded-md px-3 py-2 text-xs shrink-0 transition-colors ${
              activeTurn === record.id
                ? "bg-primary/10 text-primary ring-1 ring-primary/30"
                : "hover:bg-muted"
            }`}
          >
            <span className="font-mono font-medium">#{record.iteration}</span>
            <span className="text-muted-foreground">{(totalChars / 1000).toFixed(1)}k chars</span>
            {usage && (
              <span className="text-muted-foreground">
                {usage.prompt_tokens ?? 0}+{usage.completion_tokens ?? 0}t
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
