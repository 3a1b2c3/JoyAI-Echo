import type { PromptStackSession } from "@/hooks/usePromptStack";

interface Props {
  sessions: PromptStackSession[];
  activeSession: string | null;
  onSelect: (id: string) => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

export function SessionList({ sessions, activeSession, onSelect }: Props) {
  if (sessions.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-4 text-sm text-muted-foreground">
        No traces yet. Enable promptStacker in config and send messages.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1 p-2">
      {sessions.map((s) => (
        <button
          key={s.id}
          onClick={() => onSelect(s.id)}
          className={`flex flex-col items-start gap-0.5 rounded-md px-3 py-2 text-left text-xs transition-colors ${
            activeSession === s.id
              ? "bg-primary/10 text-primary"
              : "hover:bg-muted"
          }`}
        >
          <span className="font-medium truncate w-full">{s.last_session_key || s.id}</span>
          <span className="text-muted-foreground">
            {s.turn_count} turns &middot; {formatBytes(s.size_bytes)} &middot; {s.last_model}
          </span>
        </button>
      ))}
    </div>
  );
}
