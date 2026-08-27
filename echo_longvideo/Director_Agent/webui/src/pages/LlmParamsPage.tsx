import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router";

import type { GenerationLlmSettings } from "@/lib/api";
import { readLlmParams, writeLlmParams } from "@/lib/llm-params-storage";
import { cn } from "@/lib/utils";
import { ROUTES } from "@/router/routes";

type FieldKey = keyof GenerationLlmSettings;

const FIELDS: {
  key: FieldKey;
  label: string;
  min: number;
  max: number;
  step: number;
  hint: string;
}[] = [
  {
    key: "temperature",
    label: "Temperature",
    min: 0,
    max: 1.99,
    step: 0.1,
    hint: "0–2, leave blank for default",
  },
  {
    key: "top_p",
    label: "Top P",
    min: 0,
    max: 1,
    step: 0.05,
    hint: "0–1, leave blank for default",
  },
  {
    key: "top_k",
    label: "Top K",
    min: 1,
    max: 64,
    step: 1,
    hint: "1–64, leave blank for default",
  },
];

function draftFromSettings(s: GenerationLlmSettings): Record<FieldKey, string> {
  return {
    temperature: s.temperature != null ? String(s.temperature) : "",
    top_p: s.top_p != null ? String(s.top_p) : "",
    top_k: s.top_k != null ? String(s.top_k) : "",
  };
}

function draftsEqual(
  a: Record<FieldKey, string>,
  b: Record<FieldKey, string>,
): boolean {
  return (
    a.temperature === b.temperature &&
    a.top_p === b.top_p &&
    a.top_k === b.top_k
  );
}

function settingsEqual(
  a: GenerationLlmSettings,
  b: GenerationLlmSettings,
): boolean {
  return (
    a.temperature === b.temperature &&
    a.top_p === b.top_p &&
    a.top_k === b.top_k
  );
}

export function LlmParamsPage() {
  const [draft, setDraft] = useState(() => draftFromSettings(readLlmParams()));
  const [saved, setSaved] = useState(false);
  const savedTimerRef = useRef<number | null>(null);

  const persist = useCallback(() => {
    const payload: Partial<GenerationLlmSettings> = {};
    for (const { key } of FIELDS) {
      const raw = draft[key].trim();
      payload[key] = raw === "" ? null : Number(raw);
    }
    const storedBefore = readLlmParams();
    const next = writeLlmParams(payload);
    const draftAfter = draftFromSettings(next);
    const draftChanged = !draftsEqual(draft, draftAfter);
    const storageChanged = !settingsEqual(storedBefore, next);

    if (!draftChanged && !storageChanged) {
      return;
    }

    if (draftChanged) {
      setDraft(draftAfter);
    }

    if (!storageChanged) {
      return;
    }

    if (savedTimerRef.current != null) {
      window.clearTimeout(savedTimerRef.current);
    }
    setSaved(true);
    savedTimerRef.current = window.setTimeout(() => {
      savedTimerRef.current = null;
      setSaved(false);
    }, 1500);
  }, [draft]);

  const persistRef = useRef(persist);
  persistRef.current = persist;

  useEffect(() => {
    const timer = window.setTimeout(() => {
      persistRef.current();
    }, 600);
    return () => window.clearTimeout(timer);
  }, [draft]);

  useEffect(() => {
    return () => {
      if (savedTimerRef.current != null) {
        window.clearTimeout(savedTimerRef.current);
      }
    };
  }, []);

  return (
    <div className="flex min-h-screen w-screen flex-col bg-background text-foreground">
      <header className="flex items-center justify-between border-b px-6 py-4">
        <div>
          <h1 className="text-lg font-semibold">LLM Sampling Parameters</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Parameters are stored in this browser and applied to future generation requests.
          </p>
        </div>
        <Link
          to={ROUTES.home}
          className="rounded-md px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          Back to Home
        </Link>
      </header>

      <main className="mx-auto w-full max-w-md flex-1 px-6 py-8">
        <div className="grid gap-4 rounded-xl border border-border/60 bg-card/40 p-4">
          {FIELDS.map(({ key, label, min, max, step, hint }) => (
            <label key={key} className="grid gap-1.5">
              <span className="text-sm font-medium text-foreground">
                {label}
                <span className="ml-1 text-xs font-normal text-muted-foreground">
                  ({hint})
                </span>
              </span>
              <input
                type="number"
                min={min}
                max={max}
                step={step}
                value={draft[key]}
                placeholder="Default"
                onChange={(e) =>
                  setDraft((prev) => ({ ...prev, [key]: e.target.value }))
                }
                onBlur={() => persist()}
                className={cn(
                  "h-9 rounded-lg border border-border/60 bg-background px-3 text-sm tabular-nums outline-none focus:ring-1 focus:ring-foreground/10",
                )}
              />
            </label>
          ))}
          <p className="text-xs text-muted-foreground">
            {saved ? "Saved in this browser" : "Changes save automatically"}
          </p>
        </div>
      </main>
    </div>
  );
}
