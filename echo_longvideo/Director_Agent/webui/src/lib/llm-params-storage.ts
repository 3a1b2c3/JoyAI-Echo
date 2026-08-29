/**
 * LLM sampling parameters stored in localStorage (browser-scoped, not session).
 * Read at request time for WebSocket messages.
 */
import type { GenerationLlmSettings } from "@/lib/api";

export const LLM_PARAMS_STORAGE_KEY = "nanobot.llm-params";

const EMPTY: GenerationLlmSettings = {
  temperature: null,
  top_p: null,
  top_k: null,
};

function normalizeTemperature(value: unknown): number | null {
  try {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return null;
    return parsed >= 0 && parsed < 2 ? parsed : null;
  } catch {
    return null;
  }
}

function normalizeTopP(value: unknown): number | null {
  try {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return null;
    return parsed >= 0 && parsed <= 1 ? parsed : null;
  } catch {
    return null;
  }
}

function normalizeTopK(value: unknown): number | null {
  try {
    const parsed = Number.parseInt(String(value), 10);
    if (!Number.isFinite(parsed)) return null;
    return parsed >= 1 && parsed <= 64 ? parsed : null;
  } catch {
    return null;
  }
}

function normalizeStored(raw: unknown): GenerationLlmSettings {
  if (!raw || typeof raw !== "object") return { ...EMPTY };
  const obj = raw as Record<string, unknown>;
  return {
    temperature:
      obj.temperature == null || obj.temperature === ""
        ? null
        : normalizeTemperature(obj.temperature),
    top_p:
      obj.top_p == null || obj.top_p === "" ? null : normalizeTopP(obj.top_p),
    top_k: obj.top_k == null || obj.top_k === "" ? null : normalizeTopK(obj.top_k),
  };
}

/** Read validated params from localStorage; null fields mean gateway defaults. */
export function readLlmParams(): GenerationLlmSettings {
  if (typeof window === "undefined") return { ...EMPTY };
  try {
    const raw = window.localStorage.getItem(LLM_PARAMS_STORAGE_KEY);
    if (!raw) return { ...EMPTY };
    return normalizeStored(JSON.parse(raw));
  } catch {
    return { ...EMPTY };
  }
}

/** Merge partial settings and persist to localStorage. */
export function writeLlmParams(partial: Partial<GenerationLlmSettings>): GenerationLlmSettings {
  const current = readLlmParams();
  const next: GenerationLlmSettings = { ...current };

  if ("temperature" in partial) {
    next.temperature =
      partial.temperature == null ? null : normalizeTemperature(partial.temperature);
  }
  if ("top_p" in partial) {
    next.top_p = partial.top_p == null ? null : normalizeTopP(partial.top_p);
  }
  if ("top_k" in partial) {
    next.top_k = partial.top_k == null ? null : normalizeTopK(partial.top_k);
  }

  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(LLM_PARAMS_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // best-effort persistence
    }
  }
  return next;
}

function pickSetFields(
  settings: GenerationLlmSettings,
): Partial<GenerationLlmSettings> {
  const out: Partial<GenerationLlmSettings> = {};
  if (settings.temperature != null) out.temperature = settings.temperature;
  if (settings.top_p != null) out.top_p = settings.top_p;
  if (settings.top_k != null) out.top_k = settings.top_k;
  return out;
}

/** Non-empty fields for HTTP X-Nanobot-Body. */
export function readLlmParamsForRequest(): Partial<GenerationLlmSettings> {
  return pickSetFields(readLlmParams());
}

/** Non-empty fields for WebSocket message envelope. */
export function readLlmParamsForWire(): Partial<GenerationLlmSettings> {
  return pickSetFields(readLlmParams());
}
