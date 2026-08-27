import { useCallback, useEffect, useRef, useState } from "react";

import type { GenerationAlertVariant } from "@/components/ui/GenerationAlertDialog";
import { GENERATION_BUSY_ALERT_ENABLED } from "@/config/features";
import type { WorkplaceData } from "@/lib/types";

export const GENERATION_CONGESTED_MS = 60_000;

const DONE_STATUSES = new Set(["generated", "review_pass", "approved"]);

function hasVersionId(shot: { version_id?: string }): boolean {
  return Boolean(shot.version_id?.trim());
}

function isQueuedSubmitted(shot: {
  status: string;
  version_id?: string;
}): boolean {
  return shot.status === "queued" && hasVersionId(shot);
}

function isShotDone(shot: {
  status: string;
  video?: { url?: string } | null;
  has_video?: boolean;
}): boolean {
  if (!DONE_STATUSES.has(shot.status)) return false;
  return Boolean(shot.video?.url || shot.has_video);
}

export type ShotGenerationAlerts = {
  congestedOpen: boolean;
  errorOpen: boolean;
  /** Which dialog to show when either is open; error takes precedence. */
  activeVariant: GenerationAlertVariant | null;
  dismissCongested: () => void;
  dismissError: () => void;
  dismissActive: () => void;
};

/**
 * Watches workplace shots for R2V-submitted queued jobs and generation errors.
 * - queued + version_id for ≥1min → congested alert (once per queued cycle)
 * - status === "error" (or workplace.stage === "failed") → error alert
 */
export function useShotGenerationAlerts(
  workplace: WorkplaceData | null,
  sessionKey: string | null,
): ShotGenerationAlerts {
  const [congestedOpen, setCongestedOpen] = useState(false);
  const [errorOpen, setErrorOpen] = useState(false);

  /** shot_id → local Date.now() when queued+version_id was first seen */
  const queuedSinceRef = useRef<Map<number, number>>(new Map());
  /** shot_ids that already showed congested for the current queued cycle */
  const congestedShownRef = useRef<Set<number>>(new Set());
  /** shot_ids that already showed error for the current error cycle */
  const errorShownRef = useRef<Set<number>>(new Set());
  const stageFailedShownRef = useRef(false);
  const timersRef = useRef<Map<number, number>>(new Map());

  const clearTimer = useCallback((shotId: number) => {
    const id = timersRef.current.get(shotId);
    if (id != null) {
      window.clearTimeout(id);
      timersRef.current.delete(shotId);
    }
  }, []);

  const clearAllTimers = useCallback(() => {
    for (const id of timersRef.current.values()) {
      window.clearTimeout(id);
    }
    timersRef.current.clear();
  }, []);

  const resetAll = useCallback(() => {
    clearAllTimers();
    queuedSinceRef.current.clear();
    congestedShownRef.current.clear();
    errorShownRef.current.clear();
    stageFailedShownRef.current = false;
    setCongestedOpen(false);
    setErrorOpen(false);
  }, [clearAllTimers]);

  useEffect(() => {
    resetAll();
  }, [sessionKey, resetAll]);

  useEffect(() => {
    if (!workplace || !sessionKey) return;

    const shots = workplace.shots ?? [];
    const activeQueuedIds = new Set<number>();

    for (const shot of shots) {
      const shotId = shot.shot_id;

      if (shot.status === "error") {
        clearTimer(shotId);
        queuedSinceRef.current.delete(shotId);
        congestedShownRef.current.delete(shotId);
        if (!errorShownRef.current.has(shotId)) {
          errorShownRef.current.add(shotId);
          setErrorOpen(true);
          setCongestedOpen(false);
        }
        continue;
      }

      // Left error state → allow future error alerts for this shot.
      errorShownRef.current.delete(shotId);

      if (isShotDone(shot) || shot.status !== "queued") {
        clearTimer(shotId);
        queuedSinceRef.current.delete(shotId);
        congestedShownRef.current.delete(shotId);
        if (isShotDone(shot)) {
          setCongestedOpen(false);
        }
        continue;
      }

      if (!GENERATION_BUSY_ALERT_ENABLED || !isQueuedSubmitted(shot)) {
        // queued without version_id yet (e.g. still recaptioning) — wait.
        continue;
      }

      activeQueuedIds.add(shotId);

      if (!queuedSinceRef.current.has(shotId)) {
        queuedSinceRef.current.set(shotId, Date.now());
      }

      if (congestedShownRef.current.has(shotId)) continue;
      if (timersRef.current.has(shotId)) continue;

      const since = queuedSinceRef.current.get(shotId)!;
      const remaining = Math.max(0, GENERATION_CONGESTED_MS - (Date.now() - since));

      const timerId = window.setTimeout(() => {
        timersRef.current.delete(shotId);
        // Re-check: only fire if still tracked as queued submitted.
        if (!queuedSinceRef.current.has(shotId)) return;
        if (congestedShownRef.current.has(shotId)) return;
        congestedShownRef.current.add(shotId);
        setCongestedOpen(true);
      }, remaining);

      timersRef.current.set(shotId, timerId);
    }

    // Drop timers for shots no longer in the workplace list.
    for (const shotId of [...timersRef.current.keys()]) {
      if (!activeQueuedIds.has(shotId) && !shots.some((s) => s.shot_id === shotId)) {
        clearTimer(shotId);
        queuedSinceRef.current.delete(shotId);
      }
    }

    if (workplace.stage === "failed") {
      if (!stageFailedShownRef.current) {
        stageFailedShownRef.current = true;
        setErrorOpen(true);
        setCongestedOpen(false);
      }
    } else {
      stageFailedShownRef.current = false;
    }
  }, [workplace, sessionKey, clearTimer]);

  useEffect(() => {
    return () => {
      clearAllTimers();
    };
  }, [clearAllTimers]);

  const dismissCongested = useCallback(() => {
    setCongestedOpen(false);
  }, []);

  const dismissError = useCallback(() => {
    setErrorOpen(false);
  }, []);

  const dismissActive = useCallback(() => {
    if (errorOpen) {
      setErrorOpen(false);
      return;
    }
    setCongestedOpen(false);
  }, [errorOpen]);

  const activeVariant: GenerationAlertVariant | null = errorOpen
    ? "error"
    : congestedOpen
      ? "congested"
      : null;

  return {
    congestedOpen,
    errorOpen,
    activeVariant,
    dismissCongested,
    dismissError,
    dismissActive,
  };
}
