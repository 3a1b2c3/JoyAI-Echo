import type { MemorySelection, WorkplaceData } from "@/lib/types";

function selectionKey(selection: MemorySelection): string {
  return `${selection.memory_id}|${selection.source_shot_id}|${selection.frame_index}`;
}

/**
 * Collect every confirmed Memory selection across approved shot reviews.
 * Falls back to the canonical `memory_bank` snapshot when no approved reviews exist.
 */
export function collectConfirmedMemoryEntries(
  workplace: WorkplaceData | null | undefined,
): MemorySelection[] {
  if (!workplace) return [];

  const seen = new Set<string>();
  const collected: MemorySelection[] = [];

  for (const shot of workplace.shots ?? []) {
    const review = shot.memory_review;
    if (!review || review.status !== "approved") continue;

    for (const selection of review.selections ?? []) {
      if (!selection?.image?.url) continue;
      const key = selectionKey(selection);
      if (seen.has(key)) continue;
      seen.add(key);
      collected.push(selection);
    }
  }

  if (collected.length === 0) {
    return (workplace.memory_bank ?? []).filter((entry) =>
      Boolean(entry?.image?.url),
    );
  }

  collected.sort((a, b) => {
    if (a.source_shot_id !== b.source_shot_id) {
      return a.source_shot_id - b.source_shot_id;
    }
    return a.memory_id.localeCompare(b.memory_id);
  });

  return collected;
}
