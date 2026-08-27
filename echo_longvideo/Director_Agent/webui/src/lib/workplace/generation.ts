import type { WorkplaceData, WorkplaceShot } from "@/lib/types";

const GENERATED_STATUSES = new Set([
  "generated",
  "review_pass",
  "approved",
]);

export function shotReferenceIds(shot: WorkplaceShot): number[] {
  return shot.planned_reference_shot_ids ?? shot.reference_shot_ids ?? [];
}

export function isShotGenerated(shot: WorkplaceShot): boolean {
  if (shot.has_video || !!shot.video?.url) return true;
  return GENERATED_STATUSES.has(shot.status);
}

/** Hide/edit-lock future-shot Memory until the Agent has produced its draft. */
export function isMemoryRecommendationReady(
  workplace: WorkplaceData,
  shot: WorkplaceShot,
): boolean {
  if (shot.shot_id <= 1 || workplace.auto_generate || isShotGenerated(shot)) {
    return true;
  }
  const memoryWorkflowActive = workplace.shots.some(
    (candidate) => candidate.memory_review != null,
  );
  if (!memoryWorkflowActive) return true;
  return (
    shot.memory_slots_configured === true ||
    shot.memory_recommendation_source === "agent"
  );
}

export function missingReferenceGenerations(
  workplace: WorkplaceData,
  shot: WorkplaceShot,
): number[] {
  const refs = shotReferenceIds(shot);
  if (refs.length === 0) return [];
  const byId = new Map(workplace.shots.map((item) => [item.shot_id, item]));
  return refs.filter((refId) => {
    const refShot = byId.get(refId);
    return !refShot || !isShotGenerated(refShot);
  });
}

export function previousShotApprovalMessage(
  workplace: WorkplaceData,
  shot: WorkplaceShot,
): string | null {
  if (shot.shot_id <= 1) return null;
  const previous = [...workplace.shots]
    .filter((item) => item.shot_id < shot.shot_id)
    .sort((left, right) => right.shot_id - left.shot_id)[0];
  const previousId = previous?.shot_id ?? shot.shot_id - 1;
  const videoAccepted =
    previous?.accepted === true && previous.status === "approved";
  const memoryAccepted = previous?.memory_review?.status === "approved";
  if (videoAccepted && memoryAccepted) return null;
  return `Approve Shot ${previousId} and confirm its memory before generating Shot ${shot.shot_id}.`;
}

/** Amber hint shown on Shot 1 until its video + Memory are both confirmed. */
export function shotOneWorkflowHint(
  workplace: WorkplaceData,
): string | null {
  const hasLaterShot = workplace.shots.some((item) => item.shot_id >= 2);
  if (!hasLaterShot) return null;
  const shotOne = workplace.shots.find((item) => item.shot_id === 1);
  if (!shotOne) return null;
  const videoAccepted =
    shotOne.accepted === true && shotOne.status === "approved";
  const memoryAccepted = shotOne.memory_review?.status === "approved";
  if (videoAccepted && memoryAccepted) return null;
  return "Approve Shot 1 and confirm its memory before generating Shot 2.";
}

export function canGenerateShot(
  workplace: WorkplaceData | null,
  shot: WorkplaceShot,
): boolean {
  if (!workplace?.references_ready) return false;
  if (!shot.references_planned) return false;
  if (!isMemoryRecommendationReady(workplace, shot)) return false;
  if (
    shot.shot_id > 1 &&
    workplace.stage === "awaiting_memory_build" &&
    workplace.shots.some((candidate) => candidate.memory_review != null) &&
    shot.memory_slots_configured !== true
  ) {
    return false;
  }
  if (["queued", "generated", "review_pass", "approved"].includes(shot.status)) {
    return false;
  }
  if (previousShotApprovalMessage(workplace, shot)) return false;
  return missingReferenceGenerations(workplace, shot).length === 0;
}

export function referenceDependencyMessage(
  shot: WorkplaceShot,
  missing: number[],
): string {
  const refs = missing.join("、");
  return `Shot ${shot.shot_id} depends on Shot ${refs}. Generate the reference shot first.`;
}

export function formatReferenceLabel(referenceIds: number[]): string {
  if (referenceIds.length === 0) return "No reference shots";
  return `Reference shots: ${referenceIds.join(", ")}`;
}
