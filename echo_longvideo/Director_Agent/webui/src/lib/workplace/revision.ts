import type { StoryProfileBeat, WorkplaceData } from "@/lib/types";

export function resolveBeats(workplace: WorkplaceData): StoryProfileBeat[] {
  const beats = workplace.story_profile?.beats;
  return Array.isArray(beats) ? beats : [];
}

export function beatSummaryForShot(
  workplace: WorkplaceData,
  shotId: number,
): string {
  const beat = resolveBeats(workplace).find((item) => item.shot_id === shotId);
  return beat?.summary?.trim() ?? "";
}

export function buildBeatRevisionFeedback(
  shotId: number,
  oldSummary: string,
  newSummary: string,
): string {
  return [
    `用户修改了镜头「${shotId}」的分镜 summary。`,
    `原内容：「${oldSummary}」。`,
    `修改后：「${newSummary}」。`,
    "请同步更新 story_profile.beats 中对应 beat 的 summary、重写该镜头提示词，并重新生成该镜头。",
  ].join("\n");
}
