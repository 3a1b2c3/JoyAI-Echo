import { MemoryBankBoard } from "@/components/workplace/MemoryBankBoard";
import { ShotMemorySlots } from "@/components/workplace/ShotMemorySlots";
import { careerMemoryReviewPreview } from "@/lib/memory-review-preview";
import type { MemoryWorkspaceAsset, WorkplaceShot } from "@/lib/types";

export function MemoryBankPreviewPage() {
  const entries =
    careerMemoryReviewPreview("approved").memoryReview?.selections ?? [];
  const assets: MemoryWorkspaceAsset[] = entries.map((entry, index) => ({
    asset_id: `preview-${index}`,
    display_name: entry.display_name ?? entry.memory_id,
    source: "automatic",
    kind: entry.kind,
    memory_id: entry.memory_id,
    source_shot_id: entry.source_shot_id,
    frame_index: entry.frame_index,
    image: entry.image,
    audio: entry.audio,
  }));
  const shots: WorkplaceShot[] = [
    {
      shot_id: 2,
      shot_key: "shot_002",
      status: "prompt_ready",
      summary: "Preview shot",
      cut: true,
      has_video: false,
      has_actions: false,
      accepted: false,
      timeline: {
        start_seconds: 0,
        end_seconds: 5,
        duration_seconds: 5,
        label: "00:00 - 00:05",
      },
    },
    {
      shot_id: 3,
      shot_key: "shot_003",
      status: "prompt_ready",
      summary: "Second preview shot",
      cut: true,
      has_video: false,
      has_actions: false,
      accepted: false,
      timeline: {
        start_seconds: 5,
        end_seconds: 10,
        duration_seconds: 5,
        label: "00:05 - 00:10",
      },
    },
  ];

  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex min-h-screen w-full max-w-4xl flex-col border-x border-border/55">
        <header className="border-b border-border/55 px-4 py-4">
          <p className="text-[10px] font-medium uppercase text-muted-foreground">
            Echo Director
          </p>
          <h1 className="mt-1 text-lg font-semibold">Memory Bank Board</h1>
        </header>
        <div className="h-[42rem] overflow-y-auto bg-foreground/[0.012]">
          <div className="sticky top-0 z-20 bg-background/95 pb-2 backdrop-blur-sm">
            <MemoryBankBoard
              assets={assets}
              shots={shots}
              showSlots={false}
              onSaveAsset={async () => undefined}
              onDeleteAsset={async () => undefined}
              onApplySlots={async () => undefined}
            />
          </div>
          <div className="space-y-4 p-4">
            {shots.map((shot, index) => (
              <div key={shot.shot_id}>
                <ShotMemorySlots
                  assets={assets}
                  shot={shot}
                  conditionImage={index === 0 ? assets[0].image : null}
                  onApplySlots={async () => undefined}
                />
                <div className="grid h-64 place-items-center rounded-2xl border border-border/50 bg-background text-sm text-muted-foreground">
                  Shot {shot.shot_id} card
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
