import { useState } from "react";

import { MemoryReviewCard } from "@/components/thread/MemoryReviewCard";
import { careerMemoryReviewPreview } from "@/lib/memory-review-preview";
import type { MemoryReview } from "@/lib/types";

export function MemoryReviewPreviewPage() {
  const [status, setStatus] =
    useState<MemoryReview["status"]>("awaiting_method");
  const [attempt, setAttempt] = useState(1);
  const [selectionMode, setSelectionMode] =
    useState<MemoryReview["selection_mode"]>(null);
  const [manualSelectedIds, setManualSelectedIds] = useState<string[]>([]);
  const message = careerMemoryReviewPreview(
    status,
    attempt,
    selectionMode,
    manualSelectedIds,
  );
  const review = message.memoryReview!;

  const reselect = () => {
    setSelectionMode("vlm");
    setStatus("reselecting");
    window.setTimeout(() => {
      setAttempt((value) => value + 1);
      setStatus("awaiting_review");
    }, 900);
  };

  return (
    <main className="min-h-screen bg-background px-4 py-8 text-foreground sm:px-8">
      <div className="mx-auto w-full max-w-5xl">
        <header className="mb-8 space-y-2">
          <p className="text-xs font-medium uppercase tracking-[0.2em] text-violet-600 dark:text-violet-300">
            Local review preview
          </p>
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            Memory Selection Review
          </h1>
          <p className="max-w-2xl text-sm leading-6 text-muted-foreground">
            Local review preview for the production memory workflow.
          </p>
        </header>

        <section className="rounded-3xl border bg-card/60 p-4 shadow-sm sm:p-6">
          <div className="mb-4 flex items-start gap-3">
            <div className="grid size-8 shrink-0 place-items-center rounded-full bg-violet-600 text-xs font-semibold text-white">
              AI
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-sm leading-6">{message.content}</p>
              <MemoryReviewCard
                review={review}
                onApprove={() => setStatus("approved")}
                onReselect={reselect}
                onManualSelect={async (memoryId) => {
                  setManualSelectedIds((current) =>
                    current.includes(memoryId)
                      ? current
                      : [...current, memoryId],
                  );
                }}
                onCreateMemoryAsset={async () => {}}
                onSelectMode={(mode) => {
                  setSelectionMode(mode);
                  setStatus("selecting");
                  window.setTimeout(() => setStatus("awaiting_review"), 500);
                }}
              />
            </div>
          </div>
        </section>

        <p className="mt-4 text-center text-xs text-muted-foreground">
          Decide whether to create new Memory, then add references manually or let VLM extract them.
        </p>
      </div>
    </main>
  );
}
