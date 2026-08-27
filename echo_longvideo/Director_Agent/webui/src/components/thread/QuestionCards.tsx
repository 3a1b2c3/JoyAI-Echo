import { useCallback, useState } from "react";
import { Check } from "lucide-react";
import { cn } from "@/lib/utils";
import type { UIQuestionCard } from "@/lib/types";

interface QuestionCardsProps {
  cards: UIQuestionCard[];
  onAnswer: (cardId: string, value: string) => void;
  /** Agent turn not finished yet (waiting for stream_end resuming:false). */
  disabled?: boolean;
}

export function QuestionCards({
  cards,
  onAnswer,
  disabled = false,
}: QuestionCardsProps) {
  return (
    <div
      className={cn(
        "mt-3 flex flex-col gap-3",
        disabled && "pointer-events-none opacity-60",
      )}
      aria-disabled={disabled || undefined}
    >
      {cards.map((card) => (
        <SingleQuestionCard
          key={card.id}
          card={card}
          disabled={disabled}
          onAnswer={(value) => onAnswer(card.id, value)}
        />
      ))}
    </div>
  );
}

function SingleQuestionCard({
  card,
  onAnswer,
  disabled = false,
}: {
  card: UIQuestionCard;
  onAnswer: (value: string) => void;
  disabled?: boolean;
}) {
  const isAnswered = card.answered != null;
  const isLocked = disabled || isAnswered;
  const [customValue, setCustomValue] = useState("");
  const [showCustom, setShowCustom] = useState(false);

  const handleSelect = useCallback(
    (label: string) => {
      if (isLocked) return;
      onAnswer(label);
    },
    [isLocked, onAnswer],
  );

  const handleCustomSubmit = useCallback(() => {
    if (!customValue.trim() || isLocked) return;
    onAnswer(customValue.trim());
  }, [customValue, isLocked, onAnswer]);

  return (
    <div
      className={cn(
        "rounded-xl border border-border/50 bg-card/30 p-3.5 transition-all duration-300",
        "animate-in fade-in-0 slide-in-from-bottom-1 duration-300",
        isAnswered && "opacity-75",
      )}
    >
      <p className="mb-2.5 text-[13px] font-medium text-foreground/80">
        {card.question}
      </p>

      <div className="flex flex-wrap gap-2">
        {card.options.map((opt) => {
          const isSelected = card.answered === opt.label;
          return (
            <button
              key={opt.label}
              type="button"
              disabled={isLocked}
              onClick={() => handleSelect(opt.label)}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5",
                "text-[12.5px] font-medium transition-all duration-200",
                "border border-border/60",
                !isLocked &&
                  !isSelected && [
                    "bg-background hover:bg-foreground/[0.06] hover:border-foreground/20",
                    "active:scale-[0.97]",
                  ],
                isSelected && [
                  "bg-foreground text-background border-foreground",
                  "shadow-sm",
                ],
                isLocked && !isSelected && "opacity-40",
                "disabled:cursor-default",
              )}
            >
              {isSelected && <Check className="h-3 w-3" />}
              {opt.label}
            </button>
          );
        })}
      </div>

      {card.allowCustom && !isAnswered && !disabled && (
        <div className="mt-2.5">
          {!showCustom ? (
            <button
              type="button"
              onClick={() => setShowCustom(true)}
              className="text-[11.5px] text-muted-foreground/70 transition-colors hover:text-foreground/70"
            >
              + Custom response
            </button>
          ) : (
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={customValue}
                onChange={(e) => setCustomValue(e.target.value)}
                onBlur={() => {
                  if (!customValue.trim()) {
                    setShowCustom(false);
                  }
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    handleCustomSubmit();
                  }
                }}
                placeholder="Enter your response..."
                className={cn(
                  "flex-1 rounded-lg border border-border/60 bg-background px-3 py-1.5",
                  "text-[12.5px] text-foreground placeholder:text-muted-foreground/50",
                  "focus:border-foreground/20 focus:outline-none focus:ring-1 focus:ring-foreground/10",
                  "transition-all duration-150",
                )}
                autoFocus
              />
              <button
                type="button"
                onClick={handleCustomSubmit}
                disabled={!customValue.trim()}
                className={cn(
                  "rounded-lg px-3 py-1.5 text-[12px] font-medium",
                  "bg-foreground text-background",
                  "transition-all duration-150",
                  "hover:bg-foreground/90 active:scale-[0.97]",
                  "disabled:opacity-40 disabled:cursor-default",
                )}
              >
                Confirm
              </button>
            </div>
          )}
        </div>
      )}

      {card.allowCustom &&
        isAnswered &&
        !card.options.some((o) => o.label === card.answered) && (
          <div className="mt-2 flex items-center gap-1.5">
            <Check className="h-3 w-3 text-foreground/60" />
            <span className="text-[12px] text-foreground/70">
              {card.answered}
            </span>
          </div>
        )}
    </div>
  );
}
