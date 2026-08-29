import { cn } from "@/lib/utils";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import { useEffect, useState } from "react";

export type LikeType = 1;
export type UnLikeType = 2;
export type DefaultType = 0;
export type LikeButtonType = LikeType | UnLikeType | DefaultType;

export function LikeButton(props: {
  echoRequestId?: string | null;
  likeStatus?: LikeButtonType;
  onLike?: (action: 1 | 2) => Promise<void>;
  className: string;
  spanClassName?: string;
}) {
  const {
    echoRequestId,
    likeStatus = 0,
    onLike,
    className,
    spanClassName = "bg-foreground/40 hover:bg-foreground/80",
  } = props;

  const [liked, setLiked] = useState<LikeButtonType>(likeStatus);

  useEffect(() => {
    setLiked(likeStatus);
  }, [likeStatus]);
  console.log("likeStatus", likeStatus, echoRequestId);
  // if (!echoRequestId) return null;

  const handleLike = async (type: 1 | 2) => {
    if (!onLike) return;
    const prev = liked;
    const optimistic: LikeButtonType = prev === type ? 0 : type;
    setLiked(optimistic);
    try {
      await onLike(type);
    } catch (err) {
      setLiked(prev);
      console.warn("failed to update like status", err);
    }
  };

  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={() => void handleLike(1)}
        className={cn(
          className,
          liked === 1 ? "text-red-500 hover:text-red-500" : className,
        )}
      >
        <ThumbsUp className="h-3 w-3" />
      </button>
      <span
        className={cn("h-[10px] w-[1px] rounded-full", spanClassName)}
      ></span>
      <button
        type="button"
        onClick={() => void handleLike(2)}
        className={cn(
          className,
          liked === 2 ? "text-gray-500 hover:text-gray-500" : className,
        )}
      >
        <ThumbsDown
          className="h-3 w-3"
          fill={liked === 2 ? "currentColor" : "none"}
        />
      </button>
    </div>
  );
}
