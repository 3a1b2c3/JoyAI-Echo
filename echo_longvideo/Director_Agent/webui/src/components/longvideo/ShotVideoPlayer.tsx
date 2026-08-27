import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  Maximize,
  Minimize,
  Pause,
  Play,
  Volume2,
  VolumeX,
} from "lucide-react";

import { cn } from "@/lib/utils";

interface ShotVideoPlayerProps {
  src: string;
  className?: string;
}

function formatTime(seconds: number): string {
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const sec = total % 60;
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

/** ShotCard overlay video player: play/pause, muted, fullscreen, draggable scrubber. */
export function ShotVideoPlayer({ src, className }: ShotVideoPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const seekingRef = useRef(false);

  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const onFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement === containerRef.current);
    };
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () =>
      document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, []);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      void video.play();
    } else {
      video.pause();
    }
  }, []);

  const toggleMute = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    video.muted = !video.muted;
    setMuted(video.muted);
  }, []);

  const toggleFullscreen = useCallback(async () => {
    const container = containerRef.current;
    const video = videoRef.current;
    if (!container || !video) return;
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
        return;
      }
      if (container.requestFullscreen) {
        await container.requestFullscreen();
        return;
      }
      const webkitVideo = video as HTMLVideoElement & {
        webkitEnterFullscreen?: () => void;
      };
      webkitVideo.webkitEnterFullscreen?.();
    } catch {
      // ignore unsupported / denied fullscreen
    }
  }, []);

  const seekFromClientX = useCallback(
    (clientX: number) => {
      const bar = progressRef.current;
      const video = videoRef.current;
      if (!bar || !video || !duration) return;
      const rect = bar.getBoundingClientRect();
      const pct = Math.max(
        0,
        Math.min(1, (clientX - rect.left) / rect.width),
      );
      video.currentTime = pct * duration;
      setCurrentTime(video.currentTime);
    },
    [duration],
  );

  const onProgressPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      seekingRef.current = true;
      e.currentTarget.setPointerCapture(e.pointerId);
      seekFromClientX(e.clientX);
    },
    [seekFromClientX],
  );

  const onProgressPointerMove = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (!seekingRef.current) return;
      seekFromClientX(e.clientX);
    },
    [seekFromClientX],
  );

  const onProgressPointerUp = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (!seekingRef.current) return;
      seekingRef.current = false;
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        // already released
      }
      seekFromClientX(e.clientX);
    },
    [seekFromClientX],
  );

  const progressPct = duration ? (currentTime / duration) * 100 : 0;

  return (
    <div
      ref={containerRef}
      className={cn(
        "group/shot-player relative overflow-hidden rounded-xl border border-foreground/8 bg-black/40 shadow-sm",
        isFullscreen && "flex h-full w-full items-center justify-center rounded-none border-0 bg-black",
        className,
      )}
    >
      <video
        ref={videoRef}
        key={src}
        src={src}
        preload="metadata"
        playsInline
        className={cn(
          "w-full cursor-pointer object-contain",
          isFullscreen && "max-h-full max-w-full",
        )}
        onClick={togglePlay}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onTimeUpdate={() => {
          if (seekingRef.current || !videoRef.current) return;
          setCurrentTime(videoRef.current.currentTime);
        }}
        onLoadedMetadata={() => {
          const v = videoRef.current;
          if (!v) return;
          setDuration(v.duration || 0);
          setMuted(v.muted);
        }}
        onEnded={() => setPlaying(false)}
      />

      {/* Bottom overlay controls */}
      <div
        className={cn(
          "pointer-events-none absolute inset-x-0 bottom-0 z-10",
          "bg-gradient-to-t from-black/75 via-black/35 to-transparent px-3 pb-2.5 pt-10",
          "transition-opacity duration-200",
          playing
            ? "opacity-0 group-hover/shot-player:opacity-100 group-focus-within/shot-player:opacity-100"
            : "opacity-100",
        )}
      >
        <div className="pointer-events-auto flex items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                togglePlay();
              }}
              className="grid h-7 w-7 shrink-0 place-items-center rounded-md text-white transition-colors hover:bg-white/10"
              aria-label={playing ? "Pause" : "Play"}
            >
              {playing ? (
                <Pause className="h-3.5 w-3.5" fill="currentColor" />
              ) : (
                <Play className="ml-0.5 h-3.5 w-3.5" fill="currentColor" />
              )}
            </button>
            <span className="truncate text-[11px] tabular-nums text-white/90">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          <div className="flex shrink-0 items-center gap-0.5">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                toggleMute();
              }}
              className="grid h-7 w-7 place-items-center rounded-md text-white transition-colors hover:bg-white/10"
              aria-label={muted ? "Unmute" : "Mute"}
            >
              {muted ? (
                <VolumeX className="h-3.5 w-3.5" />
              ) : (
                <Volume2 className="h-3.5 w-3.5" />
              )}
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                void toggleFullscreen();
              }}
              className="grid h-7 w-7 place-items-center rounded-md text-white transition-colors hover:bg-white/10"
              aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            >
              {isFullscreen ? (
                <Minimize className="h-3.5 w-3.5" />
              ) : (
                <Maximize className="h-3.5 w-3.5" />
              )}
            </button>
          </div>
        </div>

        <div
          ref={progressRef}
          role="slider"
          aria-label="Playback progress"
          aria-valuemin={0}
          aria-valuemax={Math.round(duration)}
          aria-valuenow={Math.round(currentTime)}
          tabIndex={0}
          className="pointer-events-auto group/bar relative mt-2 h-1.5 cursor-pointer touch-none rounded-full bg-white/25"
          onPointerDown={onProgressPointerDown}
          onPointerMove={onProgressPointerMove}
          onPointerUp={onProgressPointerUp}
          onPointerCancel={onProgressPointerUp}
          onKeyDown={(e) => {
            const video = videoRef.current;
            if (!video || !duration) return;
            if (e.key === "ArrowLeft") {
              e.preventDefault();
              video.currentTime = Math.max(0, video.currentTime - 1);
              setCurrentTime(video.currentTime);
            } else if (e.key === "ArrowRight") {
              e.preventDefault();
              video.currentTime = Math.min(duration, video.currentTime + 1);
              setCurrentTime(video.currentTime);
            }
          }}
        >
          <div
            className="h-full rounded-full bg-white"
            style={{ width: `${progressPct}%` }}
          />
          <div
            className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-sm"
            style={{ left: `${progressPct}%` }}
          />
        </div>
      </div>
    </div>
  );
}
