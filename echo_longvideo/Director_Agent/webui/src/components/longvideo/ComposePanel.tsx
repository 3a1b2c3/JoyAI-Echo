import { useCallback, useEffect, useRef, useState } from "react";
import {
  // ArrowLeft,
  CheckCircle2,
  ChevronUp,
  Download,
  FileText,
  Film,
  Loader2,
  Maximize,
  MessageSquarePlus,
  Minimize,
  Pause,
  Play,
  RefreshCcw,
  Sparkles,
  Volume2,
  VolumeX,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { downloadMedia } from "@/lib/downloadMedia";
import { markComposeVideoPlayed, wasComposeVideoPlayed } from "@/lib/media";
import { cn } from "@/lib/utils";
import { useClient } from "@/providers/ClientProvider";
import type { RenderStatus } from "@/components/longvideo/RenderOverlay";
import { LikeButton } from "./LikeButton";
import {
  downloadStoryboardSegments,
  type StoryboardSegment,
} from "./StoryboardScriptEditor";

interface ComposePanelProps {
  renderStatus: RenderStatus;
  composingTitle?: string;
  composingHint?: string;
  videoUrl?: string;
  sessionKey?: string;
  downloadFileName?: string;
  progress?: number;
  error?: string;
  onRetry: () => void;
  onBack?: () => void;
  onNewChat?: () => void;
  /** RenderOverlay 正在播放同一成片时，避免底层重复 autoPlay */
  playbackInOverlay?: boolean;
  storyboardSegments?: StoryboardSegment[];
  echoRequestId?: string | null;
  likeStatus?: number;
  onEchoLike?: (action: 1 | 2) => Promise<void>;
  onEchoDownloadPrompt?: () => Promise<void>;
}

export function ComposePanel({
  renderStatus,
  composingTitle,
  composingHint,
  videoUrl,
  sessionKey,
  downloadFileName,
  progress,
  error,
  onRetry,
  // onBack,
  onNewChat,
  playbackInOverlay = false,
  storyboardSegments,
  echoRequestId,
  likeStatus,
  onEchoLike,
  onEchoDownloadPrompt,
}: ComposePanelProps) {
  if (renderStatus === "rendering") {
    return (
      <ComposingState
        progress={progress}
        title={composingTitle}
        hint={composingHint}
      />
    );
  }

  if (renderStatus === "error") {
    return <ComposeError error={error} onRetry={onRetry} />;
  }

  if (renderStatus === "done" && videoUrl) {
    return (
      <ComposePlayer
        videoUrl={videoUrl}
        sessionKey={sessionKey}
        downloadFileName={downloadFileName}
        onRetry={onRetry}
        onNewChat={onNewChat}
        playbackInOverlay={playbackInOverlay}
        storyboardSegments={storyboardSegments}
        echoRequestId={echoRequestId}
        likeStatus={likeStatus}
        onEchoLike={onEchoLike}
        onEchoDownloadPrompt={onEchoDownloadPrompt}
      />
    );
  }

  if (renderStatus === "done" && !videoUrl) {
    return <ComposeDone onRetry={onRetry} />;
  }

  return (
    <div className="flex h-full flex-col items-center justify-center px-8 text-center">
      <div className="relative mb-8">
        <div className="absolute -inset-6 rounded-3xl bg-gradient-to-br from-foreground/[0.03] to-transparent" />
        <div className="relative grid h-16 w-16 place-items-center rounded-2xl bg-gradient-to-br from-foreground/[0.06] to-foreground/[0.02] shadow-sm ring-1 ring-inset ring-foreground/[0.08]">
          <Film className="h-7 w-7 text-foreground/35" />
        </div>
      </div>
      <p className="text-base font-semibold tracking-tight text-foreground/90">
        Ready to assemble
      </p>
      <p className="mt-2.5 max-w-[280px] text-[13px] leading-relaxed text-muted-foreground/70">
        All shots are ready for the final cut.
      </p>
    </div>
  );
}

function ComposingState({
  progress,
  title,
  hint,
}: {
  progress?: number;
  title?: string;
  hint?: string;
}) {
  const pct = progress ?? 0;

  return (
    <div className="flex h-full flex-col items-center justify-center px-8 text-center">
      <div className="relative mb-8">
        <div className="absolute -inset-6 rounded-3xl bg-gradient-to-br from-foreground/[0.03] to-transparent" />
        <div className="relative grid h-16 w-16 place-items-center rounded-2xl bg-gradient-to-br from-foreground/[0.06] to-foreground/[0.02] shadow-sm ring-1 ring-inset ring-foreground/[0.08]">
          <Loader2 className="h-7 w-7 animate-spin text-foreground/35" />
        </div>
      </div>
      <p className="text-base font-semibold tracking-tight text-foreground/90">
        {title || "Assembling final cut"}
      </p>
      <p className="mt-2.5 text-[13px] text-muted-foreground/70">
        {hint || "Merging approved shots..."}
      </p>
      <div className="mt-6 w-60">
        <div className="h-1 overflow-hidden rounded-full bg-foreground/[0.06]">
          <div
            className="h-full rounded-full bg-gradient-to-r from-foreground/15 to-foreground/25 transition-all duration-700 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="mt-2 block text-[11px] tabular-nums text-muted-foreground/40">
          {Math.round(pct)}%
        </span>
      </div>
    </div>
  );
}

function ComposeError({
  error,
  onRetry,
}: {
  error?: string;
  onRetry: () => void;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-8 text-center">
      <div className="relative mb-8">
        <div className="absolute -inset-6 rounded-3xl bg-gradient-to-br from-red-500/[0.04] to-transparent" />
        <div className="relative grid h-16 w-16 place-items-center rounded-2xl bg-gradient-to-br from-red-500/[0.08] to-red-500/[0.03] shadow-sm ring-1 ring-inset ring-red-500/[0.12]">
          <span className="text-xl font-semibold text-red-400/60">!</span>
        </div>
      </div>
      <p className="text-base font-semibold tracking-tight text-foreground/90">
        Assembly failed
      </p>
      <p className="mt-2.5 max-w-[280px] text-[13px] leading-relaxed text-muted-foreground/70">
        {error || "An error interrupted the final assembly. Please retry."}
      </p>
      <div className="mt-6">
        <Button
          onClick={onRetry}
          className={cn(
            "h-9 rounded-xl px-5 text-[12px] font-medium",
            "bg-foreground text-background shadow-sm hover:bg-foreground/90",
            "transition-all duration-200",
          )}
        >
          <RefreshCcw className="mr-1.5 h-3.5 w-3.5" />
          Retry Assembly
        </Button>
      </div>
    </div>
  );
}

export function ComposePlayer({
  videoUrl,
  sessionKey,
  downloadFileName,
  onRetry,
  onNewChat,
  playbackInOverlay = false,
  storyboardSegments,
  echoRequestId,
  likeStatus,
  onEchoLike,
  onEchoDownloadPrompt,
}: {
  videoUrl: string;
  sessionKey?: string;
  downloadFileName?: string;
  onRetry: () => void;
  onNewChat?: () => void;
  playbackInOverlay?: boolean;
  storyboardSegments?: StoryboardSegment[];
  echoRequestId?: string | null;
  likeStatus?: number;
  onEchoLike?: (action: 1 | 2) => Promise<void>;
  onEchoDownloadPrompt?: () => Promise<void>;
}) {
  const { token } = useClient();
  const videoRef = useRef<HTMLVideoElement>(null);
  const playerContainerRef = useRef<HTMLDivElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [downloading, setDownloading] = useState(false);
  const shouldAutoPlay =
    !playbackInOverlay && !wasComposeVideoPlayed(videoUrl);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !playbackInOverlay) return;
    v.pause();
    setPlaying(false);
  }, [playbackInOverlay]);

  useEffect(() => {
    const onFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", onFullscreenChange);
    };
  }, []);

  const togglePlay = useCallback((e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      v.play().catch(() => {});
    } else {
      v.pause();
    }
  }, []);

  const toggleMute = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    v.muted = !v.muted;
    setMuted(v.muted);
  }, []);

  const toggleFullscreen = useCallback(async () => {
    const container = playerContainerRef.current;
    const video = videoRef.current;
    if (!container && !video) return;

    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
        return;
      }

      if (container?.requestFullscreen) {
        await container.requestFullscreen();
        return;
      }

      // iOS Safari: only video element supports native fullscreen
      const webkitVideo = video as HTMLVideoElement & {
        webkitEnterFullscreen?: () => void;
      };
      webkitVideo.webkitEnterFullscreen?.();
    } catch {
      // ignore fullscreen failures (unsupported / permission denied)
    }
  }, []);

  const handleSeek = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const bar = progressRef.current;
      const v = videoRef.current;
      if (!bar || !v || !duration) return;
      const rect = bar.getBoundingClientRect();
      const pct = Math.max(
        0,
        Math.min(1, (e.clientX - rect.left) / rect.width),
      );
      v.currentTime = pct * duration;
      setCurrentTime(v.currentTime);
    },
    [duration],
  );

  const formatTime = (s: number) => {
    const total = Math.round(s);
    const m = Math.floor(total / 60);
    const sec = total % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };
  // downloadMedia 内部 catch 错误，finally 统一复位按钮 loading
  const handleDownload = useCallback(async () => {
    if (downloading) return;
    setDownloading(true);
    try {
      await downloadMedia({
        url: videoUrl,
        sessionKey,
        token,
        fileName: downloadFileName,
      });
    } finally {
      setDownloading(false);
    }
  }, [downloading, downloadFileName, sessionKey, token, videoUrl]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* header badge */}
      <div className="flex items-center justify-center border-b border-border/30 px-4 py-2.5">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/[0.08] px-3 py-1 text-[11px] font-medium text-emerald-600 dark:text-emerald-400 ring-1 ring-inset ring-emerald-500/[0.15]">
          <CheckCircle2 className="h-3 w-3" />
          Final cut ready
        </span>
      </div>

      <div className="flex min-h-0 flex-1 flex-col items-center justify-center">
        <div className="overflow-hidden p-4">
          <div
            ref={playerContainerRef}
            className={cn(
              "group/player relative overflow-hidden rounded-2xl border border-border/30 bg-black/95 shadow-lg",
              "h-auto",
              isFullscreen && "flex h-full w-full items-center justify-center rounded-none border-0",
            )}
          >
            <video
              ref={videoRef}
              src={videoUrl}
              autoPlay={shouldAutoPlay}
              className={cn(
                "h-full w-auto cursor-pointer object-contain",
                isFullscreen && "max-h-full max-w-full",
              )}
              onClick={togglePlay}
              onPlay={() => {
                markComposeVideoPlayed(videoUrl);
                setPlaying(true);
              }}
              onPause={() => setPlaying(false)}
              onTimeUpdate={() => {
                if (videoRef.current)
                  setCurrentTime(videoRef.current.currentTime);
              }}
              onLoadedMetadata={() => {
                const v = videoRef.current;
                if (!v) return;
                setDuration(v.duration);
              }}
            />
            <div
              className={cn(
                "absolute inset-0 flex cursor-pointer items-center justify-center bg-black/20 transition-opacity duration-300",
                playing
                  ? "opacity-0 group-hover/player:opacity-100"
                  : "opacity-100",
              )}
              onClick={togglePlay}
            >
              <div className="grid h-14 w-14 place-items-center rounded-full bg-black/40 backdrop-blur-md ring-1 ring-white/10 transition-transform duration-200 hover:scale-105">
                {playing ? (
                  <Pause className="h-5 w-5 text-white/90" />
                ) : (
                  <Play className="ml-0.5 h-5 w-5 text-white/90" />
                )}
              </div>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                void toggleFullscreen();
              }}
              className={cn(
                "absolute bottom-3 right-3 z-10 grid h-8 w-8 place-items-center rounded-lg",
                "bg-black/40 text-white/80 backdrop-blur-md ring-1 ring-white/10",
                "transition-opacity duration-200 hover:bg-black/55 hover:text-white",
                playing
                  ? "opacity-0 group-hover/player:opacity-100"
                  : "opacity-100",
              )}
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
      </div>
      <>
        {/* controls */}
        <div className="w-full shrink-0 border-t border-border/30 px-4 py-3">
          {/* progress bar */}
          <div
            ref={progressRef}
            className={cn(
              "group/bar relative mb-3 h-1 cursor-pointer rounded-full bg-foreground/[0.06] transition-all",
              "hover:h-1.5",
            )}
            onClick={handleSeek}
          >
            <div
              className="h-full rounded-full bg-foreground/25 transition-[width] duration-100"
              style={{
                width: duration ? `${(currentTime / duration) * 100}%` : "0%",
              }}
            />
            <div
              className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-foreground/70 opacity-0 shadow-sm transition-opacity group-hover/bar:opacity-100"
              style={{
                left: duration ? `${(currentTime / duration) * 100}%` : "0%",
              }}
            />
          </div>
        </div>
        {/* buttons row */}
        <div className="flex items-center justify-between px-2 pr-4">
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={togglePlay}
              className="grid h-7 w-7 place-items-center rounded-lg text-foreground/50 transition-colors hover:bg-foreground/[0.05] hover:text-foreground/70"
            >
              {playing ? (
                <Pause className="h-3.5 w-3.5" />
              ) : (
                <Play className="ml-0.5 h-3.5 w-3.5" />
              )}
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                toggleMute();
              }}
              className="grid h-7 w-7 place-items-center rounded-lg text-foreground/50 transition-colors hover:bg-foreground/[0.05] hover:text-foreground/70"
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
              className="grid h-7 w-7 place-items-center rounded-lg text-foreground/50 transition-colors hover:bg-foreground/[0.05] hover:text-foreground/70"
              aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            >
              {isFullscreen ? (
                <Minimize className="h-3.5 w-3.5" />
              ) : (
                <Maximize className="h-3.5 w-3.5" />
              )}
            </button>
            <span className="ml-0.5 text-[10px] tabular-nums text-muted-foreground/40">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onRetry();
                }}
                className="flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-[10px] font-medium text-muted-foreground transition-colors hover:bg-foreground/[0.05] hover:text-foreground/70"
              >
                <RefreshCcw className="h-3 w-3" />
                Reassemble
            </button>
            {onNewChat && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onNewChat();
                }}
                className="flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-[10px] font-medium text-muted-foreground transition-colors hover:bg-foreground/[0.05] hover:text-foreground/70"
              >
                <MessageSquarePlus className="h-3 w-3" />
                New Project
              </button>
            )}
            <button
              type="button"
              disabled={downloading}
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                void handleDownload();
              }}
              className={cn(
                "flex items-center gap-1 rounded-lg px-3 py-1.5",
                "border border-border/40 bg-foreground/[0.03]",
                "text-[10px] font-medium text-foreground/60",
                "transition-all duration-200 hover:bg-foreground/[0.06] hover:text-foreground/80 hover:shadow-sm",
                "disabled:pointer-events-none disabled:opacity-50",
              )}
            >
              {downloading ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Download className="h-3 w-3" />
              )}
              Download
            </button>
            <LikeButton
              className="text-foreground/40 hover:text-foreground/80"
              spanClassName="bg-foreground/40 hover:bg-foreground/80"
              echoRequestId={echoRequestId}
              likeStatus={(likeStatus ?? 0) as 0 | 1 | 2}
              onLike={onEchoLike}
            />
          </div>
        </div>
        <ScriptPanel
          segments={storyboardSegments ?? []}
          onEchoDownloadPrompt={onEchoDownloadPrompt}
        />
      </>
    </div>
  );
}

function ScriptPanel({
  segments,
  onEchoDownloadPrompt,
}: {
  segments: StoryboardSegment[];
  onEchoDownloadPrompt?: () => Promise<void>;
}) {
  const [open, setOpen] = useState(false);

  const handleDownload = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (segments.length === 0) return;
      downloadStoryboardSegments(segments);
      void onEchoDownloadPrompt?.();
    },
    [onEchoDownloadPrompt, segments],
  );

  return (
    <div
      className={cn(
        "shrink-0 overflow-hidden border-t border-border/30 bg-background transition-[max-height] duration-300 ease-out",
        open ? "max-h-[40vh]" : "max-h-[42px]",
      )}
    >
      <div
        role="button"
        tabIndex={0}
        className="flex cursor-pointer select-none items-center justify-between px-4 py-2.5 transition-colors hover:bg-foreground/[0.03]"
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setOpen((v) => !v);
          }
        }}
      >
        <div className="flex items-center gap-2">
          <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground/70" />
          <span className="text-[12px] font-medium tracking-wide text-muted-foreground">
            Script
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleDownload}
            disabled={segments.length === 0}
            className={cn(
              "flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-muted-foreground transition-colors hover:bg-foreground/[0.06] hover:text-foreground",
              "disabled:pointer-events-none disabled:opacity-40",
            )}
          >
            <Download className="h-3 w-3" />
            Download
          </button>
          <ChevronUp
            className={cn(
              "h-4 w-4 shrink-0 text-muted-foreground/70 transition-transform duration-300",
              open && "rotate-180",
            )}
          />
        </div>
      </div>
      {open ? (
        <div className="max-h-[calc(40vh-42px)] overflow-y-auto px-4 pb-3.5 scrollbar-thin">
          {segments.length === 0 ? (
            <p className="py-2 text-center text-[12px] text-muted-foreground/60">
              No shot plan available
            </p>
          ) : (
            <div className="script-panel-section rounded-[10px] border border-foreground/[0.04] bg-foreground/[0.015] px-3.5 py-2.5 animate-in fade-in-0 slide-in-from-bottom-2.5 duration-300">
              {segments.map((seg) => (
                <p
                  key={seg.id}
                  className="whitespace-pre-wrap text-[12px] leading-[1.75] text-foreground/[0.42] [&+&]:mt-3"
                >
                  {seg.text}
                </p>
              ))}
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}

function ComposeDone({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-8 text-center">
      <div className="relative mb-8">
        <div className="absolute -inset-6 rounded-3xl bg-gradient-to-br from-foreground/[0.03] to-transparent" />
        <div className="relative grid h-16 w-16 place-items-center rounded-2xl bg-gradient-to-br from-foreground/[0.06] to-foreground/[0.02] shadow-sm ring-1 ring-inset ring-foreground/[0.08]">
          <Sparkles className="h-7 w-7 text-foreground/35" />
        </div>
      </div>
      <p className="text-base font-semibold tracking-tight text-foreground/90">
        Final cut ready
      </p>
      <p className="mt-2.5 max-w-[280px] text-[13px] leading-relaxed text-muted-foreground/70">
        The video is complete, but no playback URL is available.
      </p>
      <div className="mt-6">
        <Button
          onClick={onRetry}
          className={cn(
            "h-9 rounded-xl px-5 text-[12px] font-medium",
            "bg-foreground text-background shadow-sm hover:bg-foreground/90",
            "transition-all duration-200",
          )}
        >
          <RefreshCcw className="mr-1.5 h-3.5 w-3.5" />
          Reassemble
        </Button>
      </div>
    </div>
  );
}
