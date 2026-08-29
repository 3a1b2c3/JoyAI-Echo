import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  Download,
  Loader2,
  MessageSquarePlus,
  Pause,
  Play,
  RefreshCcw,
  Volume2,
  VolumeX,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { downloadMedia } from "@/lib/downloadMedia";
import { markComposeVideoPlayed, wasComposeVideoPlayed } from "@/lib/media";
import { cn } from "@/lib/utils";
import { useClient } from "@/providers/ClientProvider";
// import { LikeButton } from "./LikeButton";

export type RenderStatus = "idle" | "rendering" | "done" | "error";

interface RenderOverlayProps {
  status: RenderStatus;
  videoUrl?: string;
  sessionKey?: string;
  downloadFileName?: string;
  error?: string;
  progress?: number;
  onRetry: () => void;
  onDismiss: () => void;
  onNewChat?: () => void;
  hideBack?: boolean;
  echoRequestId?: string | null;
  likeStatus?: number;
  onEchoLike?: (action: 1 | 2) => Promise<void>;
}

function renderOverlayCopy() {
  return {
    rendering: "Assembling final cut",
    failedTitle: "Assembly failed",
    failedDefault: "An error interrupted assembly. Retry or return to the shot plan.",
    retry: "Reassemble",
    doneTitle: "Final cut ready",
  };
}

function overlayVisualTheme() {
  return {
    elapsedHint: "Keep this page open · Elapsed",
    textPrimary: "text-white/85",
    textSecondary: "text-white/30",
    textPct: "text-white/80",
    textPctSign: "text-white/30",
    textErrorIcon: "text-white/50",
    barTrack: "bg-white/[0.06]",
    particle: "bg-white/15",
    glowOuter: "radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 70%)",
    glowInner: "radial-gradient(circle, rgba(255,255,255,0.04) 0%, transparent 70%)",
    strokeTrack: "rgba(255,255,255,0.06)",
    strokeArc: "rgba(255,255,255,0.18)",
    strokeArc2: "rgba(255,255,255,0.08)",
    barFill: "linear-gradient(90deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.35) 100%)",
    barShimmer: "linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent)",
    errorBox: "border-white/[0.06] bg-white/[0.03]",
    btnGhost: "text-white/50 hover:bg-white/[0.05] hover:text-white/70",
    btnPrimary: "bg-white/[0.08] text-white/80 hover:bg-white/[0.12] border-white/[0.06]",
  };
}

type OverlayVisualTheme = ReturnType<typeof overlayVisualTheme>;

const TICK_MS = 3000;

const overlayCSS = `
@keyframes render-bar-flow {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
@keyframes render-glow-breathe {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 0.5; }
}
@keyframes render-fade-in {
  from { opacity: 0; transform: scale(0.97) translateY(10px); }
  to { opacity: 1; transform: scale(1) translateY(0); }
}
@keyframes render-float-up {
  0% { transform: translateY(0) scale(1); opacity: 0.35; }
  70% { opacity: 0.15; }
  100% { transform: translateY(-50px) scale(0.4); opacity: 0; }
}
`;

export function RenderOverlay({
  status,
  videoUrl,
  sessionKey,
  downloadFileName,
  error,
  progress,
  onRetry,
  onDismiss,
  onNewChat,
  hideBack,
  // echoRequestId,
  // likeStatus,
  // onEchoLike,
}: RenderOverlayProps) {
  if (status === "idle") return null;

  const copy = renderOverlayCopy();
  const theme = overlayVisualTheme();

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-[#0a0a0b]/95 backdrop-blur-xl"
    >
      <style>{overlayCSS}</style>
      <>
          {/* static film grain texture — dark overlay only */}
          <div
            className="pointer-events-none absolute inset-0 opacity-[0.018]"
            style={{
              backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
            }}
          />
          <div
            className="pointer-events-none absolute inset-0"
            style={{
              background:
                "radial-gradient(ellipse 50% 35% at 50% 50%, rgba(255,255,255,0.025) 0%, transparent 70%)",
            }}
          />
      </>
      {status === "rendering" && (
        <RenderingState
          progress={progress}
          renderingLabel={copy.rendering}
          theme={theme}
        />
      )}
      {status === "error" && (
        <ErrorState
          error={error}
          onRetry={onRetry}
          onDismiss={onDismiss}
          hideBack={hideBack}
          copy={copy}
          theme={theme}
        />
      )}
      {status === "done" && videoUrl && (
        <VideoPlayerPanel
          videoUrl={videoUrl}
          sessionKey={sessionKey}
          downloadFileName={downloadFileName}
          onRetry={onRetry}
          onDismiss={onDismiss}
          onNewChat={onNewChat}
          hideBack={hideBack}
          retryLabel={copy.retry}
          // echoRequestId={echoRequestId}
          // likeStatus={likeStatus}
          // onEchoLike={onEchoLike}
        />
      )}
      {status === "done" && !videoUrl && (
        <DonePlaceholder onDismiss={onDismiss} onRetry={onRetry} copy={copy} />
      )}
    </div>
  );
}

/* ── Rendering State ── */

function RenderingState({
  progress,
  renderingLabel,
  theme,
}: {
  progress?: number;
  renderingLabel: string;
  theme: OverlayVisualTheme;
}) {
  const [elapsed, setElapsed] = useState(0);
  const [simPct, setSimPct] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (progress != null) return;
    const t = setInterval(() => {
      setSimPct((p) => {
        if (p >= 95) return p;
        const delta = Math.random() * 8 + 4;
        return Math.min(95, p + delta);
      });
    }, TICK_MS);
    return () => clearInterval(t);
  }, [progress]);

  const pct = progress ?? simPct;
  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  const timeStr =
    mins > 0 ? `${mins}:${secs.toString().padStart(2, "0")}` : `${secs}s`;

  return (
    <div
      className="flex flex-col items-center gap-8"
      style={{ animation: "render-fade-in 0.6s ease-out both" }}
    >
      {/* animated film-frame icon */}
      <div className="relative">
          {/* outer glow rings — breathe in sync with TICK_MS */}
          <div
            className="absolute -inset-10 rounded-full"
            style={{
              background: theme.glowOuter,
              animation: `render-glow-breathe ${TICK_MS}ms ease-in-out infinite`,
            }}
          />
          <div
            className="absolute -inset-6 rounded-full"
            style={{
              background: theme.glowInner,
              animation: `render-glow-breathe ${TICK_MS}ms ease-in-out infinite ${TICK_MS / 3}ms`,
            }}
          />

          {/* core icon */}
          <div className="relative grid h-20 w-20 place-items-center">
            {/* spinning arcs */}
            <svg
              className="absolute inset-0 h-full w-full animate-spin"
              style={{ animationDuration: "10s" }}
              viewBox="0 0 80 80"
            >
              <circle
                cx="40"
                cy="40"
                r="36"
                fill="none"
                stroke={theme.strokeTrack}
                strokeWidth="1"
              />
              <path
                d="M40 4 A36 36 0 0 1 76 40"
                fill="none"
                stroke={theme.strokeArc}
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
            <svg
              className="absolute inset-0 h-full w-full animate-spin"
              style={{
                animationDuration: "14s",
                animationDirection: "reverse",
              }}
              viewBox="0 0 80 80"
            >
              <path
                d="M40 8 A32 32 0 0 0 8 40"
                fill="none"
                stroke={theme.strokeArc2}
                strokeWidth="1"
                strokeLinecap="round"
              />
            </svg>

            {/* percentage number — transitions smoothly over the same TICK duration */}
            <span
              className={cn(
                "relative text-2xl font-light tabular-nums tracking-tight",
                theme.textPct,
              )}
              style={{ transition: `all ${TICK_MS * 0.8}ms ease-out` }}
            >
              {Math.round(pct)}
              <span className={cn("text-sm", theme.textPctSign)}>%</span>
            </span>
          </div>
      </div>

      {/* text */}
      <div className="text-center">
        <p
          className={cn(
            "text-[15px] font-medium tracking-wide",
            theme.textPrimary,
          )}
        >
          {renderingLabel}
        </p>
        <p className={cn("mt-2 text-[12px]", theme.textSecondary)}>
          {theme.elapsedHint} {timeStr}
        </p>
      </div>

      {/* progress bar — transition synced to TICK_MS */}
      <div className="w-64">
        <div
          className={cn("h-[3px] overflow-hidden rounded-full", theme.barTrack)}
        >
          <div
            className="relative h-full rounded-full"
            style={{
              width: `${pct}%`,
              background: theme.barFill,
              transition: `width ${TICK_MS * 0.8}ms ease-out`,
            }}
          >
            {/* shimmer on the bar — same cycle as TICK */}
            <div
              className="absolute inset-0"
              style={{
                background: theme.barShimmer,
                backgroundSize: "200% 100%",
                animation: `render-bar-flow ${TICK_MS}ms ease-in-out infinite`,
              }}
            />
          </div>
        </div>
      </div>

      {/* particles floating up — cycle = TICK_MS based */}
      <div className="pointer-events-none absolute bottom-[38%] flex gap-8">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className={cn("h-0.5 w-0.5 rounded-full", theme.particle)}
            style={{
              animation: `render-float-up ${TICK_MS + i * 500}ms ease-out infinite ${i * (TICK_MS / 3)}ms`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

/* ── Error State ── */

function ErrorState({
  error,
  onRetry,
  onDismiss,
  hideBack,
  copy,
  theme,
}: {
  error?: string;
  onRetry: () => void;
  onDismiss: () => void;
  hideBack?: boolean;
  copy: ReturnType<typeof renderOverlayCopy>;
  theme: OverlayVisualTheme;
}) {
  return (
    <div
      className="flex max-w-md flex-col items-center gap-6 text-center"
      style={{ animation: "render-fade-in 0.5s ease-out both" }}
    >
      {/* error mark */}
      <div className="relative">
        <div
          className={cn(
            "grid h-16 w-16 place-items-center rounded-2xl border",
            theme.errorBox,
          )}
        >
          <span className={cn("text-2xl font-light", theme.textErrorIcon)}>
            !
          </span>
        </div>
      </div>

      <div>
        <p className={cn("text-[15px] font-medium", theme.textPrimary)}>
          {copy.failedTitle}
        </p>
        <p
          className={cn(
            "mt-2 max-w-xs text-[12px] leading-relaxed",
            theme.textSecondary,
          )}
        >
          {error || copy.failedDefault}
        </p>
      </div>

      <div className="flex items-center gap-3">
        {!hideBack && (
          <Button
            variant="ghost"
            onClick={onDismiss}
            className={cn(
              "h-9 rounded-lg px-5 text-[12px] font-medium",
              theme.btnGhost,
            )}
          >
            <ArrowLeft className="mr-1.5 h-3.5 w-3.5" />
            Back to Shots
          </Button>
        )}
        <Button
          onClick={onRetry}
          className={cn(
            "h-9 rounded-lg border px-5 text-[12px] font-medium",
            theme.btnPrimary,
          )}
        >
          <RefreshCcw className="mr-1.5 h-3.5 w-3.5" />
          {copy.retry}
        </Button>
      </div>
    </div>
  );
}

/* ── Done: Video Player ── */

function VideoPlayerPanel({
  videoUrl,
  sessionKey,
  downloadFileName,
  onRetry,
  onDismiss,
  onNewChat,
  hideBack,
  retryLabel,
  // echoRequestId,
  // likeStatus,
  // onEchoLike,
}: {
  videoUrl: string;
  sessionKey?: string;
  downloadFileName?: string;
  onRetry: () => void;
  onDismiss: () => void;
  onNewChat?: () => void;
  hideBack?: boolean;
  retryLabel: string;
  // echoRequestId?: string | null;
  // likeStatus?: number;
  // onEchoLike?: (action: 1 | 2) => Promise<void>;
}) {
  const { token } = useClient();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [seeking, setSeeking] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const progressRef = useRef<HTMLDivElement>(null);
  const shouldAutoPlay = !wasComposeVideoPlayed(videoUrl);

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
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
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
    <div
      className="flex w-full max-w-4xl flex-col gap-0"
      style={{ animation: "render-fade-in 0.6s ease-out both" }}
    >
      {/* top bar */}
      <div className="flex items-center justify-between px-1 pb-3">
        {!hideBack ? (
          <button
            type="button"
            onClick={onDismiss}
            className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-[12px] font-medium text-white/40 transition-colors hover:bg-white/[0.04] hover:text-white/60"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back
          </button>
        ) : (
          <div className="w-16" />
        )}
        <div className="w-16" />
      </div>

      {/* video container */}
      <div className="group/player relative overflow-hidden rounded-xl border border-white/[0.06] bg-black shadow-2xl shadow-black/50">
        <video
          ref={videoRef}
          src={videoUrl}
          autoPlay={shouldAutoPlay}
          className="w-full cursor-pointer object-contain"
          onClick={togglePlay}
          onPlay={() => {
            markComposeVideoPlayed(videoUrl);
            setPlaying(true);
          }}
          onPause={() => setPlaying(false)}
          onTimeUpdate={() => {
            if (!seeking && videoRef.current)
              setCurrentTime(videoRef.current.currentTime);
          }}
          onLoadedMetadata={() => {
            if (videoRef.current) setDuration(videoRef.current.duration);
          }}
        />

        {/* play/pause overlay */}
        <div
          className={cn(
            "absolute inset-0 flex cursor-pointer items-center justify-center bg-black/20 transition-opacity duration-300",
            playing
              ? "opacity-0 group-hover/player:opacity-100"
              : "opacity-100",
          )}
          onClick={togglePlay}
        >
          <div className="grid h-14 w-14 place-items-center rounded-full bg-black/40 backdrop-blur-sm">
            {playing ? (
              <Pause className="h-6 w-6 text-white/80" />
            ) : (
              <Play className="ml-0.5 h-6 w-6 text-white/80" />
            )}
          </div>
        </div>
      </div>

      {/* custom progress bar */}
      <div className="px-1 pt-3">
        <div
          ref={progressRef}
          className="group/bar relative h-1 cursor-pointer rounded-full bg-white/[0.08] transition-all hover:h-1.5"
          onClick={handleSeek}
          onMouseDown={() => setSeeking(true)}
          onMouseUp={() => setSeeking(false)}
        >
          <div
            className="h-full rounded-full bg-white/30 transition-[width] duration-100"
            style={{
              width: duration ? `${(currentTime / duration) * 100}%` : "0%",
            }}
          />
          {/* scrubber dot */}
          <div
            className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/80 opacity-0 shadow-sm transition-opacity group-hover/bar:opacity-100"
            style={{
              left: duration ? `${(currentTime / duration) * 100}%` : "0%",
            }}
          />
        </div>

        {/* controls row */}
        <div className="mt-2 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={togglePlay}
              className="grid h-8 w-8 place-items-center rounded-lg text-white/50 transition-colors hover:bg-white/[0.05] hover:text-white/70"
            >
              {playing ? (
                <Pause className="h-4 w-4" />
              ) : (
                <Play className="ml-0.5 h-4 w-4" />
              )}
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                toggleMute();
              }}
              className="grid h-8 w-8 place-items-center rounded-lg text-white/50 transition-colors hover:bg-white/[0.05] hover:text-white/70"
            >
              {muted ? (
                <VolumeX className="h-4 w-4" />
              ) : (
                <Volume2 className="h-4 w-4" />
              )}
            </button>
            <span className="text-[11px] tabular-nums text-white/30">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onRetry();
                }}
                className={cn(
                  "flex items-center gap-1.5 rounded-lg px-3 py-1.5",
                  "text-[11px] font-medium text-white/40",
                  "transition-colors hover:bg-white/[0.04] hover:text-white/60",
                )}
              >
                <RefreshCcw className="h-3 w-3" />
                {retryLabel}
            </button>
            {onNewChat && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onNewChat();
                }}
                className={cn(
                  "flex items-center gap-1.5 rounded-lg px-3 py-1.5",
                  "text-[11px] font-medium text-white/40",
                  "transition-colors hover:bg-white/[0.04] hover:text-white/60",
                )}
              >
                <MessageSquarePlus className="h-3 w-3" />
                New Project
              </button>
            )}
            <button
              type="button"
              disabled={downloading}
              onClick={(e) => {
                e.stopPropagation();
                void handleDownload();
              }}
              className={cn(
                "flex items-center gap-1.5 rounded-lg px-3 py-1.5",
                "border border-white/[0.08] bg-white/[0.04]",
                "text-[11px] font-medium text-white/60",
                "transition-colors hover:bg-white/[0.07] hover:text-white/80",
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
            {/* <LikeButton
              className="text-white/60 hover:text-white/80"
              echoRequestId={echoRequestId}
              likeStatus={(likeStatus ?? 0) as 0 | 1 | 2}
              onLike={onEchoLike}
            /> */}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Done without URL ── */

function DonePlaceholder({
  onDismiss,
  onRetry,
  copy,
}: {
  onDismiss: () => void;
  onRetry: () => void;
  copy: ReturnType<typeof renderOverlayCopy>;
}) {
  return (
    <div
      className="flex flex-col items-center gap-6 text-center"
      style={{ animation: "render-fade-in 0.5s ease-out both" }}
    >
      <div className="grid h-16 w-16 place-items-center rounded-2xl border border-white/[0.06] bg-white/[0.03]">
        <span className="text-lg font-medium text-white/60">✓</span>
      </div>
      <div>
        <p className="text-[15px] font-medium text-white/85">
          {copy.doneTitle}
        </p>
        <p className="mt-2 text-[12px] text-white/35">
          The video is complete, but no playback URL is available.
        </p>
      </div>
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          onClick={onDismiss}
          className="h-9 rounded-lg px-5 text-[12px] font-medium text-white/50 hover:bg-white/[0.05] hover:text-white/70"
        >
          Back
        </Button>
        <Button
          onClick={onRetry}
          className="h-9 rounded-lg border border-white/[0.06] bg-white/[0.08] px-5 text-[12px] font-medium text-white/80 hover:bg-white/[0.12]"
        >
          <RefreshCcw className="mr-1.5 h-3.5 w-3.5" />
          {copy.retry}
        </Button>
      </div>
    </div>
  );
}
