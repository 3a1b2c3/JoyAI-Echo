import { Pause, Play, Volume2, VolumeX } from "lucide-react";
import { useEffect, useId, useRef, useState, type CSSProperties } from "react";

import styles from "./CompactAudioPlayer.module.css";

interface CompactAudioPlayerProps {
  src: string;
  label: string;
  className?: string;
}

export function CompactAudioPlayer({
  src,
  label,
  className,
}: CompactAudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const volumePanelId = useId();

  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [muted, setMuted] = useState(false);
  const [volumeOpen, setVolumeOpen] = useState(false);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onTimeUpdate = () => {
      setProgress(audio.currentTime);
      setDuration(audio.duration || 0);
    };
    const onLoaded = () => setDuration(audio.duration || 0);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onEnded = () => {
      setPlaying(false);
      setProgress(0);
    };

    audio.addEventListener("timeupdate", onTimeUpdate);
    audio.addEventListener("loadedmetadata", onLoaded);
    audio.addEventListener("durationchange", onLoaded);
    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onEnded);

    return () => {
      audio.removeEventListener("timeupdate", onTimeUpdate);
      audio.removeEventListener("loadedmetadata", onLoaded);
      audio.removeEventListener("durationchange", onLoaded);
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onEnded);
    };
  }, [src]);

  useEffect(() => {
    if (!volumeOpen) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setVolumeOpen(false);
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [volumeOpen]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.volume = muted ? 0 : volume;
  }, [muted, volume]);

  const togglePlay = async () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      try {
        await audio.play();
      } catch {
        /* autoplay / decode failures are ignored */
      }
    } else {
      audio.pause();
    }
  };

  const seek = (value: number) => {
    const audio = audioRef.current;
    if (!audio || !Number.isFinite(duration) || duration <= 0) return;
    audio.currentTime = value;
    setProgress(value);
  };

  const effectiveVolume = muted ? 0 : volume;
  const progressPct =
    duration > 0 ? Math.min(100, Math.max(0, (progress / duration) * 100)) : 0;

  return (
    <div
      ref={rootRef}
      className={[styles.player, className].filter(Boolean).join(" ")}
      style={
        {
          "--progress": `${progressPct}%`,
          "--volume": `${effectiveVolume * 100}%`,
        } as CSSProperties
      }
    >
      <audio ref={audioRef} preload="metadata" src={src} aria-label={label} />

      <button
        type="button"
        onClick={togglePlay}
        aria-label={playing ? "Pause" : "Play"}
        className={styles.iconBtn}
      >
        {playing ? (
          <Pause className="size-3.5 fill-current" aria-hidden />
        ) : (
          <Play className="size-3.5 fill-current" aria-hidden />
        )}
      </button>

      <input
        type="range"
        min={0}
        max={duration || 0}
        step={0.01}
        value={duration > 0 ? progress : 0}
        disabled={!duration}
        onChange={(e) => seek(Number(e.target.value))}
        aria-label="Playback progress"
        className={styles.progress}
      />

      <div className={styles.volumeWrap}>
        {volumeOpen ? (
          <div
            id={volumePanelId}
            role="dialog"
            aria-label="Volume"
            className={styles.volumePanel}
          >
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={effectiveVolume}
              onChange={(e) => {
                const next = Number(e.target.value);
                setVolume(next);
                setMuted(next === 0);
              }}
              aria-label="Adjust volume"
              className={styles.volumeSlider}
            />
          </div>
        ) : null}

        <button
          type="button"
          aria-label="Volume"
          aria-expanded={volumeOpen}
          aria-controls={volumeOpen ? volumePanelId : undefined}
          onClick={() => setVolumeOpen((open) => !open)}
          className={styles.iconBtn}
        >
          {effectiveVolume === 0 ? (
            <VolumeX className="size-3.5" aria-hidden />
          ) : (
            <Volume2 className="size-3.5" aria-hidden />
          )}
        </button>
      </div>
    </div>
  );
}
