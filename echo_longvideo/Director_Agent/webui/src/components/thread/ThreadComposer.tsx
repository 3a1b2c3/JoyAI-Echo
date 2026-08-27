import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { ArrowUp, ImageIcon, Loader2, X } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { CountableTextarea } from "@/components/thread/CountableTextarea";
import {
  DEFAULT_DURATION,
  DurationPicker,
} from "@/components/thread/DurationPicker";
import {
  AspectRatioPicker,
  DEFAULT_VIDEO_SIZE,
  type VideoSize,
} from "@/components/thread/AspectRatioPicker";
import { FirstFrameUploader } from "@/components/thread/FirstFrameUploader";
import {
  DEFAULT_STORY_LANGUAGE,
  LanguagePicker,
  type StoryLanguage,
} from "@/components/thread/LanguagePicker";
import { FIRST_FRAME_UPLOAD_ENABLED } from "@/config/features";
import {
  useAttachedImages,
  type AttachedImage,
  type AttachmentError,
  MAX_IMAGES_PER_MESSAGE,
} from "@/hooks/useAttachedImages";
import type { FirstFrameData } from "@/hooks/useFirstFrameImage";
import { useClipboardAndDrop } from "@/hooks/useClipboardAndDrop";
import type { SendImage } from "@/hooks/useNanobotStream";
import { cn } from "@/lib/utils";

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

interface ThreadComposerProps {
  /** Return false (or reject) to keep the draft and attachments for retry. */
  onSend: (
    content: string,
    images?: SendImage[],
  ) => void | boolean | Promise<void | boolean>;
  disabled?: boolean;
  /** Blocks send (ArrowUp) only; textarea stays editable when false alongside disabled=false. */
  sendDisabled?: boolean;
  placeholder?: string;
  modelLabel?: string | null;
  variant?: "thread" | "hero";
  durationSec?: number;
  onDurationChange?: (sec: number) => void;
  videoSize?: VideoSize;
  onVideoSizeChange?: (size: VideoSize) => void;
  videoSizeLocked?: boolean;
  storyLanguage?: StoryLanguage;
  onStoryLanguageChange?: (language: StoryLanguage) => void;
  /** Generate the remaining shots automatically after story confirmation. */
  autoGenerate?: boolean;
  onAutoGenerateChange?: (enabled: boolean) => void;
  showAutoGenerateToggle?: boolean;
  durationOptions?: readonly number[];
  /** Short-video first-frame uploader (inside composer card, left of textarea). */
  showFirstFrame?: boolean;
  firstFrame?: FirstFrameData | null;
  firstFrameCropping?: boolean;
  firstFrameLocked?: boolean;
  onFirstFramePick?: (file: File) => void;
  onFirstFrameClear?: () => void;
  onFirstFrameLockedAttempt?: () => void;
}

export function ThreadComposer({
  onSend,
  disabled,
  sendDisabled,
  placeholder,
  modelLabel = null,
  variant = "thread",
  durationSec,
  onDurationChange,
  videoSize = DEFAULT_VIDEO_SIZE,
  onVideoSizeChange,
  videoSizeLocked = false,
  storyLanguage = DEFAULT_STORY_LANGUAGE,
  onStoryLanguageChange,
  autoGenerate = false,
  onAutoGenerateChange,
  showAutoGenerateToggle = true,
  durationOptions,
  showFirstFrame = false,
  firstFrame = null,
  firstFrameCropping = false,
  firstFrameLocked = false,
  onFirstFramePick,
  onFirstFrameClear,
  onFirstFrameLockedAttempt,
}: ThreadComposerProps) {
  const { t } = useTranslation();
  const [value, setValue] = useState("");
  const [internalDuration, setInternalDuration] = useState(DEFAULT_DURATION);
  const [inlineError, setInlineError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chipRefs = useRef(new Map<string, HTMLButtonElement>());
  const isHero = variant === "hero";
  const resolvedDuration = durationSec ?? internalDuration;
  const resolvedPlaceholder =
    placeholder ?? t("thread.composer.placeholderThread");

  const { images, enqueue, remove, clear, encoding } = useAttachedImages();

  const formatRejection = useCallback(
    (reason: AttachmentError): string => {
      const key = `thread.composer.imageRejected.${reason}`;
      return t(key, { max: MAX_IMAGES_PER_MESSAGE });
    },
    [t],
  );

  const addFiles = useCallback(
    (files: File[]) => {
      if (files.length === 0) return;
      // First-frame mode: drop/paste replaces the single reference image.
      if (FIRST_FRAME_UPLOAD_ENABLED && showFirstFrame && onFirstFramePick) {
        onFirstFramePick(files[0]!);
        setInlineError(null);
        return;
      }
      if (showFirstFrame && !FIRST_FRAME_UPLOAD_ENABLED) {
        return;
      }
      const { rejected } = enqueue(files);
      if (rejected.length > 0) {
        setInlineError(formatRejection(rejected[0].reason));
      } else {
        setInlineError(null);
      }
    },
    [enqueue, formatRejection, showFirstFrame, onFirstFramePick],
  );

  const { isDragging, onPaste, onDragEnter, onDragOver, onDragLeave, onDrop } =
    useClipboardAndDrop(addFiles);

  useEffect(() => {
    if (disabled) return;
    const el = textareaRef.current;
    if (!el) return;
    const id = requestAnimationFrame(() => el.focus());
    return () => cancelAnimationFrame(id);
  }, [disabled]);

  const readyImages = useMemo(
    () =>
      images.filter(
        (img): img is AttachedImage & { dataUrl: string } =>
          img.status === "ready" && typeof img.dataUrl === "string",
      ),
    [images],
  );
  const hasErrors = images.some((img) => img.status === "error");

  const canSend =
    !disabled &&
    !sendDisabled &&
    !submitting &&
    !encoding &&
    !hasErrors &&
    (value.trim().length > 0 || readyImages.length > 0);

  const handleDurationChange = useCallback(
    (sec: number) => {
      if (durationSec === undefined) {
        setInternalDuration(sec);
      }
      onDurationChange?.(sec);
    },
    [durationSec, onDurationChange],
  );

  const submit = useCallback(async () => {
    if (!canSend) return;
    const trimmed = value.trim();
    // Share the same normalized ``data:`` URL with both the wire payload and
    // the optimistic bubble preview: data URLs are self-contained (no blob
    // lifetime, safe under React StrictMode double-mount) and keep the
    // bubble in sync with whatever the backend actually sees.
    const payload: SendImage[] | undefined =
      readyImages.length > 0
        ? readyImages.map((img) => ({
            media: {
              data_url: img.dataUrl,
              name: img.file.name,
            },
            preview: { url: img.dataUrl, name: img.file.name },
          }))
        : undefined;
    setSubmitting(true);
    try {
      const accepted = await onSend(trimmed, payload);
      if (accepted === false) return;
      setValue("");
      setInlineError(null);
      // Bubble owns the data URL copy; safe to revoke every staged blob
      // preview here without affecting the rendered message.
      clear();
      requestAnimationFrame(() => {
        const el = textareaRef.current;
        if (el) {
          el.style.height = "auto";
          el.focus();
        }
      });
    } catch {
      // Keep the complete draft intact so the user can retry.
    } finally {
      setSubmitting(false);
    }
  }, [canSend, clear, onSend, readyImages, value]);

  const onKeyDown = (_e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    // Enter inserts newline; send via submit button only.
  };

  const removeChip = useCallback(
    (id: string) => {
      const { nextFocusId } = remove(id);
      setInlineError(null);
      requestAnimationFrame(() => {
        const el = nextFocusId ? chipRefs.current.get(nextFocusId) : null;
        if (el) {
          el.focus();
        } else {
          textareaRef.current?.focus();
        }
      });
    },
    [remove],
  );

  const onChipKey = useCallback(
    (id: string) => (e: ReactKeyboardEvent<HTMLButtonElement>) => {
      if (
        e.key === "Delete" ||
        e.key === "Backspace" ||
        e.key === "Enter" ||
        e.key === " "
      ) {
        e.preventDefault();
        removeChip(id);
      }
    },
    [removeChip],
  );

  const textarea = (
    <CountableTextarea
      ref={textareaRef}
      value={value}
      onChange={setValue}
      onKeyDown={onKeyDown}
      onPaste={onPaste}
      rows={1}
      placeholder={resolvedPlaceholder}
      disabled={disabled}
      aria-label={t("thread.composer.inputAria")}
      showCount={false}
      countPosition="bottom-right"
      formatCount={(count) => t("thread.composer.charCount", { count })}
      className={cn(
        "w-full resize-none bg-transparent",
        showFirstFrame
          ? "min-h-[72px] px-1 pb-1 pt-1 text-[15px] leading-6"
          : isHero
            ? "min-h-[96px] px-4 pb-2 pt-4 text-[15px] leading-6"
            : "min-h-[50px] px-4 pb-1.5 pt-3 text-[15px]",
        "placeholder:text-muted-foreground",
        "focus:outline-none focus-visible:outline-none",
        "disabled:cursor-not-allowed",
      )}
    />
  );

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        void submit();
      }}
      onDragEnter={onDragEnter}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      className={cn("w-full", isHero ? "px-0" : "px-1 pb-1.5 pt-1 sm:px-0")}
    >
      <div
        className={cn(
          "relative mx-auto flex w-full flex-col overflow-hidden transition-all duration-200",
          isHero
            ? "max-w-[40rem] rounded-[24px] border border-border/75 bg-card shadow-[0_10px_30px_rgba(0,0,0,0.10)]"
            : "max-w-[49.5rem] rounded-[16px] border border-border/70 bg-card",
          "focus-within:ring-1 focus-within:ring-foreground/8",
          disabled && "opacity-60",
          isDragging &&
            "ring-2 ring-primary/40 motion-reduce:ring-0 motion-reduce:border-primary",
        )}
      >
        {images.length > 0 && !showFirstFrame ? (
          <div
            className="flex flex-wrap gap-2 px-3 pt-3"
            aria-label={t("thread.composer.attachImage")}
          >
            {images.map((img) => (
              <AttachmentChip
                key={img.id}
                image={img}
                labelRemove={t("thread.composer.remove")}
                labelEncoding={t("thread.composer.encoding")}
                normalizedHint={(orig, current) =>
                  t("thread.composer.normalizedSizeHint", {
                    orig: formatBytes(orig),
                    current: formatBytes(current),
                  })
                }
                formatError={formatRejection}
                onRemove={() => removeChip(img.id)}
                onKeyDown={onChipKey(img.id)}
                registerRef={(el) => {
                  if (el) chipRefs.current.set(img.id, el);
                  else chipRefs.current.delete(img.id);
                }}
              />
            ))}
          </div>
        ) : null}

        {showFirstFrame ? (
          <div
            className={cn(
              "flex items-start gap-3 px-3.5",
              "pt-2",
            )}
          >
            <FirstFrameUploader
              value={firstFrame}
              cropping={firstFrameCropping}
              disabled={disabled || !FIRST_FRAME_UPLOAD_ENABLED}
              clearable={!firstFrameLocked}
              locked={firstFrameLocked}
              onLockedAttempt={onFirstFrameLockedAttempt}
              onPickFile={onFirstFramePick ?? (() => {})}
              onClear={onFirstFrameClear ?? (() => {})}
            />
            <div className="min-w-0 flex-1">{textarea}</div>
          </div>
        ) : (
          textarea
        )}
        {inlineError ? (
          <div
            role="alert"
            className={cn(
              "mx-3 mb-1 rounded-md border border-destructive/40 bg-destructive/8 px-2.5 py-1",
              "text-[11.5px] font-medium text-destructive",
            )}
          >
            {inlineError}
          </div>
        ) : null}
        <div
          className={cn(
            "flex items-center justify-between gap-2",
            isHero ? "px-3.5 pb-3.5" : "px-3 pb-2",
          )}
        >
          <div className="flex min-w-0 items-center gap-2">
            {modelLabel ? (
              <span
                title={modelLabel}
                className={cn(
                  "inline-flex min-w-0 items-center gap-1.5 rounded-full border px-2.5 py-1",
                  "border-foreground/10 bg-foreground/[0.035] font-medium text-foreground/80",
                  isHero ? "text-[11px]" : "text-[12px]",
                )}
              >
                <span className="truncate">Echo Director</span>
              </span>
            ) : null}
            <>
              <AspectRatioPicker
                value={videoSize}
                onChange={onVideoSizeChange ?? (() => {})}
                disabled={disabled || videoSizeLocked}
                size={isHero ? "hero" : "thread"}
              />
              {videoSizeLocked ? (
                <span className="text-[10px] text-muted-foreground/50">
                  Size locked
                </span>
              ) : null}
              <LanguagePicker
                value={storyLanguage}
                onChange={onStoryLanguageChange ?? (() => {})}
                disabled={disabled}
                size={isHero ? "hero" : "thread"}
              />
              {autoGenerate ? (
                <DurationPicker
                  value={resolvedDuration}
                  onChange={handleDurationChange}
                  disabled={disabled}
                  size={isHero ? "hero" : "thread"}
                  options={durationOptions}
                />
              ) : null}
              {showAutoGenerateToggle ? (
                <div className="flex justify-end">
                    <button
                      type="button"
                      role="switch"
                      aria-checked={autoGenerate}
                      disabled={disabled}
                      onClick={() => onAutoGenerateChange?.(!autoGenerate)}
                      className="inline-flex select-none items-center gap-1.5 rounded-full border border-foreground/10 bg-foreground/[0.035] px-2.5 py-1 text-[11px] text-foreground/75 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <span
                        className={cn(
                          "relative inline-block h-3.5 w-6 shrink-0 overflow-hidden rounded-full transition-colors",
                          autoGenerate
                            ? "bg-foreground/70"
                            : "bg-foreground/15",
                          "break-keep",
                        )}
                      >
                        <span
                          className={cn(
                            "absolute left-0.5 top-0.5 h-2.5 w-2.5 rounded-full bg-background shadow-sm transition-transform",
                            autoGenerate ? "translate-x-2.5" : "translate-x-0",
                          )}
                        />
                      </span>
                      <span className="break-keep">
                        {t("thread.composer.autoGenerate")}
                      </span>
                    </button>
                </div>
              ) : null}
            </>
            <span className="hidden select-none text-[12px] text-muted-foreground/60 sm:inline">
              {t("thread.composer.sendHint")}
            </span>
          </div>
          <span className="sm:hidden" aria-hidden />
          <Button
            type="submit"
            size="icon"
            disabled={!canSend}
            aria-label={t("thread.composer.send")}
            className={cn(
              "rounded-full border border-border/70 bg-secondary/85 text-secondary-foreground shadow-none transition-transform hover:bg-accent",
              isHero ? "h-8.5 w-8.5" : "h-7.5 w-7.5",
              canSend && "hover:scale-[1.03] active:scale-95",
            )}
          >
            {submitting ? (
              <Loader2
                className={cn(
                  "animate-spin motion-reduce:animate-none",
                  isHero ? "h-4.5 w-4.5" : "h-4 w-4",
                )}
              />
            ) : (
              <ArrowUp className={cn(isHero ? "h-4.5 w-4.5" : "h-4 w-4")} />
            )}
          </Button>
        </div>
      </div>
    </form>
  );
}

interface AttachmentChipProps {
  image: AttachedImage;
  labelRemove: string;
  labelEncoding: string;
  normalizedHint: (origBytes: number, currentBytes: number) => string;
  formatError: (reason: AttachmentError) => string;
  onRemove: () => void;
  onKeyDown: (e: ReactKeyboardEvent<HTMLButtonElement>) => void;
  registerRef: (el: HTMLButtonElement | null) => void;
}

function AttachmentChip({
  image,
  labelRemove,
  labelEncoding,
  normalizedHint,
  formatError,
  onRemove,
  onKeyDown,
  registerRef,
}: AttachmentChipProps) {
  const sizeLabel =
    image.status === "ready" && image.normalized && image.encodedBytes
      ? normalizedHint(image.file.size, image.encodedBytes)
      : formatBytes(image.file.size);
  const tone =
    image.status === "error"
      ? "border-destructive/40 bg-destructive/5 text-destructive"
      : "border-border/70 bg-muted/60";

  return (
    <div
      className={cn(
        "group relative flex items-center gap-2 rounded-[12px] border px-2 py-1.5",
        "transition-colors motion-reduce:transition-none",
        tone,
      )}
      data-testid="composer-chip"
    >
      <div className="relative h-10 w-10 overflow-hidden rounded-md bg-background">
        {image.previewUrl ? (
          <img
            src={image.previewUrl}
            alt=""
            aria-hidden
            loading="eager"
            draggable={false}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <ImageIcon className="h-4 w-4 text-muted-foreground" aria-hidden />
          </div>
        )}
        {image.status === "encoding" ? (
          <div
            className="absolute inset-0 flex items-center justify-center bg-background/60"
            aria-label={labelEncoding}
          >
            <Loader2
              className="h-4 w-4 animate-spin motion-reduce:animate-none"
              aria-hidden
            />
          </div>
        ) : null}
      </div>
      <div className="flex min-w-0 flex-col text-[11.5px] leading-4">
        <span
          className="truncate max-w-[14rem] font-medium"
          title={image.file.name}
        >
          {image.file.name}
        </span>
        <span className="truncate text-muted-foreground">
          {image.status === "error" && image.error
            ? formatError(image.error)
            : sizeLabel}
        </span>
      </div>
      <button
        type="button"
        ref={registerRef}
        onClick={onRemove}
        onKeyDown={onKeyDown}
        aria-label={labelRemove}
        className={cn(
          "ml-1 grid h-5 w-5 flex-none place-items-center rounded-full",
          "text-muted-foreground/80 hover:bg-foreground/8 hover:text-foreground",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-foreground/30",
        )}
      >
        <X className="h-3.5 w-3.5" aria-hidden />
      </button>
    </div>
  );
}
