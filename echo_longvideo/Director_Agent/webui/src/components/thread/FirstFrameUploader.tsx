import { useCallback, useRef, useState } from "react";
import { Loader2, Plus, X } from "lucide-react";
import { useTranslation } from "react-i18next";

import { ImageLightbox } from "@/components/ImageLightbox";
import { cn } from "@/lib/utils";
import type { FirstFrameData } from "@/hooks/useFirstFrameImage";

const ACCEPT_ATTR = "image/png,image/jpeg,image/webp";

interface FirstFrameUploaderProps {
  value: FirstFrameData | null;
  cropping?: boolean;
  disabled?: boolean;
  /** When false, hide the remove control (e.g. locked after generate). */
  clearable?: boolean;
  locked?: boolean;
  onLockedAttempt?: () => void;
  onPickFile: (file: File) => void;
  onClear: () => void;
  className?: string;
}

/** Dashed “+” tile for short-video first-frame reference image (hero only). */
export function FirstFrameUploader({
  value,
  cropping = false,
  disabled = false,
  clearable = true,
  locked = false,
  onLockedAttempt,
  onPickFile,
  onClear,
  className,
}: FirstFrameUploaderProps) {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);
  const [lightboxOpen, setLightboxOpen] = useState(false);

  const openPicker = useCallback(() => {
    if (locked) {
      onLockedAttempt?.();
      return;
    }
    if (disabled || cropping) return;
    inputRef.current?.click();
  }, [cropping, disabled, locked, onLockedAttempt]);

  const onInputChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = "";
      if (file) onPickFile(file);
    },
    [onPickFile],
  );

  if (value) {
    return (
      <>
        {/* Outer wrapper stays overflow-visible so the clear badge can hang outside. */}
        <div className={cn("relative h-16 w-16 shrink-0", className)}>
          <div
            className={cn(
              "relative h-full w-full overflow-hidden rounded-xl",
              "border border-foreground/10 bg-foreground/[0.03]",
            )}
          >
            <button
              type="button"
              onClick={() => setLightboxOpen(true)}
              className="h-full w-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-foreground/20"
              aria-label={t("thread.firstFrame.previewAria")}
            >
              <img
                src={value.previewUrl}
                alt=""
                className="h-full w-full object-cover"
                draggable={false}
              />
            </button>
            {cropping ? (
              <div className="absolute inset-0 flex items-center justify-center bg-background/50">
                <Loader2 className="h-4 w-4 animate-spin text-foreground/70" />
              </div>
            ) : null}
          </div>
          {clearable ? (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onClear();
              }}
              disabled={disabled}
              className={cn(
                "absolute -right-1.5 -top-1.5 z-10 flex h-5 w-5 items-center justify-center",
                "rounded-full bg-foreground text-background shadow-sm",
                "hover:bg-foreground/90 focus-visible:outline-none focus-visible:ring-2",
                "focus-visible:ring-foreground/30 disabled:opacity-50",
              )}
              aria-label={t("thread.firstFrame.remove")}
            >
              <X className="h-3 w-3" aria-hidden />
            </button>
          ) : null}
        </div>
        <ImageLightbox
          images={[{ url: value.previewUrl, name: value.name }]}
          index={lightboxOpen ? 0 : null}
          onIndexChange={() => {}}
          onOpenChange={(open) => {
            if (!open) setLightboxOpen(false);
          }}
        />
      </>
    );
  }

  return (
    <div className={cn("relative shrink-0", className)}>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTR}
        className="sr-only"
        disabled={disabled || cropping || locked}
        onChange={onInputChange}
      />
      <button
        type="button"
        onClick={openPicker}
        disabled={(disabled || cropping) && !locked}
        aria-label={t("thread.firstFrame.uploadAria")}
        className={cn(
          "flex h-16 w-16 items-center justify-center rounded-xl",
          "border border-dashed border-foreground/20 bg-foreground/[0.02]",
          "text-foreground/45 transition-colors",
          "hover:border-foreground/30 hover:bg-foreground/[0.04] hover:text-foreground/60",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-foreground/20",
          "disabled:cursor-not-allowed disabled:opacity-50",
          locked && "cursor-not-allowed opacity-50 hover:border-foreground/20 hover:bg-foreground/[0.02]",
        )}
      >
        {cropping ? (
          <Loader2 className="h-5 w-5 animate-spin" aria-hidden />
        ) : (
          <Plus className="h-5 w-5" aria-hidden />
        )}
      </button>
    </div>
  );
}
