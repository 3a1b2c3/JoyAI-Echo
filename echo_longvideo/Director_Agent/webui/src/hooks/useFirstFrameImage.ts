import { useCallback, useEffect, useRef, useState } from "react";

import type { VideoSize } from "@/components/thread/AspectRatioPicker";
import { cropImageToVideoSize } from "@/lib/cropImageToVideoSize";

const ACCEPT_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);

export type FirstFrameData = {
  /** Object URL of the *original* image (preview / lightbox). */
  previewUrl: string;
  /** Cropped blob sized to the current videoSize (upload / send). */
  croppedBlob: Blob;
  width: number;
  height: number;
  name: string;
};

export type FirstFrameRejectReason = "unsupported_type" | "decode_failed";

export type UseFirstFrameImageApi = {
  value: FirstFrameData | null;
  cropping: boolean;
  setFile: (file: File) => Promise<boolean>;
  clear: () => void;
};

function isAcceptedImage(file: File): boolean {
  if (ACCEPT_TYPES.has(file.type)) return true;
  // Some browsers omit type for dragged files; sniff by extension.
  const lower = file.name.toLowerCase();
  return (
    lower.endsWith(".png") ||
    lower.endsWith(".jpg") ||
    lower.endsWith(".jpeg") ||
    lower.endsWith(".webp")
  );
}

/** Cropped upload blob is always JPEG; keep FormData filename in sync. */
function toJpegUploadName(name: string | undefined): string {
  const trimmed = name?.trim();
  if (!trimmed) return "first-frame.jpg";
  const base = trimmed.replace(/\.[^.]+$/, "");
  return `${base || "first-frame"}.jpg`;
}

/** Manage a single first-frame reference image with aspect-aware canvas crop. */
export function useFirstFrameImage(
  videoSize: VideoSize,
  onReject?: (reason: FirstFrameRejectReason) => void,
): UseFirstFrameImageApi {
  const [value, setValue] = useState<FirstFrameData | null>(null);
  const [cropping, setCropping] = useState(false);
  const originalRef = useRef<File | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const cropGenRef = useRef(0);
  const onRejectRef = useRef(onReject);
  onRejectRef.current = onReject;

  const revokePreview = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
  }, []);

  const clear = useCallback(() => {
    cropGenRef.current += 1;
    originalRef.current = null;
    revokePreview();
    setValue(null);
    setCropping(false);
  }, [revokePreview]);

  const runCrop = useCallback(
    async (file: File, previewUrl: string) => {
      const gen = ++cropGenRef.current;
      setCropping(true);
      try {
        const cropped = await cropImageToVideoSize(file, videoSize);
        if (gen !== cropGenRef.current) return;
        setValue({
          previewUrl,
          croppedBlob: cropped.blob,
          width: cropped.width,
          height: cropped.height,
          name: toJpegUploadName(file.name),
        });
      } catch {
        if (gen !== cropGenRef.current) return;
        onRejectRef.current?.("decode_failed");
        originalRef.current = null;
        revokePreview();
        setValue(null);
      } finally {
        if (gen === cropGenRef.current) setCropping(false);
      }
    },
    [revokePreview, videoSize],
  );

  const setFile = useCallback(
    async (file: File): Promise<boolean> => {
      if (!isAcceptedImage(file)) {
        onRejectRef.current?.("unsupported_type");
        return false;
      }
      revokePreview();
      const previewUrl = URL.createObjectURL(file);
      previewUrlRef.current = previewUrl;
      originalRef.current = file;
      await runCrop(file, previewUrl);
      return originalRef.current === file;
    },
    [revokePreview, runCrop],
  );

  // Re-crop when the user changes aspect ratio while an image is selected.
  useEffect(() => {
    const file = originalRef.current;
    const previewUrl = previewUrlRef.current;
    if (!file || !previewUrl) return;
    void runCrop(file, previewUrl);
  }, [videoSize.width, videoSize.height, runCrop]);

  useEffect(() => () => revokePreview(), [revokePreview]);

  return { value, cropping, setFile, clear };
}
