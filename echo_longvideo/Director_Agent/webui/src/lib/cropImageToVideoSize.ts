/** Center-crop an image to a target aspect ratio, then scale to pixel size. */

export type VideoSize = { width: number; height: number };

export type CropRect = {
  sx: number;
  sy: number;
  sw: number;
  sh: number;
};

/** Largest centered source rect that matches ``targetW/targetH`` aspect. */
export function computeCenterCropRect(
  srcWidth: number,
  srcHeight: number,
  targetWidth: number,
  targetHeight: number,
): CropRect {
  if (srcWidth <= 0 || srcHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) {
    return { sx: 0, sy: 0, sw: Math.max(0, srcWidth), sh: Math.max(0, srcHeight) };
  }
  const targetRatio = targetWidth / targetHeight;
  const srcRatio = srcWidth / srcHeight;
  if (srcRatio > targetRatio) {
    const sw = srcHeight * targetRatio;
    return { sx: (srcWidth - sw) / 2, sy: 0, sw, sh: srcHeight };
  }
  const sh = srcWidth / targetRatio;
  return { sx: 0, sy: (srcHeight - sh) / 2, sw: srcWidth, sh };
}

function loadBitmap(source: Blob): Promise<ImageBitmap> {
  if (typeof createImageBitmap === "function") {
    return createImageBitmap(source);
  }
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(source);
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        URL.revokeObjectURL(url);
        reject(new Error("canvas unavailable"));
        return;
      }
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      createImageBitmap(canvas).then(resolve, reject);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("decode_failed"));
    };
    img.src = url;
  });
}

function blobFromCanvas(
  canvas: HTMLCanvasElement,
  type: string,
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("toBlob failed"));
          return;
        }
        resolve(blob);
      },
      type,
      0.92,
    );
  });
}

/**
 * Center-crop ``source`` to ``target`` aspect ratio and draw at target pixels.
 * Resizes when the source is smaller than the target.
 */
export async function cropImageToVideoSize(
  source: File | Blob,
  target: VideoSize,
): Promise<{ blob: Blob; width: number; height: number }> {
  const bitmap = await loadBitmap(source);
  try {
    const { sx, sy, sw, sh } = computeCenterCropRect(
      bitmap.width,
      bitmap.height,
      target.width,
      target.height,
    );
    const canvas = document.createElement("canvas");
    canvas.width = target.width;
    canvas.height = target.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("canvas unavailable");
    }
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, target.width, target.height);

    // Always JPEG so dense PNG/WebP sources stay under gateway body limits.
    const blob = await blobFromCanvas(canvas, "image/jpeg");
    return { blob, width: target.width, height: target.height };
  } finally {
    bitmap.close?.();
  }
}
