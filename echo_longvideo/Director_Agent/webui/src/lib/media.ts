import type { UIImage, UIVideo } from "@/lib/types";

export interface WireMediaRef {
  url: string;
  name?: string;
}

const IMAGE_EXT_RE = /\.(avif|bmp|gif|jpe?g|png|webp)(?:$|[?#])/i;
const VIDEO_EXT_RE = /\.(m4v|mov|mp4|og[gv]|webm)(?:$|[?#])/i;

function mediaLeaf(url: string): string | undefined {
  if (!url) return undefined;
  try {
    const parsed = new URL(url, "http://localhost");
    const parts = parsed.pathname.split("/");
    return decodeURIComponent(parts[parts.length - 1] || "") || undefined;
  } catch {
    const parts = url.split("/");
    return parts[parts.length - 1] || undefined;
  }
}

export function inferMediaKind(
  url?: string,
  name?: string,
): "image" | "video" | null {
  const candidates = [name ?? "", url ?? ""];
  if (candidates.some((candidate) => IMAGE_EXT_RE.test(candidate))) return "image";
  if (candidates.some((candidate) => VIDEO_EXT_RE.test(candidate))) return "video";
  return null;
}

function mediaName(ref: WireMediaRef): string | undefined {
  return ref.name || mediaLeaf(ref.url);
}

export function splitMediaByKind(
  refs?: WireMediaRef[] | null,
): {
  images?: UIImage[];
  videos?: UIVideo[];
} {
  const images: UIImage[] = [];
  const videos: UIVideo[] = [];
  for (const ref of refs ?? []) {
    if (!ref?.url) continue;
    const name = mediaName(ref);
    const kind = inferMediaKind(ref.url, name);
    if (kind === "image") {
      images.push({ url: ref.url, name });
    } else if (kind === "video") {
      videos.push({ url: ref.url, name });
    }
  }
  return {
    ...(images.length > 0 ? { images } : {}),
    ...(videos.length > 0 ? { videos } : {}),
  };
}

const COMPOSE_PLAYED_PREFIX = "echo:compose-played:";

/** True if this tab already auto-played (or user-played) the compose video. */
export function wasComposeVideoPlayed(url: string): boolean {
  if (!url) return false;
  try {
    return sessionStorage.getItem(COMPOSE_PLAYED_PREFIX + url) === "1";
  } catch {
    return false;
  }
}

export function markComposeVideoPlayed(url: string): void {
  if (!url) return;
  try {
    sessionStorage.setItem(COMPOSE_PLAYED_PREFIX + url, "1");
  } catch {
    // private mode / quota
  }
}

export function wireMediaRefs(
  mediaUrls?: WireMediaRef[] | null,
  fallbackUrls?: string[] | null,
): WireMediaRef[] {
  if (Array.isArray(mediaUrls) && mediaUrls.length > 0) {
    return mediaUrls.filter((item): item is WireMediaRef => !!item?.url);
  }
  return (fallbackUrls ?? [])
    .filter((url): url is string => typeof url === "string" && url.length > 0)
    .map((url) => ({ url, name: mediaLeaf(url) }));
}
