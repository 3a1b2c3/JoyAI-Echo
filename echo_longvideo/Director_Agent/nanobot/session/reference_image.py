"""First-frame reference image persisted on session metadata and workplace state."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

SESSION_REFERENCE_IMAGE_KEY = "reference_image"
SESSION_REFERENCE_IMAGE_LOCKED_KEY = "reference_image_locked"
SESSION_REFERENCE_IMAGE_REWRITE_KEY = "reference_image_needs_story_rewrite"
SESSION_STORY_REWRITE_SUPPRESS_KEY = "story_rewrite_suppressed"

STORY_CONFIRM_INJECT_PREFIX = (
    "STORY_DIRECTION_CONFIRM lock=true. The user accepted the CURRENT story_md "
    "on the confirmation card. Ignore reference_image_needs_story_rewrite even if "
    "it is true. Do NOT rewrite plot, characters, wardrobe, or setting. Call "
    "write_story with the existing story_md and confirmed=true. Then ask shot "
    "count via ask_user. Do not show the story-direction card again."
)

_BLOCKED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


def normalize_reference_image(payload: Any) -> dict[str, Any] | None:
    """Return a stored reference-image dict, or None when empty/invalid."""
    if not isinstance(payload, dict):
        return None
    url = (
        payload.get("url")
        or payload.get("reference_image_url")
        or payload.get("referenceImageUrl")
    )
    if not isinstance(url, str) or not url.strip():
        return None
    name = (
        payload.get("name")
        or payload.get("reference_image_name")
        or payload.get("referenceImageName")
        or ""
    )
    width = (
        payload.get("width")
        or payload.get("reference_image_width")
        or payload.get("referenceImageWidth")
        or 0
    )
    height = (
        payload.get("height")
        or payload.get("reference_image_height")
        or payload.get("referenceImageHeight")
        or 0
    )
    try:
        width_i = int(width or 0)
    except (TypeError, ValueError):
        width_i = 0
    try:
        height_i = int(height or 0)
    except (TypeError, ValueError):
        height_i = 0
    return {
        "url": url.strip(),
        "name": name.strip() if isinstance(name, str) else "",
        "width": width_i,
        "height": height_i,
    }


def reference_image_present(payload: Any) -> bool:
    ref = normalize_reference_image(payload)
    return bool(ref and ref.get("url"))


def reference_image_url(payload: Any) -> str:
    ref = normalize_reference_image(payload)
    return str(ref.get("url") or "").strip() if ref else ""


def story_rewrite_suppressed(metadata: Any) -> bool:
    return bool(
        isinstance(metadata, dict) and metadata.get(SESSION_STORY_REWRITE_SUPPRESS_KEY)
    )


def suppress_story_rewrite(metadata: dict[str, Any]) -> None:
    """Hide rewrite from the agent until the user picks 我想修改/增删参考图."""
    metadata[SESSION_STORY_REWRITE_SUPPRESS_KEY] = True


def allow_story_rewrite(metadata: dict[str, Any]) -> None:
    metadata.pop(SESSION_STORY_REWRITE_SUPPRESS_KEY, None)


def apply_story_direction_answer(metadata: dict[str, Any], answer: str) -> str | None:
    """Confirm locks the current story; edit-image re-enables rewrite.

    Returns an agent inject note for 可以，按这个来; otherwise None.
    """
    from nanobot.agent.tools.ask_user import (
        is_reference_image_edit_option,
        is_story_confirm_option,
    )

    if is_story_confirm_option(answer):
        suppress_story_rewrite(metadata)
        return STORY_CONFIRM_INJECT_PREFIX
    if is_reference_image_edit_option(answer):
        allow_story_rewrite(metadata)
    return None


def reference_image_needs_story_rewrite(metadata: Any) -> bool:
    if story_rewrite_suppressed(metadata):
        return False
    return bool(
        isinstance(metadata, dict) and metadata.get(SESSION_REFERENCE_IMAGE_REWRITE_KEY)
    )


def mark_reference_image_needs_story_rewrite(
    metadata: dict[str, Any],
    *,
    previous_url: str,
    next_url: str,
) -> bool:
    """Flag a rewrite when an existing first-frame is replaced or removed.

    The flag is still stored while suppress is on so a later 我想修改/增删参考图
    can see the change; reads stay hidden until then.
    """
    prev = (previous_url or "").strip()
    nxt = (next_url or "").strip()
    if prev == nxt or not prev:
        return False
    metadata[SESSION_REFERENCE_IMAGE_REWRITE_KEY] = True
    return True


def clear_reference_image_needs_story_rewrite(metadata: Any) -> None:
    if isinstance(metadata, dict):
        metadata.pop(SESSION_REFERENCE_IMAGE_REWRITE_KEY, None)


def story_reference_image_inject_note(*, replaced: bool) -> str:
    """Vision prompt attached with the current first-frame image."""
    note = (
        "The attached image is the user-uploaded CURRENT first-frame reference. "
        "Write the screenplay and shot-1 opening so they continue from this image as frame 0."
    )
    if replaced:
        return (
            note
            + " This image REPLACES a previous first-frame. Discard the previous "
            "screenplay, characters, wardrobe, and setting unless they are visible "
            "in THIS image. Do not paraphrase the old story. The confirmation card "
            "must describe this new screenplay."
        )
    return note


def is_blocked_local_url(url: str) -> bool:
    """True when *url* is a local/file locator that must not be stored as a public address."""
    raw = (url or "").strip()
    if not raw:
        return True
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme in {"file", ""}:
        return True
    host = (parsed.hostname or "").lower()
    if host in _BLOCKED_HOSTS:
        return True
    if host.startswith("127.") or host.endswith(".localhost"):
        return True
    return False


def is_reference_image_locked(state: dict[str, Any] | None) -> bool:
    """Lock after shot_count is confirmed (PDF: 确认完分镜数后不可改)."""
    if not isinstance(state, dict):
        return False
    if state.get(SESSION_REFERENCE_IMAGE_LOCKED_KEY) is True:
        return True
    goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
    try:
        shot_count = int(goal.get("shot_count") or 0)
    except (TypeError, ValueError):
        shot_count = 0
    return shot_count > 0


def lock_reference_image(state: dict[str, Any]) -> None:
    state[SESSION_REFERENCE_IMAGE_LOCKED_KEY] = True


def unlock_reference_image(state: dict[str, Any]) -> None:
    state[SESSION_REFERENCE_IMAGE_LOCKED_KEY] = False


def download_reference_image_data_uri(url: str) -> str:
    """Return a reference image as a data URI for LLM vision."""
    if url.startswith("data:image/"):
        return url
    from nanobot.security.http_download import HttpDownloadError, download_http_bytes
    from nanobot.security.url_validator import UrlValidationError, validate_external_url

    if is_blocked_local_url(url):
        raise ValueError("reference image url must be a public HTTP(S) address")
    try:
        validated = validate_external_url(url)
    except UrlValidationError as exc:
        raise ValueError(str(exc)) from exc
    try:
        result = download_http_bytes(validated, max_bytes=20 * 1024 * 1024, timeout_s=15.0)
    except HttpDownloadError as exc:
        raise ValueError(str(exc)) from exc
    mime_type = (result.content_type or "").split(";", 1)[0].strip()
    if (not mime_type) or mime_type == "application/octet-stream":
        from os.path import splitext
        from urllib.parse import urlparse

        ext = splitext(urlparse(validated).path)[1].lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }.get(ext, "image/jpeg")
    import base64

    b64 = base64.b64encode(result.data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"
