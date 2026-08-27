"""Resolve browser-reachable video URLs from Echo / algorithm payloads."""

from __future__ import annotations

from typing import Any

_HTTP_PREFIXES = ("http://", "https://")
_STORAGE_ENTRY_KEYS = ("url", "public_url")
_STORAGE_CONTAINER_KEYS = ("asset_urls", "story_asset_urls", "oss_urls")
_DIRECT_URL_KEYS = (
    "result_url",
    "artifact_url",
    "video_url",
    "download_url",
    "output_url",
    "url",
    "final_output_url",
    "story_mp4",
)


def _is_public_http_url(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(_HTTP_PREFIXES):
            return stripped
    return None


def _storage_entry_public_url(entry: Any) -> str | None:
    if isinstance(entry, dict):
        for key in _STORAGE_ENTRY_KEYS:
            url = _is_public_http_url(entry.get(key))
            if url:
                return url
        return None
    return _is_public_http_url(entry)


def _iter_sources(*sources: dict[str, Any] | None) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for source in sources:
        if isinstance(source, dict):
            ordered.append(source)
    return ordered


def resolve_storage_video_url(*sources: dict[str, Any] | None) -> str | None:
    """Return the first public URL from a provider-neutral storage mapping."""
    for source in _iter_sources(*sources):
        for container_key in _STORAGE_CONTAINER_KEYS:
            entries = source.get(container_key)
            if not isinstance(entries, dict):
                continue
            for entry in entries.values():
                url = _storage_entry_public_url(entry)
                if url:
                    return url
    return None


def resolve_direct_video_url(*sources: dict[str, Any] | None) -> str | None:
    """Return the first HTTP(S) URL from direct result fields."""
    for source in _iter_sources(*sources):
        for key in _DIRECT_URL_KEYS:
            url = _is_public_http_url(source.get(key))
            if url:
                return url
    return None


def resolve_public_video_url(*sources: dict[str, Any] | None) -> str | None:
    """Pick a public video URL, preferring configured storage result mappings.

    Never returns PFS or other local filesystem paths — those are not playable
    in the browser and should not be stored as ``artifact_url``.
    """
    return resolve_storage_video_url(*sources) or resolve_direct_video_url(*sources)
