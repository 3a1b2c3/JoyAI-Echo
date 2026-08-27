"""Session source metadata for the Director workflow."""

from __future__ import annotations

from typing import Any

SESSION_SOURCE_KEY = "source"
SOURCE_STEPWISE = "stepwise"

VALID_SOURCES = frozenset({SOURCE_STEPWISE})
DIRECTOR_SKILL_NAME = "director"


def normalize_source(value: Any) -> str | None:
    """Normalize persisted or resolved source strings."""
    if not isinstance(value, str):
        return None
    raw = value.strip().lower()
    if raw in VALID_SOURCES:
        return raw
    return None

def resolve_source_from_wire(data: dict[str, Any] | None) -> str | None:
    """Resolve session source from WebSocket envelope or inbound message metadata."""
    if not isinstance(data, dict):
        return None
    return normalize_source(data.get("source"))

def get_source(metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    return normalize_source(metadata.get(SESSION_SOURCE_KEY))

def apply_source(metadata: dict[str, Any], source: str | None) -> bool:
    """Set source on metadata if not already set. Returns True when updated."""
    normalized = normalize_source(source)
    if not normalized:
        return False
    if get_source(metadata) is not None:
        return False
    metadata[SESSION_SOURCE_KEY] = normalized
    return True
