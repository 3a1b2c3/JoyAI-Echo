"""Strip LLM-only inject prefixes from user-visible session content.

Gate / story-confirm notes are prepended onto the inbound user text so the
current Agent turn can follow them. They must not be persisted or shown in
the chat thread / sidebar preview.
"""

from __future__ import annotations

AGENT_INJECT_STARTS = (
    "REFERENCE_IMAGE_GATE ",
    "STORY_DIRECTION_CONFIRM ",
)


def is_agent_inject_user_text(content: str) -> bool:
    text = (content or "").lstrip()
    return any(text.startswith(prefix) for prefix in AGENT_INJECT_STARTS)


def visible_user_content(content: str) -> str:
    """Return the user-facing remainder after an inject prefix, else *content*."""
    text = content if isinstance(content, str) else ""
    stripped = text.strip()
    if not is_agent_inject_user_text(stripped):
        return text
    parts = stripped.split("\n\n", 1)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines[-1]
    return ""
