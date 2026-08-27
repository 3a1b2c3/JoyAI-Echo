"""WebUI session key helpers for the local browser gateway."""

from __future__ import annotations

WEBUI_CHANNEL = "websocket"
_PREFIX = f"{WEBUI_CHANNEL}:"


class WebuiSessionKeyError(ValueError):
    """Raised when a WebUI session key does not meet the persisted key shape."""


def webui_session_key(user_id: str | None, chat_id: str) -> str:
    """Build a browser session key (normally ``websocket:local:<chat>``)."""
    if not user_id or not str(user_id).strip():
        raise WebuiSessionKeyError("user_id is required for WebUI session keys")
    if not chat_id or not str(chat_id).strip():
        raise WebuiSessionKeyError("chat_id is required for WebUI session keys")
    return f"{WEBUI_CHANNEL}:{user_id}:{chat_id}"


def is_legacy_two_part_webui_session_key(key: str) -> bool:
    """Return True for deprecated ``websocket:<chat_id>`` keys (no user segment)."""
    if not key.startswith(_PREFIX):
        return False
    rest = key[len(_PREFIX) :]
    return bool(rest) and ":" not in rest


def parse_webui_session_key(key: str) -> tuple[str, str] | None:
    """Return ``(user_id, chat_id)`` for valid three-part keys; None otherwise."""
    if is_legacy_two_part_webui_session_key(key):
        return None
    if not key.startswith(_PREFIX):
        return None
    rest = key[len(_PREFIX) :]
    if not rest or ":" not in rest:
        return None
    user_id, _, chat_id = rest.partition(":")
    if not user_id or not chat_id:
        return None
    return user_id, chat_id


def webui_wire_chat_id(session_key: str) -> str:
    """Extract the wire ``chat_id`` (uuid) from a persisted session key."""
    parsed = parse_webui_session_key(session_key)
    return parsed[1] if parsed else ""


def session_workspace_dir_name(session_key: str) -> str:
    """Map a persisted session key to a workspace subdirectory name.

    ``websocket:local:<chatId>`` becomes ``websocket_local_<chatId>``,
    matching how WebUI sessions are laid out on disk.
    """
    return session_key.replace(":", "_").replace("/", "_")
