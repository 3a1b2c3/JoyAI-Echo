"""Stepwise director auto-generate flag, orthogonal to session ``source``."""

from __future__ import annotations

from typing import Any

SESSION_AUTO_GENERATE_KEY = "auto_generate"
DEFAULT_AUTO_GENERATE_DURATION_SEC = 30


def coerce_bool(value: Any) -> bool | None:
    """Parse wire/session booleans. Returns None when the value is absent/unknown."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off", ""}:
            return False
    return None


def resolve_auto_generate_from_wire(data: dict[str, Any] | None) -> bool | None:
    """Read ``autoGenerate`` / ``auto_generate`` from a WS envelope or HTTP body."""
    if not isinstance(data, dict):
        return None
    if "autoGenerate" in data:
        return coerce_bool(data.get("autoGenerate"))
    if "auto_generate" in data:
        return coerce_bool(data.get("auto_generate"))
    return None


def get_auto_generate(metadata: dict[str, Any] | None) -> bool:
    if not isinstance(metadata, dict):
        return False
    parsed = coerce_bool(metadata.get(SESSION_AUTO_GENERATE_KEY))
    return bool(parsed)


def apply_auto_generate(metadata: dict[str, Any], enabled: bool | None) -> bool:
    """Set ``auto_generate`` on session metadata. Returns True when changed."""
    if not isinstance(metadata, dict) or enabled is None:
        return False
    current = get_auto_generate(metadata)
    if current == enabled and SESSION_AUTO_GENERATE_KEY in metadata:
        return False
    metadata[SESSION_AUTO_GENERATE_KEY] = bool(enabled)
    return True


def resolve_duration_sec_from_wire(data: dict[str, Any] | None) -> Any:
    if not isinstance(data, dict):
        return None
    if "durationSec" in data:
        return data.get("durationSec")
    if "duration_sec" in data:
        return data.get("duration_sec")
    return None


def shot_count_for_auto_generate(duration_sec: Any) -> int:
    """Map whole-video duration to shot count. Reject unknown tiers."""
    from nanobot.session.generation_settings import VALID_DURATIONS, duration_to_n_shots

    try:
        parsed = int(duration_sec)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid auto_generate duration_sec={duration_sec!r}") from exc
    mapped = duration_to_n_shots(parsed)
    if mapped is None:
        raise ValueError(
            f"unsupported auto_generate duration_sec={parsed}; "
            f"valid={sorted(VALID_DURATIONS)}"
        )
    return mapped


def default_auto_generate_shot_count() -> int:
    return shot_count_for_auto_generate(DEFAULT_AUTO_GENERATE_DURATION_SEC)


def locked_shot_count_from_goal(goal: Any) -> int | None:
    """Return a positive locked ``goal.shot_count``, or None when unset/invalid."""
    if not isinstance(goal, dict):
        return None
    raw = goal.get("shot_count")
    try:
        parsed = int(raw) if raw not in (None, "") else 0
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def locked_shot_count_from_state(state: Any) -> int | None:
    if not isinstance(state, dict):
        return None
    return locked_shot_count_from_goal(state.get("goal"))


def session_n_shots(metadata: dict[str, Any] | None) -> int | None:
    """Read the raw session ``n_shots`` value without applying defaults."""
    if not isinstance(metadata, dict):
        return None
    from nanobot.session.generation_settings import SESSION_NSHOT_KEY

    raw = metadata.get(SESSION_NSHOT_KEY)
    try:
        parsed = int(raw) if raw not in (None, "") else 0
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def effective_auto_generate_shot_count(
    *,
    goal: Any = None,
    state: Any = None,
    metadata: dict[str, Any] | None = None,
) -> int | None:
    """Prefer the locked workplace shot count over session generation settings."""
    locked = locked_shot_count_from_goal(goal)
    if locked is None:
        locked = locked_shot_count_from_state(state)
    if locked is not None:
        return locked
    return session_n_shots(metadata)
