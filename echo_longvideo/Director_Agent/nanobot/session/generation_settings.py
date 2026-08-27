"""Director generation settings: duration, shot count, canvas, and language."""

from __future__ import annotations

from typing import Any

_UNSET = object()

SESSION_NSHOT_KEY = "n_shots"
SESSION_DURATION_KEY = "duration_sec"
SESSION_VIDEO_WIDTH_KEY = "video_width"
SESSION_VIDEO_HEIGHT_KEY = "video_height"
SESSION_LANGUAGE_KEY = "language"
SESSION_LLM_TEMPERATURE_KEY = "llm_temperature"
SESSION_LLM_TOP_P_KEY = "llm_top_p"
SESSION_LLM_TOP_K_KEY = "llm_top_k"

DEFAULT_NSHOT = 1
DEFAULT_DURATION_SEC = 10
DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 736
DEFAULT_LANGUAGE = "zh"

# UI / story_profile.language values
LANGUAGE_ZH = "zh"
LANGUAGE_EN = "en"
VALID_LANGUAGES = frozenset({LANGUAGE_ZH, LANGUAGE_EN})

# story_profile.dialogue_language values used by PE / shot prompts
DIALOGUE_LANGUAGE_BY_LANGUAGE: dict[str, str] = {
    LANGUAGE_ZH: "Mandarin Chinese",
    LANGUAGE_EN: "English",
}
CAPTION_LANGUAGE_BY_LANGUAGE: dict[str, str] = {
    LANGUAGE_ZH: "Simplified Chinese",
    LANGUAGE_EN: "English",
}

# duration_sec → n_shots
DURATION_TO_NSHOT: dict[int, int] = {
    10: 1,
    20: 2,
    30: 3,
    60: 6,
    90: 9,
    120: 12,
    150: 15,
    180: 18,
}
NSHOT_TO_DURATION: dict[int, int] = {n: d for d, n in DURATION_TO_NSHOT.items()}
VALID_DURATIONS = frozenset(DURATION_TO_NSHOT)
VALID_NSHOTS = frozenset(NSHOT_TO_DURATION)


def duration_to_n_shots(duration_sec: int) -> int | None:
    return DURATION_TO_NSHOT.get(int(duration_sec))


def n_shots_to_duration(n_shots: int) -> int | None:
    return NSHOT_TO_DURATION.get(int(n_shots))


def normalize_duration_sec(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed in VALID_DURATIONS else None


def normalize_n_shots(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed in VALID_NSHOTS else None


def normalize_language(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if cleaned in VALID_LANGUAGES:
        return cleaned
    lowered = cleaned.lower()
    if lowered in {
        "zh",
        "zh-cn",
        "zh_cn",
        "chinese",
        "mandarin",
        "mandarin chinese",
        "中文",
    }:
        return LANGUAGE_ZH
    if lowered in {"en", "en-us", "en_us", "english"}:
        return LANGUAGE_EN
    return None


def language_to_dialogue_language(language: str | None) -> str | None:
    if not language:
        return None
    return DIALOGUE_LANGUAGE_BY_LANGUAGE.get(language)


def language_to_caption_language(language: str | None) -> str | None:
    if not language:
        return None
    return CAPTION_LANGUAGE_BY_LANGUAGE.get(language)


def normalize_llm_temperature(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0 <= parsed < 2 else None


def normalize_llm_top_p(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0 <= parsed <= 1 else None


def normalize_llm_top_k(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if 1 <= parsed <= 64 else None


def default_settings() -> dict[str, int | str]:
    return {
        "n_shots": DEFAULT_NSHOT,
        "duration_sec": DEFAULT_DURATION_SEC,
        "width": DEFAULT_VIDEO_WIDTH,
        "height": DEFAULT_VIDEO_HEIGHT,
        "language": DEFAULT_LANGUAGE,
    }


def get_generation_settings(metadata: dict[str, Any] | None) -> dict[str, int | str]:
    """Return persisted generation settings, falling back to defaults."""
    base = default_settings()
    if not isinstance(metadata, dict):
        return base

    n_shots = normalize_n_shots(metadata.get(SESSION_NSHOT_KEY))
    duration_sec = normalize_duration_sec(metadata.get(SESSION_DURATION_KEY))
    try:
        width = int(metadata.get(SESSION_VIDEO_WIDTH_KEY))
        height = int(metadata.get(SESSION_VIDEO_HEIGHT_KEY))
    except (TypeError, ValueError):
        width = height = 0
    if width > 0 and height > 0:
        base["width"] = width
        base["height"] = height

    language = normalize_language(metadata.get(SESSION_LANGUAGE_KEY))
    if language is not None:
        base["language"] = language

    if n_shots is not None:
        base["n_shots"] = n_shots
        mapped = n_shots_to_duration(n_shots)
        if mapped is not None:
            base["duration_sec"] = mapped
    elif duration_sec is not None:
        base["duration_sec"] = duration_sec
        mapped = duration_to_n_shots(duration_sec)
        if mapped is not None:
            base["n_shots"] = mapped
    return base


def get_llm_sampling_settings(metadata: dict[str, Any] | None) -> dict[str, float | int]:
    """Return only explicitly set LLM sampling params (empty dict = gateway defaults)."""
    if not isinstance(metadata, dict):
        return {}
    out: dict[str, float | int] = {}
    temp = normalize_llm_temperature(metadata.get(SESSION_LLM_TEMPERATURE_KEY))
    if temp is not None:
        out["temperature"] = temp
    top_p = normalize_llm_top_p(metadata.get(SESSION_LLM_TOP_P_KEY))
    if top_p is not None:
        out["top_p"] = top_p
    top_k = normalize_llm_top_k(metadata.get(SESSION_LLM_TOP_K_KEY))
    if top_k is not None:
        out["top_k"] = top_k
    return out


def get_llm_sampling_for_api(metadata: dict[str, Any] | None) -> dict[str, float | int | None]:
    """API-facing view: keys always present, null when unset."""
    sampling = get_llm_sampling_settings(metadata)
    return {
        "temperature": sampling.get("temperature"),
        "top_p": sampling.get("top_p"),
        "top_k": sampling.get("top_k"),
    }


def apply_llm_sampling_settings(
    metadata: dict[str, Any],
    *,
    temperature: Any = _UNSET,
    top_p: Any = _UNSET,
    top_k: Any = _UNSET,
) -> dict[str, float | int | None]:
    """Persist optional LLM sampling params. Pass ``None`` to clear a field."""
    resolved = get_llm_sampling_for_api(metadata)

    if temperature is not _UNSET:
        if temperature is None or temperature == "":
            metadata.pop(SESSION_LLM_TEMPERATURE_KEY, None)
            resolved["temperature"] = None
        else:
            normalized = normalize_llm_temperature(temperature)
            if normalized is None:
                raise ValueError(f"invalid temperature: {temperature}")
            metadata[SESSION_LLM_TEMPERATURE_KEY] = normalized
            resolved["temperature"] = normalized

    if top_p is not _UNSET:
        if top_p is None or top_p == "":
            metadata.pop(SESSION_LLM_TOP_P_KEY, None)
            resolved["top_p"] = None
        else:
            normalized = normalize_llm_top_p(top_p)
            if normalized is None:
                raise ValueError(f"invalid top_p: {top_p}")
            metadata[SESSION_LLM_TOP_P_KEY] = normalized
            resolved["top_p"] = normalized

    if top_k is not _UNSET:
        if top_k is None or top_k == "":
            metadata.pop(SESSION_LLM_TOP_K_KEY, None)
            resolved["top_k"] = None
        else:
            normalized = normalize_llm_top_k(top_k)
            if normalized is None:
                raise ValueError(f"invalid top_k: {top_k}")
            metadata[SESSION_LLM_TOP_K_KEY] = normalized
            resolved["top_k"] = normalized

    return resolved


def apply_llm_sampling_from_wire(metadata: dict[str, Any], wire: dict[str, Any] | None) -> bool:
    """Apply LLM sampling overrides from a WS message envelope. Returns True if updated."""
    if not isinstance(wire, dict):
        return False
    updates: dict[str, Any] = {}
    for src, dst in (
        ("temperature", "temperature"),
        ("topP", "top_p"),
        ("top_p", "top_p"),
        ("topK", "top_k"),
        ("top_k", "top_k"),
    ):
        if src in wire:
            updates[dst] = wire[src]
    if not updates:
        return False
    apply_llm_sampling_settings(metadata, **updates)
    return True


def apply_generation_settings(
    metadata: dict[str, Any],
    *,
    n_shots: int | None = None,
    duration_sec: int | None = None,
    width: int | None = None,
    height: int | None = None,
    language: str | None = None,
) -> dict[str, int | str]:
    """Persist Director generation settings and return the resolved values."""
    resolved = get_generation_settings(metadata)

    if duration_sec is not None:
        normalized_duration = normalize_duration_sec(duration_sec)
        if normalized_duration is None:
            raise ValueError(f"invalid duration_sec: {duration_sec}")
        resolved["duration_sec"] = normalized_duration
        resolved["n_shots"] = duration_to_n_shots(normalized_duration) or DEFAULT_NSHOT
    elif n_shots is not None:
        normalized_n = normalize_n_shots(n_shots)
        if normalized_n is None:
            raise ValueError(f"invalid n_shots: {n_shots}")
        resolved["n_shots"] = normalized_n
        resolved["duration_sec"] = n_shots_to_duration(normalized_n) or DEFAULT_DURATION_SEC

    metadata[SESSION_NSHOT_KEY] = resolved["n_shots"]
    metadata[SESSION_DURATION_KEY] = resolved["duration_sec"]
    if width is not None:
        parsed_width = int(width)
        if parsed_width <= 0:
            raise ValueError(f"invalid width: {width}")
        resolved["width"] = parsed_width
    if height is not None:
        parsed_height = int(height)
        if parsed_height <= 0:
            raise ValueError(f"invalid height: {height}")
        resolved["height"] = parsed_height
    metadata[SESSION_VIDEO_WIDTH_KEY] = resolved["width"]
    metadata[SESSION_VIDEO_HEIGHT_KEY] = resolved["height"]
    if language is not None:
        normalized_language = normalize_language(language)
        if normalized_language is None:
            raise ValueError(f"invalid language: {language}")
        resolved["language"] = normalized_language
    metadata[SESSION_LANGUAGE_KEY] = resolved["language"]
    return resolved


def resolve_n_shots_from_wire(data: dict[str, Any] | None) -> int | None:
    """Resolve explicit shot count from WebSocket envelope or inbound message metadata."""
    if not isinstance(data, dict):
        return None
    for key in ("nShot", "nshot", "n_shots", "nShots"):
        raw = data.get(key)
        if raw is None or raw == "":
            continue
        parsed = normalize_n_shots(raw)
        if parsed is not None:
            return parsed
    duration = normalize_duration_sec(data.get("durationSec") or data.get("duration_sec"))
    if duration is not None:
        return duration_to_n_shots(duration)
    return None

