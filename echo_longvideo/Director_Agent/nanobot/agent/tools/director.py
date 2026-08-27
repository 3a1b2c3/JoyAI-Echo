"""Director tools for structured video-workspace orchestration.

These tools keep a work-specific state machine on disk so the model can rely
on tool-managed state instead of reconstructing progress from conversation
history alone.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import unquote, urlparse

from loguru import logger

from nanobot.agent.tools import (
    ArraySchema,
    BooleanSchema,
    IntegerSchema,
    ObjectSchema,
    StringSchema,
    Tool,
    tool_parameters,
    tool_parameters_schema,
)
from nanobot.integrations.echo_admission import (
    UNAVAILABLE_MESSAGE,
    EchoAdmissionController,
    EchoGeneratorBusyError,
    EchoGeneratorUnavailableError,
    is_connection_refused,
)
from nanobot.prompts import prompts
from nanobot.prompts.manager import PEManager
from nanobot.session.auto_generate import (
    effective_auto_generate_shot_count,
    get_auto_generate,
    locked_shot_count_from_goal,
)
from nanobot.session.reference_image import (
    clear_reference_image_needs_story_rewrite,
    is_reference_image_locked,
    lock_reference_image,
    normalize_reference_image,
    reference_image_needs_story_rewrite,
    reference_image_present,
)
from nanobot.utils.helpers import write_json_atomic

DIRECTOR_CONTEXT_TOOL_NAMES = frozenset(
    {
        "start_director",
        "set_director_goal",
        "get_workplace_status",
        "get_story",
        "write_story",
        "get_shot",
        "create_shot_prompt",
        "review_shot",
        "set_shot_references",
        "set_shot_memory_recommendations",
        "generate_echo_shot",
        "merge_shot",
    }
)

DIRECTOR_MUTATING_TOOL_NAMES = frozenset(
    {
        "start_director",
        "set_director_goal",
        "write_story",
        "create_shot_prompt",
        "review_shot",
        "set_shot_references",
        "set_shot_memory_recommendations",
        "generate_echo_shot",
        "merge_shot",
    }
)

# Shown in stepwise chat after the user locks shot_count and before they click
# Workplace 01 「下一步」. Never used for input-box one-click (auto_generate).
SHOT_COUNT_NEXT_STEP_HINT = (
    "点击「下一步」即可预览分镜脚本。满意脚本后接下来可以生成分镜镜头，"
    "确认无误并接受所有分镜后，就能合成最终成片了。有任何问题可以随时找我~"
)
SHOT_COUNT_NEXT_STEP_HINT_PENDING_KEY = "shot_count_next_step_hint_pending"
_STAGES_PAST_SHOT_COUNT_HINT = frozenset(
    {
        "shot_planning",
        "shot_generating",
        "shot_reviewing",
        "shot_revising",
        "merging",
        "done",
        "cancelled",
        "awaiting_memory_review",
        "failed",
    }
)


def consume_shot_count_next_step_hint(
    workspace: Path,
    session_key: str | None,
    *,
    auto_generate: bool = False,
    emit: bool = True,
) -> str | None:
    """Return the stepwise 「下一步」 hint once after shot_count is first locked."""
    if not session_key:
        return None
    tool = GetWorkplaceStatusTool(workspace=workspace)
    tool.set_context("websocket", "direct", effective_key=session_key)
    return tool._consume_shot_count_next_step_hint(
        auto_generate=auto_generate,
        emit=emit,
    )


def stepwise_shot_count_next_step_hint_eligible(
    workspace: Path,
    session_key: str | None,
    *,
    auto_generate: bool = False,
) -> bool:
    """True when stepwise work is still on 01 with a locked shot_count."""
    if not session_key or auto_generate:
        return False
    tool = GetWorkplaceStatusTool(workspace=workspace)
    tool.set_context("websocket", "direct", effective_key=session_key)
    work_id = tool._active_work_id()
    if not work_id:
        return False
    state = tool._load_state(work_id)
    if bool(state.get("auto_generate")):
        return False
    if str(state.get("stage") or "") in _STAGES_PAST_SHOT_COUNT_HINT:
        return False
    goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
    try:
        return int(goal.get("shot_count") or 0) > 0
    except (TypeError, ValueError):
        return False

_REMOTE_PROTOCOL_VERSION = "director-http-v1"
_R2V_SUBMIT_ATTEMPTS = 3
_R2V_TRANSIENT_HTTP_CODES = frozenset({429, 502, 503, 504})

_REMOTE_ENDPOINT_PATHS = {
    # Stable Echo Server routes used by the release workflow.
    "merge_shot": "/merge",
    # R2V unified generation (T2V / I2V / R2V)
    "r2v_generate": "/r2v",
}

_REMOTE_CALLBACK_PATHS = {
    "generate_echo_shot": "/api/director/echo-generate-shot/callback",
    "merge_shot": "/api/director/merge-shot/callback",
}

# Workplace workflow transitions are button-driven. Chat turns may only advance
# stages when the agent loop sets one of these injected workplace events.
WORKFLOW_GATE_BYPASS = "workplace_test_bypass"
_WORKFLOW_INJECTED_EVENT: ContextVar[str | None] = ContextVar(
    "director_workflow_injected_event",
    default=None,
)
_WORKFLOW_CONTEXT_UNSET = object()
_WORKFLOW_GATE_OPERATIONS: dict[str, frozenset[str]] = {
    "write_story_confirmed": frozenset(
        {
            "workplace_workflow_confirm_story",
            "workplace_workflow_start_generation",
            "workplace_beats_edit",
            WORKFLOW_GATE_BYPASS,
        }
    ),
    "create_shot_prompt": frozenset(
        {
            "workplace_workflow_start_generation",
            "workplace_beats_edit",
            "workplace_shot_revision",
            WORKFLOW_GATE_BYPASS,
        }
    ),
    "set_shot_references": frozenset(
        {
            "workplace_workflow_start_generation",
            WORKFLOW_GATE_BYPASS,
        }
    ),
    "set_shot_memory_recommendations": frozenset(
        {
            "workplace_memory_recommendation",
            WORKFLOW_GATE_BYPASS,
        }
    ),
    "generate_echo_shot": frozenset(
        {
            "workplace_workflow_start_generation",
            "workplace_shot_revision",
            WORKFLOW_GATE_BYPASS,
        }
    ),
    "merge_shot": frozenset(
        {
            "workplace_workflow_start_merge",
            WORKFLOW_GATE_BYPASS,
        }
    ),
    "review_shot": frozenset({WORKFLOW_GATE_BYPASS}),
}


def _workflow_gate_error(operation: str) -> str:
    return prompts.text("director.workflow_gate_error", operation=operation)


def _allow_workflow_operation(operation: str) -> bool:
    event = _WORKFLOW_INJECTED_EVENT.get()
    if event == WORKFLOW_GATE_BYPASS:
        return True
    if not event:
        return False
    return event in _WORKFLOW_GATE_OPERATIONS.get(operation, frozenset())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _slugify(value: str, *, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or fallback


def _json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _story_profile_validation_error(story_profile: Any) -> str | None:
    if not isinstance(story_profile, dict):
        return "Error: story_profile must be a JSON object with summary and beats."
    summary = story_profile.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return "Error: story_profile.summary must be a non-empty string."
    beats = story_profile.get("beats")
    if not isinstance(beats, list) or len(beats) < 1:
        return "Error: story_profile.beats must contain at least one beat."
    for index, beat in enumerate(beats):
        if not isinstance(beat, dict):
            return (
                f"Error: story_profile.beats[{index}] must be an object with shot_id and summary."
            )
        beat_summary = beat.get("summary")
        if not isinstance(beat_summary, str) or not beat_summary.strip():
            return f"Error: story_profile.beats[{index}].summary must be a non-empty string."
    return None


def _story_profile_language_validation_error(
    story_profile: dict[str, Any],
) -> str | None:
    """Keep all natural-language story-profile prose in the selected language."""
    from nanobot.session.generation_settings import normalize_language

    normalized = normalize_language(story_profile.get("language"))
    if normalized is None:
        normalized = normalize_language(story_profile.get("caption_language"))
    if normalized is None:
        normalized = normalize_language(story_profile.get("dialogue_language"))
    if normalized is None:
        return None

    summaries: list[tuple[str, str]] = []
    prose: list[tuple[str, str]] = []

    def collect_text(value: Any, path: str) -> None:
        if isinstance(value, str):
            if value.strip():
                prose.append((path, value))
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                collect_text(item, f"{path}[{index}]")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                collect_text(item, f"{path}.{key}")

    summary = story_profile.get("summary")
    if isinstance(summary, str):
        summaries.append(("story_profile.summary", summary))
    beats = story_profile.get("beats")
    if isinstance(beats, list):
        for index, beat in enumerate(beats):
            if not isinstance(beat, dict):
                continue
            if isinstance(beat.get("summary"), str):
                summaries.append((f"story_profile.beats[{index}].summary", beat["summary"]))
            collect_text(
                beat.get("dialogue_intent"),
                f"story_profile.beats[{index}].dialogue_intent",
            )

    for field in ("anchors", "scene_anchors", "shot_to_content"):
        collect_text(story_profile.get(field), f"story_profile.{field}")

    def _has_chinese(value: str) -> bool:
        return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", value))

    if normalized == "zh":
        invalid_summaries = [name for name, value in summaries if not _has_chinese(value)]
        invalid_prose = [name for name, value in prose if not _has_chinese(value)]
        if invalid_prose:
            invalid = invalid_summaries + invalid_prose
            return (
                "Error: Chinese story-profile prose is required for this work. "
                "Rewrite all natural-language story_profile fields in Simplified Chinese. "
                f"Invalid fields: {', '.join(invalid)}."
            )
        if invalid_summaries:
            return (
                "Error: Chinese storyboard summaries are required for this work. "
                "Rewrite story_profile.summary and every beats[].summary in Simplified Chinese. "
                f"Invalid fields: {', '.join(invalid_summaries)}."
            )
    elif normalized == "en":
        invalid_summaries = [name for name, value in summaries if _has_chinese(value)]
        invalid_prose = [name for name, value in prose if _has_chinese(value)]
        if invalid_prose:
            invalid = invalid_summaries + invalid_prose
            return (
                "Error: English story-profile prose is required for this work. "
                "Rewrite all natural-language story_profile fields in English. "
                f"Invalid fields: {', '.join(invalid)}."
            )
        if invalid_summaries:
            return (
                "Error: English storyboard summaries are required for this work. "
                "Rewrite story_profile.summary and every beats[].summary in English. "
                f"Invalid fields: {', '.join(invalid_summaries)}."
            )
    return None


def _story_md_language_validation_error(
    story_md: str,
    story_profile: dict[str, Any],
) -> str | None:
    """Keep the displayed screenplay aligned with the selected story language."""
    from nanobot.session.generation_settings import normalize_language

    normalized = normalize_language(story_profile.get("language"))
    if normalized is None:
        normalized = normalize_language(story_profile.get("caption_language"))
    if normalized is None:
        normalized = normalize_language(story_profile.get("dialogue_language"))
    if normalized is None or not story_md.strip():
        return None

    has_chinese = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", story_md))
    if normalized == "zh" and not has_chinese:
        return (
            "Error: Chinese screenplay prose is required for this work. "
            "Rewrite story_md in Simplified Chinese."
        )
    if normalized == "en" and has_chinese:
        return (
            "Error: English screenplay prose is required for this work. "
            "Rewrite story_md in English."
        )
    return None


def _normalize_story_profile(profile: dict[str, Any]) -> None:
    beats = profile.get("beats")
    if not isinstance(beats, list):
        return
    normalized: list[dict[str, Any]] = []
    for index, beat in enumerate(beats):
        if not isinstance(beat, dict):
            continue
        summary = str(beat.get("summary") or "").strip()
        if not summary:
            continue
        normalized.append({"shot_id": index + 1, "summary": summary})
    profile["beats"] = normalized
    shot_to_content: dict[str, str] = {}
    content_to_shots: dict[str, list[str]] = {}
    for beat in normalized:
        shot_id = int(beat["shot_id"])
        shot_key = f"shot_{shot_id:03d}"
        shot_to_content[shot_key] = str(beat["summary"])
        content_to_shots[f"beat_{shot_id:03d}"] = [shot_key]
    profile["shot_to_content"] = shot_to_content
    profile["content_to_shots"] = content_to_shots


def _apply_story_profile_language(profile: dict[str, Any], language: str | None) -> None:
    """Stamp UI language onto story_profile and keep caption/dialogue locks in sync."""
    from nanobot.session.generation_settings import (
        language_to_caption_language,
        language_to_dialogue_language,
        normalize_language,
    )

    normalized = normalize_language(language)
    if normalized is None:
        return
    profile["language"] = normalized
    dialogue = language_to_dialogue_language(normalized)
    if dialogue:
        profile["dialogue_language"] = dialogue
    caption = language_to_caption_language(normalized)
    if caption and not str(profile.get("caption_language") or "").strip():
        profile["caption_language"] = caption


def _ensure_story_profile_caption_language(profile: dict[str, Any]) -> None:
    """Derive the full-caption language for legacy profiles that only locked dialogue."""
    from nanobot.session.generation_settings import (
        language_to_caption_language,
        normalize_language,
    )

    if str(profile.get("caption_language") or "").strip():
        return
    normalized = normalize_language(profile.get("language"))
    if normalized is None:
        normalized = normalize_language(profile.get("dialogue_language"))
    caption = language_to_caption_language(normalized)
    if caption:
        profile["caption_language"] = caption


def _preserve_story_profile_language(
    profile: dict[str, Any],
    previous: dict[str, Any] | None = None,
) -> None:
    """Keep language / dialogue_language when an overwrite omits them."""
    from nanobot.session.generation_settings import normalize_language

    if normalize_language(profile.get("language")) is not None:
        _apply_story_profile_language(profile, profile.get("language"))
        return
    if isinstance(previous, dict):
        prev_language = normalize_language(previous.get("language"))
        if prev_language is not None:
            _apply_story_profile_language(profile, prev_language)
            return
        prev_dialogue = previous.get("dialogue_language")
        if isinstance(prev_dialogue, str) and prev_dialogue.strip() and "dialogue_language" not in profile:
            profile["dialogue_language"] = prev_dialogue.strip()
    _ensure_story_profile_caption_language(profile)


def _caption_language_validation_error(
    caption: str,
    story_profile: dict[str, Any],
) -> str | None:
    """Reject natural-language prose that violates the work's caption-language lock.

    The target language must account for at least 90% of the meaningful character
    count (excluding technical tokens).  A handful of proper-noun transliterations
    in the source language are tolerated below the 10% threshold.
    """
    from nanobot.session.generation_settings import normalize_language

    caption_language = story_profile.get("caption_language")
    normalized = normalize_language(caption_language)
    if normalized is None:
        normalized = normalize_language(story_profile.get("language"))
    if normalized is None:
        normalized = normalize_language(story_profile.get("dialogue_language"))
    if normalized is None:
        return None

    # Strip technical tokens that are allowed in either language.
    prose = re.sub(
        r"(?<![A-Za-z0-9_])ID_[A-Z0-9]+(?![A-Za-z0-9_])",
        "",
        caption,
        flags=re.IGNORECASE,
    )
    prose = re.sub(
        r"(?<![A-Za-z0-9_])shot\d+(?![A-Za-z0-9_])",
        "",
        prose,
        flags=re.IGNORECASE,
    )
    prose = re.sub(
        r"(?<![A-Za-z])OCR(?![A-Za-z])",
        "",
        prose,
        flags=re.IGNORECASE,
    )

    chinese_chars = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", prose)
    english_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", prose)

    chinese_char_count = len(chinese_chars)
    english_char_count = sum(len(w) for w in english_words)

    total_meaningful = chinese_char_count + english_char_count
    if total_meaningful == 0:
        return None

    if normalized == "en":
        en_ratio = english_char_count / total_meaningful
        if en_ratio >= 0.9:
            return None
        preview = "".join(chinese_chars[:16])
        return (
            "Error: English caption required for this work (currently "
            f"{en_ratio:.0%} English). Rewrite the entire caption in English "
            "before calling create_shot_prompt again; translate all Chinese "
            f"descriptions, actions, dialogue, and declarations. "
            f"Chinese found: {preview}."
        )

    if normalized == "zh":
        zh_ratio = chinese_char_count / total_meaningful
        if zh_ratio >= 0.9:
            return None
        preview = ", ".join(english_words[:8])
        return (
            "Error: Chinese caption required for this work (currently "
            f"{zh_ratio:.0%} Chinese). Rewrite the entire caption in Chinese "
            "before calling create_shot_prompt again. Keep only required "
            "technical tokens such as ID_A, shot1:, and OCR; use ID_A说 for "
            "speech; translate all descriptions, actions, camera, sound, music, "
            f"and declarations. English found: {preview}."
        )

    return None


def _shot_key(shot_id: int) -> str:
    return f"shot_{shot_id:03d}"


def _job_id(kind: str, work_id: str, suffix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{kind}-{work_id}-{suffix}-{stamp}"


def _shot_id_from_key(shot_key: str) -> int:
    if not shot_key.startswith("shot_"):
        raise ValueError(f"Invalid shot key: {shot_key}")
    return int(shot_key.split("_", 1)[1])


# Echo generate-shot timing: see echo_generate_shot.md (25fps, num_frames = 1 + 8k, clamp [25, 241]).
ECHO_SHOT_FPS = 25
ECHO_MIN_NUM_FRAMES = 25
ECHO_MAX_NUM_FRAMES = 241
ECHO_DEFAULT_NUM_FRAMES = 241
ECHO_DEFAULT_DURATION_SEC = 4.0


def snap_echo_num_frames(raw_frames: int) -> int:
    """Snap upward to the nearest valid 1+8k frame count and clamp to Echo bounds.

    Echo requires ``num_frames = 1 + 8k``. Snapping up keeps generated duration
    from falling below the caller's requested length (except at the hard max).
    """
    frames = int(raw_frames)
    frames = max(ECHO_MIN_NUM_FRAMES, min(ECHO_MAX_NUM_FRAMES, frames))
    remainder = (frames - 1) % 8
    if remainder:
        frames += 8 - remainder
    if frames > ECHO_MAX_NUM_FRAMES:
        # Largest valid 1+8k at or below the hard max (241 == 1+8*30).
        frames = ECHO_MAX_NUM_FRAMES
        remainder = (frames - 1) % 8
        if remainder:
            frames -= remainder
    return max(ECHO_MIN_NUM_FRAMES, min(ECHO_MAX_NUM_FRAMES, frames))


def duration_sec_to_num_frames(duration_sec: float) -> int:
    """Convert desired seconds to the frame count Echo will actually generate.

    Ceil to frames then snap upward so playback length is >= the request
    (unless capped by ``ECHO_MAX_NUM_FRAMES``).
    """
    import math

    return snap_echo_num_frames(math.ceil(float(duration_sec) * ECHO_SHOT_FPS))


def num_frames_to_duration_sec(num_frames: int) -> float:
    """Return duration in whole seconds (nearest second of frames/fps) for UI/state."""
    return float(round(int(num_frames) / ECHO_SHOT_FPS))


def num_frames_to_exact_duration_sec(num_frames: int) -> float:
    """Exact playback seconds for the given frame count (no rounding).

    Sent to the Echo backend so it does not re-derive frames from a rounded
    ``duration_sec`` (e.g. 7.0) and snap back downward.
    """
    return float(num_frames) / float(ECHO_SHOT_FPS)


def sync_shot_echo_duration(shot: dict[str, Any], duration_sec: float) -> int:
    """Persist snapped Echo timing on the shot record."""
    num_frames = duration_sec_to_num_frames(duration_sec)
    actual_duration_sec = num_frames_to_duration_sec(num_frames)
    shot["duration_sec"] = actual_duration_sec
    shot["num_frames"] = num_frames
    return num_frames


def resolve_echo_duration_seconds(
    shot: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> float:
    """Resolve per-shot seconds for Echo generation (aligned with workplace UI defaults)."""
    for key in ("duration_sec", "duration_seconds"):
        try:
            value = float(shot.get(key))
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    try:
        shot_frames = shot.get("num_frames")
        if shot_frames is not None:
            value = float(num_frames_to_duration_sec(int(shot_frames)))
            if value > 0:
                return value
    except (TypeError, ValueError):
        pass
    goal = (
        state.get("goal") if isinstance(state, dict) and isinstance(state.get("goal"), dict) else {}
    )
    try:
        goal_duration = float(goal.get("shot_duration_sec") or 0)
        if goal_duration > 0:
            return goal_duration
    except (TypeError, ValueError):
        pass
    return ECHO_DEFAULT_DURATION_SEC


def rewrite_prompt_for_i2v(original_prompt: str, caption_language: str) -> str:
    """Rewrite a shot prompt for I2V by prepending the language-matched first-frame sentence.

    Follows ``pe/v7_cinematic_full/skills/i2v-tail-frame-prompt-rewriter/SKILL.md``.
    Only the opening sentence is added; the rest of the prompt is preserved unchanged.
    """
    caption_language = (caption_language or "").strip().lower()
    is_chinese = caption_language in {"simplified chinese", "zh", "chinese", "mandarin chinese"}

    first_frame_zh = (
        "以当前图片作为视频首帧，并基于首帧中已有的人物、物体、环境、构图、机位、光线和动作状态自然延续。"
    )
    first_frame_en = (
        "Use the current image as the first frame of the video, and continue naturally "
        "from the characters, objects, environment, composition, camera position, lighting, "
        "and action state already shown in it."
    )

    first_frame_sentence = first_frame_zh if is_chinese else first_frame_en

    trimmed = original_prompt.strip()

    # Detect if the prompt has a cut-count style opening (e.g. "1 cut" / "1个镜头").
    # Insert the first-frame sentence before the cut-count sentence.
    cut_count_pattern = re.compile(
        r"^(\d+)\s*(?:cuts?|个镜头|个景别)",
        re.IGNORECASE,
    )
    match = cut_count_pattern.match(trimmed)
    if match:
        prefix = trimmed[: match.end()]
        rest = trimmed[match.end() :]
        return f"{first_frame_sentence}\n{prefix}{rest}"

    return f"{first_frame_sentence}\n{trimmed}"


class DirectorTool(Tool):
    """Shared helpers for director-state tools."""

    _DEFAULT_STAGE = "story_discussion"
    _FINAL_STAGES = frozenset({"done", "cancelled"})

    @property
    def description(self) -> str:
        """Pull each tool's description from the active PE set, keyed by tool name."""
        return prompts.text(f"director.tool.{self.name}.description")

    def __init__(
        self,
        workspace: Path,
        *,
        tools_config: Any | None = None,
        callback_base_url: str | None = None,
    ):
        from nanobot.config.schema import ToolsConfig

        self.workspace = workspace
        self._tools_config = tools_config or ToolsConfig()
        self._callback_base_url = (
            callback_base_url.rstrip("/")
            if isinstance(callback_base_url, str) and callback_base_url.strip()
            else None
        )
        self._channel: ContextVar[str] = ContextVar("director_channel", default="cli")
        self._chat_id: ContextVar[str] = ContextVar("director_chat_id", default="direct")
        self._session_key: ContextVar[str] = ContextVar(
            "director_session_key",
            default="cli:direct",
        )

    def set_context(
        self,
        channel: str,
        chat_id: str,
        effective_key: str | None = None,
        *,
        injected_event: str | None | object = _WORKFLOW_CONTEXT_UNSET,
    ) -> None:
        self._channel.set(channel)
        self._chat_id.set(chat_id)
        self._session_key.set(effective_key or f"{channel}:{chat_id}")
        if injected_event is not _WORKFLOW_CONTEXT_UNSET:
            _WORKFLOW_INJECTED_EVENT.set(injected_event)

    @staticmethod
    def allow_workflow_gate_bypass() -> None:
        """Test helper: allow gated workflow tools without a workplace injection."""
        _WORKFLOW_INJECTED_EVENT.set(WORKFLOW_GATE_BYPASS)

    @property
    def director_root(self) -> Path:
        return self.workspace / "director"

    @property
    def works_root(self) -> Path:
        return self.director_root / "works"

    @property
    def active_work_path(self) -> Path:
        return self.director_root / "active_work.json"

    @property
    def session_map_path(self) -> Path:
        return self.director_root / "session_map.json"

    def _ensure_root(self) -> None:
        self.works_root.mkdir(parents=True, exist_ok=True)

    def _read_reference_image_from_session(self) -> dict[str, Any] | None:
        """从 session metadata 读取首帧参考图信息。"""
        try:
            from nanobot.session.manager import SessionManager

            session_key = self._session_key.get()
            if not session_key:
                return None
            session = SessionManager(self.workspace).get_or_create(session_key)
            metadata = session.metadata if isinstance(session.metadata, dict) else {}
            return normalize_reference_image(metadata.get("reference_image"))
        except Exception:
            logger.exception("director: failed to read session reference_image")
            return None

    def _session_auto_generate(self) -> bool:
        try:
            from nanobot.session.manager import SessionManager

            session_key = self._session_key.get()
            if not session_key:
                return False
            session = SessionManager(self.workspace).get_or_create(session_key)
            metadata = session.metadata if isinstance(session.metadata, dict) else {}
            return get_auto_generate(metadata)
        except Exception:
            logger.exception("director: failed to read session auto_generate")
            return False

    def _consume_shot_count_next_step_hint(
        self,
        *,
        auto_generate: bool = False,
        emit: bool = True,
    ) -> str | None:
        work_id = self._active_work_id()
        if not work_id:
            return None
        state = self._load_state(work_id)
        pending = bool(state.pop(SHOT_COUNT_NEXT_STEP_HINT_PENDING_KEY, False))
        if not pending:
            return None
        stage = str(state.get("stage") or "")
        should_emit = (
            emit
            and not auto_generate
            and not bool(state.get("auto_generate"))
            and stage not in _STAGES_PAST_SHOT_COUNT_HINT
        )
        self._save_state(work_id, state)
        return SHOT_COUNT_NEXT_STEP_HINT if should_emit else None

    def _lock_session_reference_image(self) -> None:
        try:
            from nanobot.session.manager import SessionManager

            session_key = self._session_key.get()
            if not session_key:
                return
            manager = SessionManager(self.workspace)
            session = manager.get_or_create(session_key)
            if not isinstance(session.metadata, dict):
                session.metadata = {}
            if session.metadata.get("reference_image_locked") is True:
                return
            session.metadata["reference_image_locked"] = True
            manager.save(session)
        except Exception:
            logger.exception("director: failed to lock session reference_image")

    def _clear_reference_image_story_rewrite_flag(self) -> None:
        try:
            from nanobot.session.manager import SessionManager

            session_key = self._session_key.get()
            if not session_key:
                return
            manager = SessionManager(self.workspace)
            session = manager.get_or_create(session_key)
            if not isinstance(session.metadata, dict):
                return
            if not session.metadata.get("reference_image_needs_story_rewrite"):
                return
            clear_reference_image_needs_story_rewrite(session.metadata)
            manager.save(session)
        except Exception:
            logger.exception(
                "director: failed to clear reference_image_needs_story_rewrite"
            )

    def _session_reference_inject_failed(self) -> bool:
        try:
            from nanobot.session.manager import SessionManager

            session_key = self._session_key.get()
            if not session_key:
                return False
            session = SessionManager(self.workspace).get_or_create(session_key)
            metadata = session.metadata if isinstance(session.metadata, dict) else {}
            return bool(metadata.get("reference_image_inject_failed"))
        except Exception:
            logger.exception("director: failed to read reference_image_inject_failed")
            return False

    def _session_reference_needs_rewrite(self) -> bool:
        try:
            from nanobot.session.manager import SessionManager

            session_key = self._session_key.get()
            if not session_key:
                return False
            session = SessionManager(self.workspace).get_or_create(session_key)
            metadata = session.metadata if isinstance(session.metadata, dict) else {}
            return reference_image_needs_story_rewrite(metadata)
        except Exception:
            logger.exception("director: failed to read reference_image_needs_story_rewrite")
            return False

    def _effective_auto_generate_shot_count(self, goal: dict[str, Any] | None) -> int | None:
        try:
            from nanobot.session.manager import SessionManager

            session_key = self._session_key.get()
            metadata: dict[str, Any] | None = None
            if session_key:
                session = SessionManager(self.workspace).get_or_create(session_key)
                metadata = session.metadata if isinstance(session.metadata, dict) else {}
            return effective_auto_generate_shot_count(goal=goal, metadata=metadata)
        except Exception:
            logger.exception("director: failed to resolve auto_generate shot_count")
            return locked_shot_count_from_goal(goal)

    def _state_first_frame_url(self, state: dict[str, Any]) -> str | None:
        ref = normalize_reference_image(state.get("reference_image"))
        if not ref:
            return None
        url = ref.get("url")
        return url if isinstance(url, str) and url.strip() else None

    # ── tail-frame extraction pipeline (shared by agent + REST paths) ──

    @staticmethod
    def _extract_tail_frame(video_path: Path, output_path: Path) -> bool:
        """Extract the last frame of *video_path* as a PNG using ffmpeg."""
        from nanobot.director.memory_coordinator import _resolve_media_binary

        ffmpeg = _resolve_media_binary("ffmpeg")
        cmd = [
            ffmpeg, "-sseof", "-1", "-i", str(video_path),
            "-update", "1", "-q:v", "1", str(output_path), "-y",
        ]
        import subprocess

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            return output_path.is_file() and output_path.stat().st_size > 0
        except Exception:
            return False

    def _publish_tail_frame(
        self, image_path: Path, work_id: str, shot_id: int
    ) -> str | None:
        """Persist tail frame locally. Returns the public URL or None on failure."""
        from nanobot.storage.files import configured_file_publisher

        name = f"tail_frames/shot_{shot_id:03d}.png"
        try:
            publisher = configured_file_publisher(
                work_id,
                storage=self._tools_config.file_storage,
                workspace=self.workspace,
            )
            return publisher(str(image_path), name)
        except Exception:
            return None

    def _extract_and_publish_tail_frame(
        self, work_id: str, shot_id: int, video_url: str,
    ) -> str | None:
        """Download video, extract last frame, publish locally. Returns public URL."""
        import shutil
        import tempfile
        import urllib.request

        tmp_dir = Path(tempfile.mkdtemp(prefix="tail_frame_"))
        try:
            video_path = tmp_dir / "source.mp4"
            frame_path = tmp_dir / "tail.png"

            # download
            if video_url.startswith(("http://", "https://")):
                req = urllib.request.Request(video_url, headers={"Accept": "video/mp4,*/*"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    with open(video_path, "wb") as f:
                        shutil.copyfileobj(resp, f)
            else:
                src = Path(video_url)
                if not src.is_file():
                    return None
                shutil.copyfile(str(src), str(video_path))

            if not video_path.is_file() or video_path.stat().st_size <= 0:
                return None

            # extract
            if not DirectorTool._extract_tail_frame(video_path, frame_path):
                return None

            # upload
            return self._publish_tail_frame(frame_path, work_id, shot_id)
        except Exception:
            return None
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        write_json_atomic(path, data)

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")

    def _remote_http_base_url(self) -> str | None:
        echo_generator = getattr(self._tools_config, "echo_generator", None)
        base = ""
        if echo_generator is not None:
            base = str(getattr(echo_generator, "base_url", "") or "").strip()
        return base.rstrip("/") if base else None

    def _remote_callback_base_url(self) -> str | None:
        if self._callback_base_url:
            return self._callback_base_url
        echo_generator = getattr(self._tools_config, "echo_generator", None)
        base = ""
        if echo_generator is not None:
            base = str(getattr(echo_generator, "callback_base_url", "") or "").strip()
        return base.rstrip("/") if base else None

    def _remote_http_timeout_sec(self) -> float:
        echo_generator = getattr(self._tools_config, "echo_generator", None)
        raw_timeout = (
            getattr(echo_generator, "http_timeout_sec", 30.0) if echo_generator is not None else 30.0
        )
        try:
            return max(1.0, float(raw_timeout))
        except (TypeError, ValueError):
            return 30.0

    def _remote_endpoint_path(self, operation: str) -> str:
        endpoint = _REMOTE_ENDPOINT_PATHS.get(operation)
        if not endpoint:
            raise RuntimeError(f"No remote endpoint mapping exists for operation '{operation}'.")
        return endpoint

    def _remote_callback_path(self, operation: str) -> str | None:
        return _REMOTE_CALLBACK_PATHS.get(operation)

    def _remote_callback_url(self, operation: str) -> str | None:
        base_url = self._remote_callback_base_url()
        callback_path = self._remote_callback_path(operation)
        if not base_url or not callback_path:
            return None
        return f"{base_url}{callback_path}"

    def _build_remote_callback_contract(
        self,
        work_id: str,
        job_id: str,
        operation: str,
        target: str | list[str],
    ) -> dict[str, Any]:
        contract = {
            "event_type": "director_remote_result",
            "protocol_version": _REMOTE_PROTOCOL_VERSION,
            "operation": operation,
            "work_id": work_id,
            "job_id": job_id,
            "target": target,
            "channel": self._channel.get(),
            "chat_id": self._chat_id.get(),
            "session_key": self._session_key.get(),
            "inject_back_to_agent": True,
            "note": (
                "When the backend finishes, your client-side callback handler should "
                "clear the pending_remote_jobs entry, update the director workspace, "
                "and publish an InboundMessage for this session."
            ),
        }
        callback_url = self._remote_callback_url(operation)
        if callback_url:
            contract["url"] = callback_url
        return contract

    def _build_remote_request_envelope(
        self,
        operation: str,
        work_id: str,
        job_id: str,
        target: str | list[str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "protocol_version": _REMOTE_PROTOCOL_VERSION,
            "operation": operation,
            "job": {
                "job_id": job_id,
                "work_id": work_id,
                "target": target,
                "created_at": _now_iso(),
            },
            "callback": self._build_remote_callback_contract(work_id, job_id, operation, target),
            "payload": payload,
        }

    def _post_remote_http_request(
        self,
        endpoint_url: str,
        envelope: dict[str, Any],
    ) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        callback = envelope.get("callback")
        if isinstance(callback, dict):
            callback_url = callback.get("url")
            if isinstance(callback_url, str) and callback_url.strip():
                headers["X-Nanobot-Director-Callback-Url"] = callback_url.strip()
        body = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
        request = urllib_request.Request(
            endpoint_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(
                request, timeout=self._remote_http_timeout_sec()
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Remote HTTP request failed: {exc}") from exc
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_response": raw}
        return parsed if isinstance(parsed, dict) else {"response": parsed}

    def _active_work_id(self) -> str | None:
        session_map = self._read_json(self.session_map_path, {})
        if not isinstance(session_map, dict):
            return None
        entry = session_map.get(self._session_key.get())
        if isinstance(entry, dict):
            return entry.get("active")
        # Backwards compat: old format stored bare work_id string
        if isinstance(entry, str):
            return entry
        return None

    def _session_work_history(self) -> list[str]:
        session_map = self._read_json(self.session_map_path, {})
        if not isinstance(session_map, dict):
            return []
        entry = session_map.get(self._session_key.get())
        if isinstance(entry, dict):
            history = entry.get("history", [])
            return history if isinstance(history, list) else []
        # Backwards compat: old format stored bare work_id string
        if isinstance(entry, str):
            return [entry]
        return []

    def _set_active_work(self, work_id: str) -> None:
        self._ensure_root()
        session_key = self._session_key.get()
        session_map = self._read_json(self.session_map_path, {})
        if not isinstance(session_map, dict):
            session_map = {}
        entry = session_map.get(session_key)
        # Migrate old bare-string entries
        if isinstance(entry, str):
            entry = {"active": entry, "history": [entry]}
        elif not isinstance(entry, dict):
            entry = {"active": None, "history": []}
        history = entry.get("history", [])
        if not isinstance(history, list):
            history = []
        if work_id not in history:
            history.append(work_id)
        entry["active"] = work_id
        entry["history"] = history
        session_map[session_key] = entry
        self._write_json(self.session_map_path, session_map)
        self._write_json(
            self.active_work_path,
            {
                "work_id": work_id,
                "channel": self._channel.get(),
                "chat_id": self._chat_id.get(),
                "session_key": session_key,
                "updated_at": _now_iso(),
            },
        )

    def _resolve_work_id(self, work_id: str | None = None) -> tuple[str | None, Path | None]:
        self._ensure_root()
        candidate = work_id or self._active_work_id()
        if not candidate:
            return None, None
        work_dir = self.works_root / candidate
        if not work_dir.exists():
            return None, None
        return candidate, work_dir

    def _paths(self, work_id: str) -> dict[str, Path]:
        work_dir = self.works_root / work_id
        return {
            "work_dir": work_dir,
            "state": work_dir / "state.json",
            "fact": work_dir / "fact.md",
            "work_memory": work_dir / "work_memory_lite.md",
            "story": work_dir / "story.md",
            "story_profile": work_dir / "story_profile.json",
            "shots": work_dir / "shots",
            "jobs": work_dir / "jobs",
            "outputs": work_dir / "outputs",
            "memory_bank": work_dir / "memory" / "memory_bank.json",
            "previous_shot_memory": work_dir / "memory" / "previous_shot.json",
            "manual_memory_workspace": work_dir / "memory" / "manual" / "workspace.json",
            "memory_asset_profiles": work_dir / "memory" / "asset_profiles.json",
        }

    @staticmethod
    def _automatic_memory_asset_id(raw: dict[str, Any], memory_id: str, kind: str) -> str:
        fingerprint = json.dumps(
            [
                memory_id,
                int(raw.get("source_shot_id") or 0),
                int(raw.get("frame_index") or 0),
                kind,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return "auto_" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:20]

    def _memory_asset_catalog(self, work_id: str) -> list[dict[str, Any]]:
        """Return profile-bearing assets safe for the agent to reason over."""
        paths = self._paths(work_id)
        overrides = self._read_json(paths["memory_asset_profiles"], {})
        overrides = overrides if isinstance(overrides, dict) else {}
        assets: list[dict[str, Any]] = []

        def add_automatic(raw: Any, memory_id: str, kind: str) -> None:
            if not isinstance(raw, dict):
                return
            asset_id = self._automatic_memory_asset_id(raw, memory_id, kind)
            override = overrides.get(asset_id)
            override = override if isinstance(override, dict) else {}
            profile_text = str(
                override.get("profile_text")
                or raw.get("profile_text")
                or raw.get("reasoning")
                or ""
            ).strip()
            if not profile_text:
                return
            identities = override.get("identity_ids") or raw.get("visible_character_ids")
            if not isinstance(identities, list):
                identities = [memory_id] if memory_id.startswith("ID_") else []
            reference_type = str(
                override.get("reference_type")
                if "reference_type" in override
                else raw.get("reference_type") or ""
            ).strip()
            reference_label = str(
                override.get("reference_label")
                if "reference_label" in override
                else raw.get("reference_label") or ""
            ).strip()
            assets.append({
                "asset_id": asset_id,
                "media_type": (
                    "image_audio" if raw.get("image_path") and raw.get("audio_path")
                    else "audio" if raw.get("audio_path")
                    else "image"
                ),
                "profile_text": profile_text,
                "identity_ids": [str(value) for value in identities if str(value).strip()],
                **({"reference_type": reference_type} if reference_type else {}),
                **({"reference_label": reference_label} if reference_label else {}),
                "source": {
                    "type": "generated_shot",
                    "shot_id": int(raw.get("source_shot_id") or 0),
                    "timestamp_sec": float(raw.get("timestamp_sec") or 0),
                },
            })

        bank = self._read_json(paths["memory_bank"], {})
        if isinstance(bank, dict):
            for memory_id, raw in bank.items():
                add_automatic(raw, str(memory_id), "character")
        previous = self._read_json(paths["previous_shot_memory"], None)
        add_automatic(previous, "PREVIOUS_SHOT", "previous_shot")

        manual = self._read_json(paths["manual_memory_workspace"], {})
        rows = manual.get("assets") if isinstance(manual, dict) else []
        for raw in rows if isinstance(rows, list) else []:
            if not isinstance(raw, dict):
                continue
            profile_text = str(raw.get("profile_text") or "").strip()
            asset_id = str(raw.get("asset_id") or "").strip()
            if not asset_id or not profile_text:
                continue
            has_image = bool(raw.get("image_path"))
            has_audio = bool(raw.get("audio_path"))
            reference_type = str(raw.get("reference_type") or "").strip()
            reference_label = str(raw.get("reference_label") or "").strip()
            source_shot_id = int(raw.get("source_shot_id") or 0)
            source = (
                {
                    "type": "generated_shot",
                    "shot_id": source_shot_id,
                    "timestamp_sec": float(raw.get("timestamp_sec") or 0),
                    **(
                        {"audio_start_sec": float(raw["audio_start_sec"])}
                        if raw.get("audio_start_sec") is not None
                        else {}
                    ),
                    **(
                        {"audio_end_sec": float(raw["audio_end_sec"])}
                        if raw.get("audio_end_sec") is not None
                        else {}
                    ),
                }
                if source_shot_id > 0
                else {"type": "local_upload"}
            )
            assets.append({
                "asset_id": asset_id,
                "media_type": (
                    "image_audio" if has_image and has_audio
                    else "audio" if has_audio
                    else "image"
                ),
                "profile_text": profile_text,
                "identity_ids": [
                    str(value) for value in raw.get("identity_ids", []) if str(value).strip()
                ],
                **({"reference_type": reference_type} if reference_type else {}),
                **({"reference_label": reference_label} if reference_label else {}),
                "source": source,
            })
        return assets

    def _load_state(self, work_id: str) -> dict[str, Any]:
        state = self._read_json(self._paths(work_id)["state"], {})
        return state if isinstance(state, dict) else {}

    def _save_state(self, work_id: str, state: dict[str, Any]) -> None:
        state["story_profile"] = self._load_story_profile(work_id)
        state["updated_at"] = _now_iso()
        self._write_json(self._paths(work_id)["state"], state)

    def _shot_path(self, work_id: str, shot_id: int) -> Path:
        return self._paths(work_id)["shots"] / f"{_shot_key(shot_id)}.json"

    def _job_path(self, work_id: str, job_id: str) -> Path:
        return self._paths(work_id)["jobs"] / f"{job_id}.json"

    def _load_job(self, work_id: str, job_id: str) -> dict[str, Any]:
        data = self._read_json(self._job_path(work_id, job_id), {})
        return data if isinstance(data, dict) else {}

    def _save_job(self, work_id: str, job_id: str, job: dict[str, Any]) -> None:
        self._write_json(self._job_path(work_id, job_id), job)

    def _load_shot(self, work_id: str, shot_id: int) -> dict[str, Any]:
        data = self._read_json(self._shot_path(work_id, shot_id), {})
        return data if isinstance(data, dict) else {}

    def _save_shot(self, work_id: str, shot_id: int, shot: dict[str, Any]) -> None:
        shot["updated_at"] = _now_iso()
        self._write_json(self._shot_path(work_id, shot_id), shot)

    def _load_story_profile(self, work_id: str) -> dict[str, Any]:
        data = self._read_json(self._paths(work_id)["story_profile"], {})
        return data if isinstance(data, dict) else {}

    def _save_story_profile(self, work_id: str, story_profile: dict[str, Any]) -> None:
        previous = self._load_story_profile(work_id)
        profile = dict(story_profile)
        _preserve_story_profile_language(profile, previous)
        # Prefer session UI language when profile still has none.
        if "language" not in profile:
            from nanobot.session.generation_settings import get_generation_settings
            from nanobot.session.manager import SessionManager

            session = SessionManager(self.workspace).get_or_create(self._session_key.get())
            metadata = session.metadata if isinstance(session.metadata, dict) else {}
            settings = get_generation_settings(metadata)
            _apply_story_profile_language(profile, str(settings.get("language") or ""))
        _normalize_story_profile(profile)
        self._write_json(self._paths(work_id)["story_profile"], profile)

    def _default_state(self, work_id: str, *, title: str | None, goal: str) -> dict[str, Any]:
        return {
            "work_id": work_id,
            "title": title or "",
            "goal_brief": goal,
            "stage": self._DEFAULT_STAGE,
            "story_confirmed": False,
            "goal": {
                "shot_count": None,
                "shot_duration_sec": None,
                "generation_mode": "sequential",
            },
            "shots": {},
            "pending_remote_jobs": {},
            "latest_story_summary": "",
            "story_profile": {},
            "latest_merge_job_id": None,
            "final_output_path": None,
            "final_output_url": None,
            "reference_image": None,
            "reference_image_locked": False,
            "auto_generate": False,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }

    def _ensure_work_files(self, work_id: str, *, title: str | None, goal: str) -> dict[str, Any]:
        self._ensure_root()
        paths = self._paths(work_id)
        for key in ("work_dir", "shots", "jobs", "outputs"):
            paths[key].mkdir(parents=True, exist_ok=True)
        if not paths["story"].exists():
            self._write_text(paths["story"], "")
        if not paths["work_memory"].exists():
            self._write_text(paths["work_memory"], "# Work Memory Lite\n")
        if not paths["story_profile"].exists():
            from nanobot.session.generation_settings import get_generation_settings
            from nanobot.session.manager import SessionManager

            profile: dict[str, Any] = {}
            session = SessionManager(self.workspace).get_or_create(self._session_key.get())
            metadata = session.metadata if isinstance(session.metadata, dict) else {}
            settings = get_generation_settings(metadata)
            _apply_story_profile_language(profile, str(settings.get("language") or ""))
            self._write_json(paths["story_profile"], profile)
        if not paths["state"].exists():
            self._save_state(work_id, self._default_state(work_id, title=title, goal=goal))
        state = self._load_state(work_id)
        changed = False
        if not is_reference_image_locked(state):
            ref = self._read_reference_image_from_session()
            if ref and state.get("reference_image") != ref:
                state["reference_image"] = ref
                changed = True
        auto = self._session_auto_generate()
        if auto and not bool(state.get("auto_generate")):
            state["auto_generate"] = True
            changed = True
        if changed:
            self._save_state(work_id, state)
        self._refresh_fact(work_id, state)
        return state

    def _is_unfinished(self, state: dict[str, Any]) -> bool:
        stage = str(state.get("stage") or self._DEFAULT_STAGE)
        return stage not in self._FINAL_STAGES

    def _shot_entries(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        shots = state.get("shots", {})
        if not isinstance(shots, dict):
            return []
        items = []
        for shot_key, payload in shots.items():
            if isinstance(payload, dict):
                items.append({"shot_key": shot_key, **payload})
        return sorted(items, key=lambda item: int(item.get("shot_id", 0)))

    def _status_counts(self, state: dict[str, Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self._shot_entries(state):
            status = str(item.get("status") or "planned")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def _pending_remote_jobs(self, state: dict[str, Any]) -> dict[str, Any]:
        pending = state.setdefault("pending_remote_jobs", {})
        if not isinstance(pending, dict):
            pending = {}
            state["pending_remote_jobs"] = pending
        return pending

    def _register_pending_remote_job(self, state: dict[str, Any], job: dict[str, Any]) -> None:
        if job.get("status") != "queued":
            return
        pending = self._pending_remote_jobs(state)
        pending[str(job["job_id"])] = {
            "kind": job.get("kind"),
            "target": job.get("target"),
            "created_at": job.get("created_at"),
        }

    def _clear_pending_remote_job(self, state: dict[str, Any], job_id: str) -> None:
        pending = self._pending_remote_jobs(state)
        pending.pop(job_id, None)

    def _clear_pending_remote_jobs_for_target(
        self,
        state: dict[str, Any],
        kind: str,
        target: str | list[str],
    ) -> None:
        pending = self._pending_remote_jobs(state)
        target_key = json.dumps(target, ensure_ascii=False, sort_keys=True)
        stale_job_ids = [
            job_id
            for job_id, item in pending.items()
            if isinstance(item, dict)
            and item.get("kind") == kind
            and json.dumps(item.get("target"), ensure_ascii=False, sort_keys=True) == target_key
        ]
        for job_id in stale_job_ids:
            pending.pop(job_id, None)

    def _clear_pending_remote_jobs_for_shot(
        self,
        state: dict[str, Any],
        shot_id: int,
        *,
        kinds: set[str] | None = None,
    ) -> None:
        pending = self._pending_remote_jobs(state)
        shot_key = _shot_key(shot_id)
        stale_job_ids = []
        for job_id, item in pending.items():
            if not isinstance(item, dict):
                continue
            if kinds is not None and str(item.get("kind")) not in kinds:
                continue
            target = item.get("target")
            if target == shot_key or (
                isinstance(target, list) and any(str(value) == shot_key for value in target)
            ):
                stale_job_ids.append(job_id)
        for job_id in stale_job_ids:
            pending.pop(job_id, None)

    @staticmethod
    def _final_output_is_playable(locator: str) -> bool:
        raw = locator.strip()
        if not raw:
            return False
        parsed = urlparse(raw)
        if parsed.scheme in {"http", "https"}:
            return True
        suffix = Path(unquote(parsed.path or raw)).suffix.lower()
        return suffix in {".mp4", ".webm", ".mov", ".m4v", ".mkv"}

    def _sync_stage_from_state(self, state: dict[str, Any]) -> None:
        final_output = state.get("final_output_url") or state.get("final_output_path")
        if final_output and self._final_output_is_playable(str(final_output)):
            state["stage"] = "done"
            return
        if final_output:
            state["stage"] = "merging"
            return
        pending_remote_jobs = self._pending_remote_jobs(state)
        pending_kinds = {
            str(item.get("kind")) for item in pending_remote_jobs.values() if isinstance(item, dict)
        }
        if "merge_shot" in pending_kinds:
            state["stage"] = "merging"
            return
        if "generate_echo_shot" in pending_kinds:
            state["stage"] = "shot_generating"
            return
        current_stage = str(state.get("stage") or "")
        # Keep Memory review / generate-fail on 03 even if shot rows look idle.
        if current_stage in {"awaiting_memory_review", "failed"}:
            return
        shots = self._shot_entries(state)
        if any(item.get("status") in {"review_fail", "error"} for item in shots):
            state["stage"] = "shot_revising"
            return
        if any(item.get("status") in {"generated", "review_pass", "approved"} for item in shots):
            state["stage"] = "shot_reviewing"
            return
        if any(item.get("status") == "queued" for item in shots):
            state["stage"] = "shot_generating"
            return
        goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
        try:
            shot_count = int(goal.get("shot_count") or 0)
        except (TypeError, ValueError):
            shot_count = 0
        if current_stage in {
            "shot_generating",
            "shot_reviewing",
            "shot_revising",
            "merging",
        }:
            return
        # 02 分镜脚本 only after workplace confirm_story (「下一步」).
        # Chat set_director_goal / early shot files must not skip 策划剧本.
        if shot_count <= 0:
            if state.get("story_confirmed"):
                state["stage"] = "story_confirmed"
            else:
                state["stage"] = self._DEFAULT_STAGE
            return
        if current_stage == "shot_planning":
            return
        if state.get("story_confirmed"):
            state["stage"] = "story_confirmed"
            return
        state["stage"] = self._DEFAULT_STAGE

    def _refresh_fact(self, work_id: str, state: dict[str, Any]) -> str:
        paths = self._paths(work_id)
        story_exists = (
            paths["story"].exists() and paths["story"].read_text(encoding="utf-8").strip() != ""
        )
        story_profile = self._load_story_profile(work_id)
        goal = state.get("goal", {}) if isinstance(state.get("goal"), dict) else {}
        shot_items = self._shot_entries(state)
        counts = self._status_counts(state)
        pending_remote = self._pending_remote_jobs(state)
        lines = [
            "# Director Fact",
            "",
            f"- work_id: `{work_id}`",
            f"- work_dir: `{paths['work_dir']}`",
            f"- stage: `{state.get('stage', self._DEFAULT_STAGE)}`",
            f"- story_confirmed: `{bool(state.get('story_confirmed'))}`",
            f"- story_exists: `{story_exists}`",
            f"- story_profile_exists: `{bool(story_profile)}`",
            f"- goal_brief: {state.get('goal_brief') or '(empty)'}",
            f"- reference_image_present: `{reference_image_present(state.get('reference_image'))}`",
            f"- reference_image_locked: `{is_reference_image_locked(state)}`",
            f"- auto_generate: `{bool(state.get('auto_generate'))}`",
            f"- auto_generate_shot_count: `{self._effective_auto_generate_shot_count(goal)}`",
            f"- reference_image_needs_story_rewrite: `{self._session_reference_needs_rewrite()}`",
            f"- reference_image_inject_failed: `{self._session_reference_inject_failed()}`",
            "",
            "## Goal",
            "",
            f"- shot_count: `{goal.get('shot_count')}`",
            f"- shot_duration_sec: `{goal.get('shot_duration_sec')}`",
            f"- generation_mode: `{goal.get('generation_mode', 'sequential')}`",
            "",
            "## Progress",
            "",
            f"- total_shots: `{len(shot_items)}`",
            f"- pending_remote_jobs: `{len(pending_remote)}`",
        ]
        if counts:
            for status, count in sorted(counts.items()):
                lines.append(f"- {status}: `{count}`")
        else:
            lines.append("- shot_statuses: `(none yet)`")
        lines += [
            "",
            "## Paths",
            "",
            f"- state_json: `{paths['state']}`",
            f"- story_md: `{paths['story']}`",
            f"- story_profile_json: `{paths['story_profile']}`",
            f"- shots_dir: `{paths['shots']}`",
            f"- jobs_dir: `{paths['jobs']}`",
            f"- outputs_dir: `{paths['outputs']}`",
            "",
            "## Tool Ownership",
            "",
            "- Director state files are tool-owned. Use director tools instead of raw file edits whenever possible.",
        ]
        content = "\n".join(lines)
        self._write_text(paths["fact"], content)
        return content

    def _summary_from_shot(self, shot: dict[str, Any]) -> str:
        if isinstance(shot.get("summary"), str) and shot["summary"].strip():
            return shot["summary"].strip()
        caption = shot.get("caption")
        if isinstance(caption, str) and caption.strip():
            return caption.strip()[:160]
        prompt = shot.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()[:160]
        return ""

    def _shot_artifact_locator(self, shot: dict[str, Any]) -> str | None:
        artifact_url = shot.get("artifact_url")
        if isinstance(artifact_url, str) and artifact_url.strip():
            return artifact_url.strip()
        echo = shot.get("echo")
        if isinstance(echo, dict):
            result_url = echo.get("result_url")
            if isinstance(result_url, str) and result_url.strip():
                return result_url.strip()
        artifact_path = shot.get("artifact_path")
        if isinstance(artifact_path, str) and artifact_path.strip():
            return artifact_path.strip()
        remote_result = shot.get("remote_result")
        if isinstance(remote_result, dict):
            video_path = remote_result.get("video_path")
            if isinstance(video_path, str) and video_path.strip():
                return video_path.strip()
        return None

    def _state_shot_entry(self, shot: dict[str, Any]) -> dict[str, Any]:
        return {
            "shot_id": int(shot["shot_id"]),
            "status": shot.get("status"),
            "summary": self._summary_from_shot(shot),
            "cut": bool(shot.get("cut", True)),
            "has_shot_spec": bool(shot.get("caption")),
            "has_artifact": bool(self._shot_artifact_locator(shot)),
            "artifact_path": shot.get("artifact_path"),
            "artifact_url": shot.get("artifact_url"),
            "last_review": shot.get("last_review"),
            "review_notes": shot.get("review_notes") or "",
            "generation_error": shot.get("generation_error") or "",
            "updated_at": _now_iso(),
        }

    def _mark_shot_generation_error(
        self,
        work_id: str,
        shot_id: int,
        *,
        error_message: str,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        shot = self._load_shot(work_id, shot_id)
        if not shot:
            raise ValueError(f"Shot {shot_id} does not exist in work {work_id}.")
        shot["status"] = "error"
        shot["generation_error"] = error_message
        shot["last_review"] = "error"
        if job_id:
            shot["last_job_id"] = job_id
        echo = shot.get("echo")
        if isinstance(echo, dict):
            echo.update(
                {
                    "status": "failed",
                    "last_error": error_message,
                }
            )
        self._save_shot(work_id, shot_id, shot)
        return shot

    @staticmethod
    def _review_history(shot: dict[str, Any]) -> list[dict[str, Any]]:
        history = shot.get("review_history")
        if isinstance(history, list):
            return [item for item in history if isinstance(item, dict)]
        return []

    @staticmethod
    def _is_revised_prompt_update(shot: dict[str, Any]) -> bool:
        if str(shot.get("status") or "") == "review_fail":
            return True
        if shot.get("last_review") == "revise":
            return True
        return any(item.get("verdict") == "revise" for item in DirectorTool._review_history(shot))

    def _append_review_history(
        self,
        shot: dict[str, Any],
        *,
        verdict: str,
        review_source: str,
        feedback: str | None,
    ) -> None:
        history = self._review_history(shot)
        history.append(
            {
                "verdict": verdict,
                "source": review_source,
                "feedback": feedback or "",
                "created_at": _now_iso(),
            }
        )
        shot["review_history"] = history

    def _apply_shot_review(
        self,
        work_id: str,
        shot_id: int,
        *,
        verdict: str,
        review_source: str,
        feedback: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        shot = self._load_shot(work_id, shot_id)
        if not shot:
            raise ValueError(f"Shot {shot_id} does not exist in work {work_id}.")

        normalized_feedback = (
            feedback.strip() if isinstance(feedback, str) and feedback.strip() else None
        )
        if verdict == "revise" and not normalized_feedback:
            raise ValueError("feedback is required when verdict='revise'.")

        if verdict == "accept":
            shot["status"] = "approved"
            shot["last_review"] = "accepted"
            shot["review_notes"] = ""
            shot["approved_at"] = _now_iso()
        else:
            shot["status"] = "review_fail"
            shot["last_review"] = "revise"
            shot["review_notes"] = normalized_feedback or ""

        self._append_review_history(
            shot,
            verdict=verdict,
            review_source=review_source,
            feedback=normalized_feedback,
        )
        self._save_shot(work_id, shot_id, shot)

        state = self._load_state(work_id)
        if verdict == "revise":
            self._clear_pending_remote_jobs_for_shot(
                state,
                shot_id,
                kinds={"generate_echo_shot"},
            )
            state.pop("merge_confirmation_requested_at", None)
        shots = state.setdefault("shots", {})
        if isinstance(shots, dict):
            shots[_shot_key(shot_id)] = self._state_shot_entry(shot)
        self._sync_stage_from_state(state)
        self._save_state(work_id, state)
        self._refresh_fact(work_id, state)
        return shot, state

    def _normalize_reference_shot_ids(
        self,
        shot_id: int,
        reference_shot_ids: list[int],
        *,
        cut: bool,
    ) -> list[int]:
        normalized_reference_ids = sorted({int(item) for item in reference_shot_ids})
        if len(normalized_reference_ids) != len(reference_shot_ids):
            raise ValueError("reference_shot_ids must not contain duplicates.")
        if any(ref_id <= 0 for ref_id in normalized_reference_ids):
            raise ValueError("reference_shot_ids must contain positive shot IDs only.")
        if any(ref_id >= shot_id for ref_id in normalized_reference_ids):
            raise ValueError("reference_shot_ids must refer only to earlier shots.")
        if not cut and shot_id > 1 and (shot_id - 1) not in normalized_reference_ids:
            raise ValueError(
                "Shots with cut=false must include the immediately previous shot as a reference."
            )
        return normalized_reference_ids

    def _build_echo_payload(
        self,
        work_id: str,
        shot_id: int,
        reference_shot_ids: list[int],
        selection_note: str | None,
        condition_image_url: str | None = None,
        i2v_prompt: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
        shot = self._load_shot(work_id, shot_id)
        if not shot:
            raise ValueError(f"Shot {shot_id} does not exist in work {work_id}.")
        caption = shot.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Shot {shot_id} has no caption yet. Call create_shot_prompt first.")

        normalized_reference_ids = self._normalize_reference_shot_ids(
            shot_id,
            reference_shot_ids,
            cut=bool(shot.get("cut", True)),
        )

        reference_shots: list[dict[str, Any]] = []
        for ref_id in normalized_reference_ids:
            reference_shot = self._load_shot(work_id, ref_id)
            if not reference_shot:
                raise ValueError(f"Reference shot {ref_id} does not exist in work {work_id}.")
            reference_shots.append(
                {
                    "shot_id": ref_id,
                    "shot_key": _shot_key(ref_id),
                    "summary": self._summary_from_shot(reference_shot),
                    "cut": bool(reference_shot.get("cut", True)),
                    "artifact_url": reference_shot.get("artifact_url"),
                    "artifact_path": reference_shot.get("artifact_path"),
                }
            )

        state = self._load_state(work_id)
        story_profile = self._load_story_profile(work_id)
        duration_value = resolve_echo_duration_seconds(shot, state)
        num_frames = sync_shot_echo_duration(shot, duration_value)
        shot_payload: dict[str, Any] = {
            "shot_id": shot_id,
            "shot_key": _shot_key(shot_id),
            "cut": bool(shot.get("cut", True)),
            "summary": self._summary_from_shot(shot),
            "text": i2v_prompt.strip() if i2v_prompt else caption.strip(),
            "num_frames": num_frames,
            # Exact seconds matching num_frames (e.g. 177 → 7.08). A rounded
            # 7.0 would let the Echo service re-snap downward to 169 frames.
            "duration_sec": num_frames_to_exact_duration_sec(num_frames),
        }
        if condition_image_url:
            shot_payload["condition_image_url"] = condition_image_url
            shot_payload["generation_mode"] = "i2v"
        memory_slots = shot.get("approved_memory_slots")
        if isinstance(memory_slots, list) and memory_slots:
            shot_payload["memory_slots"] = memory_slots
        goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
        width = goal.get("width")
        height = goal.get("height")
        if width is not None:
            shot_payload["width"] = int(width)
        if height is not None:
            shot_payload["height"] = int(height)
        payload = {
            "work_id": work_id,
            "shot": shot_payload,
            "reference_shot_ids": normalized_reference_ids,
            "reference_shots": reference_shots,
            "selection_note": selection_note.strip()
            if isinstance(selection_note, str) and selection_note.strip()
            else None,
            "story_context": {
                "latest_story_summary": state.get("latest_story_summary") or None,
                "story_profile_summary": story_profile.get("summary")
                if isinstance(story_profile, dict)
                else None,
            },
        }
        return payload, shot, normalized_reference_ids

    def _build_r2v_payload(
        self,
        work_id: str,
        shot_id: int,
        *,
        prompt: str,
        num_frames: int,
        width: int | None = None,
        height: int | None = None,
        condition_image_url: str | None = None,
        memory_slots: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build request body for POST /r2v (unified T2V/I2V/R2V)."""
        payload: dict[str, Any] = {
            "work_id": work_id,
            "shot_id": _shot_key(shot_id),
            "prompt": prompt.strip(),
            "num_frames": num_frames,
            "memory_slots": memory_slots if isinstance(memory_slots, list) else [],
        }
        if condition_image_url:
            from nanobot.storage.files import outbound_file_url

            payload["condition_img"] = outbound_file_url(
                condition_image_url,
                workspace=self.workspace,
                work_id=work_id,
                name=f"request_assets/shot_{shot_id:03d}_condition.jpg",
                storage=self._tools_config.file_storage,
            )
        if width is not None:
            payload["width"] = int(width)
        if height is not None:
            payload["height"] = int(height)
        return payload

    def _build_memory_slots(
        self,
        approved_memory_slots: Any,
        reference_shot_ids: list[int],
        *,
        work_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve only human/auto-approved Memory slots for R2V.

        ``reference_shot_ids`` remains screenplay context and request metadata;
        it must never be converted into a media slot behind the user's back.
        """
        slots: list[dict[str, Any]] = (
            [dict(slot) for slot in approved_memory_slots if isinstance(slot, dict)]
            if isinstance(approved_memory_slots, list)
            else []
        )
        from nanobot.storage.files import outbound_file_url

        resolved_work_id = work_id or self._active_work_id() or "work"
        for index, slot in enumerate(slots):
            for key in ("image_url", "audio_url"):
                value = slot.get(key)
                if isinstance(value, str) and value.strip():
                    suffix = Path(value.split("?", 1)[0]).suffix
                    if not suffix:
                        suffix = ".jpg" if key == "image_url" else ".wav"
                    slot[key] = outbound_file_url(
                        value,
                        workspace=self.workspace,
                        work_id=resolved_work_id,
                        name=f"request_assets/slot_{index:02d}_{key}{suffix}",
                        storage=self._tools_config.file_storage,
                    )
        if len(slots) > 7:
            raise ValueError("approved memory slots cannot exceed 7")
        return slots

    def _build_merge_payload(
        self,
        work_id: str,
        shot_ids: list[int],
        selected_shots: list[dict[str, Any]],
    ) -> dict[str, Any]:
        from nanobot.storage.files import LocalFilePublisher, resolve_local_asset_path

        inputs: list[dict[str, Any]] = []
        for shot in selected_shots:
            shot_id = int(shot["shot_id"])
            echo = shot.get("echo")
            version_id = echo.get("version_id") if isinstance(echo, dict) else None
            item: dict[str, Any] = {"shot_id": shot_id}
            source = None
            if isinstance(echo, dict):
                source = echo.get("result_url") or echo.get("base_result_url")
            source = source or shot.get("artifact_url") or shot.get("artifact_path")
            if isinstance(source, str) and source.strip():
                source = source.strip()
                if source.startswith(("http://", "https://")):
                    item["video_url"] = source
                else:
                    local_config = self._tools_config.file_storage.local
                    local_path = resolve_local_asset_path(
                        source,
                        workspace=self.workspace,
                        config=local_config,
                    ) or Path(source).expanduser()
                    if not str(local_config.base_url).strip():
                        raise ValueError(
                            "tools.fileStorage.local.baseUrl is required to merge a local video."
                        )
                    else:
                        item["video_url"] = LocalFilePublisher(
                            local_config,
                            workspace=self.workspace,
                            work_id=work_id,
                        )(str(local_path), f"merge_inputs/shot_{shot_id:03d}.mp4")
            elif isinstance(version_id, str) and version_id.strip():
                # Version records are process-local on the public Echo server and
                # disappear after a restart. Use them only when no durable artifact
                # locator was saved with the completed shot.
                item["version_id"] = version_id.strip()
            else:
                raise ValueError(
                    f"Shot {shot_id} has no Echo version or playable video artifact."
                )
            inputs.append(item)
        return {
            "work_id": work_id,
            "shot_ids": shot_ids,
            "shots": inputs,
        }

    def _submit_remote_request(
        self,
        work_id: str,
        job_id: str,
        request_payload: dict[str, Any],
        *,
        target: str | list[str],
        operation: str,
    ) -> dict[str, Any]:
        payload_path = self._paths(work_id)["outputs"] / f"{job_id}.payload.json"
        envelope = self._build_remote_request_envelope(
            operation,
            work_id,
            job_id,
            target,
            request_payload,
        )
        request_path = self._paths(work_id)["outputs"] / f"{job_id}.request.json"
        self._write_json(payload_path, request_payload)
        self._write_json(request_path, envelope)

        job: dict[str, Any] = {
            "job_id": job_id,
            "kind": operation,
            "status": "queued",
            "work_id": work_id,
            "target": target,
            "created_at": _now_iso(),
            "completed_at": None,
            "request_payload_path": str(payload_path),
            "request_envelope_path": str(request_path),
            "remote": {
                "transport": "http",
                "protocol_version": _REMOTE_PROTOCOL_VERSION,
                "endpoint_path": self._remote_endpoint_path(operation),
                "remote_task_id": None,
                "callback_expected": True,
                "callback_contract": envelope["callback"],
            },
        }

        callback_url = envelope.get("callback", {}).get("url")
        if not isinstance(callback_url, str) or not callback_url.strip():
            raise RuntimeError(
                "Echo callback URL is not configured. "
                "Set tools.echoGenerator.callbackBaseUrl to the local Agent URL."
            )

        EchoAdmissionController.from_tools_config(self._tools_config).ensure_allowed(
            operation=operation,
        )

        base_url = self._remote_http_base_url()
        if not base_url:
            raise RuntimeError(
                "No Echo generator base URL configured. "
                "Set tools.echoGenerator.baseUrl for real HTTP submission."
            )
        endpoint_url = f"{base_url}{self._remote_endpoint_path(operation)}"
        remote_ack = self._post_remote_http_request(endpoint_url, envelope)
        remote_task_id = remote_ack.get("remote_task_id") or remote_ack.get("task_id")
        job["remote"].update(
            {
                "endpoint_url": endpoint_url,
                "remote_task_id": remote_task_id,
                "version_id": remote_ack.get("version_id"),
                "status_url": remote_ack.get("status_url"),
                "ack": remote_ack,
            }
        )
        return job

    def _submit_r2v_request(
        self,
        work_id: str,
        job_id: str,
        request_payload: dict[str, Any],
        *,
        target: str,
    ) -> dict[str, Any]:
        """Submit a generation job via POST /r2v (no director envelope)."""
        payload_path = self._paths(work_id)["outputs"] / f"{job_id}.payload.json"
        request_path = self._paths(work_id)["outputs"] / f"{job_id}.request.json"
        session = {
            "session_key": self._session_key.get() if self._session_key.get() else None,
            "channel": self._channel.get() if self._channel.get() else None,
            "chat_id": self._chat_id.get() if self._chat_id.get() else None,
        }
        callback_url = self._remote_callback_url("generate_echo_shot")
        outbound_payload = dict(request_payload)
        outbound_payload.update(
            {
                "job_id": job_id,
                "callback_context": session,
            }
        )
        if callback_url:
            outbound_payload["callback_url"] = callback_url
        self._write_json(payload_path, request_payload)
        self._write_json(request_path, outbound_payload)

        job: dict[str, Any] = {
            "job_id": job_id,
            "kind": "generate_echo_shot",
            "status": "queued",
            "work_id": work_id,
            "target": target,
            "created_at": _now_iso(),
            "completed_at": None,
            "request_payload_path": str(payload_path),
            "request_envelope_path": str(request_path),
            "remote": {
                "transport": "http",
                "endpoint_path": self._remote_endpoint_path("r2v_generate"),
                "remote_task_id": None,
                "r2v": True,
                "callback_expected": True,
                "callback_url": callback_url,
            },
            "session": session,
        }
        if not callback_url:
            raise RuntimeError(
                "Echo callback URL is not configured. "
                "Set tools.echoGenerator.callbackBaseUrl to the local Agent URL."
            )

        EchoAdmissionController.from_tools_config(self._tools_config).ensure_allowed(
            operation="r2v_generate",
        )

        base_url = self._remote_http_base_url()
        if not base_url:
            raise RuntimeError(
                "No Echo generator base URL configured. "
                "Set tools.echoGenerator.baseUrl in config."
            )
        endpoint_url = f"{base_url}{self._remote_endpoint_path('r2v_generate')}"

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        body = json.dumps(outbound_payload, ensure_ascii=False).encode("utf-8")
        # This request only submits the job. Rendering completes through the
        # callback, so it must not inherit a generation-length timeout.
        timeout = self._remote_http_timeout_sec()
        last_exc: Exception | None = None
        raw = ""
        for attempt in range(1, _R2V_SUBMIT_ATTEMPTS + 1):
            req = urllib_request.Request(
                endpoint_url, data=body, headers=headers, method="POST"
            )
            try:
                with urllib_request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                last_exc = None
                break
            except urllib_error.HTTPError as exc:
                last_exc = exc
                transient = exc.code in _R2V_TRANSIENT_HTTP_CODES
                try:
                    response_detail = exc.read(4096).decode("utf-8", errors="replace").strip()
                except Exception:  # noqa: BLE001 - HTTPError may have no response stream.
                    response_detail = ""
                logger.error(
                    "R2V submit HTTP {} work_id={} job_id={} attempt={}/{} "
                    "transient={} error={} response={}",
                    exc.code,
                    work_id,
                    job_id,
                    attempt,
                    _R2V_SUBMIT_ATTEMPTS,
                    transient,
                    exc,
                    response_detail or "-",
                )
                if not transient or attempt >= _R2V_SUBMIT_ATTEMPTS:
                    suffix = f"; response: {response_detail}" if response_detail else ""
                    raise RuntimeError(
                        f"R2V submit failed with HTTP {exc.code}: {exc}{suffix}"
                    ) from exc
                time.sleep(min(2 * attempt, 6))
            except (urllib_error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                if is_connection_refused(exc):
                    logger.error(
                        "R2V submit unreachable work_id={} job_id={} error={}",
                        work_id,
                        job_id,
                        exc,
                    )
                    raise EchoGeneratorUnavailableError(UNAVAILABLE_MESSAGE) from exc
                logger.error(
                    "R2V submit failed work_id={} job_id={} attempt={}/{} "
                    "retryable={} timeout_s={} error={}",
                    work_id,
                    job_id,
                    attempt,
                    _R2V_SUBMIT_ATTEMPTS,
                    True,
                    timeout,
                    exc,
                )
                if attempt >= _R2V_SUBMIT_ATTEMPTS:
                    raise RuntimeError(f"R2V submit failed: {exc}") from exc
                time.sleep(min(2 * attempt, 6))
        if last_exc is not None:
            raise RuntimeError(f"R2V submit failed: {last_exc}") from last_exc

        try:
            remote_ack = json.loads(raw) if raw.strip() else {}
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"R2V submit returned invalid JSON: {exc}") from exc

        if not isinstance(remote_ack, dict):
            raise RuntimeError(
                f"R2V submit returned unexpected payload type: {type(remote_ack).__name__}"
            )

        version_id = remote_ack.get("version_id")
        if not isinstance(version_id, str) or not version_id.strip():
            raise RuntimeError(
                "R2V submit response is missing a non-empty 'version_id'"
            )

        task_id = remote_ack.get("task_id")
        status_url = remote_ack.get("status_url") or f"/version/{version_id}"
        job["remote"].update(
            {
                "endpoint_url": endpoint_url,
                "remote_task_id": task_id,
                "version_id": version_id,
                "status_url": status_url,
                "ack": remote_ack,
            }
        )
        return job

    @staticmethod
    def _target_shot_ids(target: Any) -> list[int]:
        if isinstance(target, list):
            shot_ids: list[int] = []
            for item in target:
                if isinstance(item, str):
                    shot_ids.append(_shot_id_from_key(item))
                elif isinstance(item, int):
                    shot_ids.append(item)
            return shot_ids
        if isinstance(target, str):
            return [_shot_id_from_key(target)]
        if isinstance(target, int):
            return [target]
        return []

    @staticmethod
    def _extract_remote_stored_locator(*sources: dict[str, Any]) -> str | None:
        """Resolve the canonical stored locator from a remote callback payload.

        Preference order: configured ``asset_urls`` entries, then direct HTTP-style URLs.
        PFS/local paths are ignored because the browser cannot play them.
        """
        from nanobot.integrations.remote_video_url import resolve_public_video_url

        return resolve_public_video_url(*sources)

    def _normalize_echo_callback_result(
        self,
        callback_payload: dict[str, Any],
        job: dict[str, Any],
    ) -> dict[str, Any]:
        default_shot_ids = self._target_shot_ids(job.get("target"))
        default_shot_id = default_shot_ids[0] if default_shot_ids else None
        raw_result = callback_payload.get("result")
        sources: list[dict[str, Any]] = [callback_payload]
        if isinstance(raw_result, dict):
            sources.insert(0, raw_result)
            shot_id = raw_result.get("shot_id", default_shot_id)
        else:
            shot_id = callback_payload.get("shot_id", default_shot_id)
        result_url = self._extract_remote_stored_locator(*sources)
        if shot_id is None or not result_url:
            raise ValueError(
                "Echo callback requires shot_id plus a public asset_urls URL or result_url, "
                "either at the top level or inside result={...}."
            )
        normalized_shot_id = (
            _shot_id_from_key(shot_id)
            if isinstance(shot_id, str) and shot_id.startswith("shot_")
            else int(shot_id)
        )
        normalized: dict[str, Any] = {
            "shot_id": normalized_shot_id,
            "result_url": result_url,
        }
        for key in ("video_id",):
            value = next((source.get(key) for source in sources if source.get(key)), None)
            if value is not None:
                normalized[key] = value
        return normalized

    def apply_echo_callback_payload(self, callback_payload: dict[str, Any]) -> dict[str, Any]:
        work_id = callback_payload.get("work_id")
        job_id = callback_payload.get("job_id")
        if not isinstance(work_id, str) or not work_id.strip():
            raise ValueError("Echo callback missing work_id.")
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("Echo callback missing job_id.")

        job = self._load_job(work_id, job_id)
        if not job:
            raise ValueError(f"Director job '{job_id}' was not found for work '{work_id}'.")
        if job.get("kind") != "generate_echo_shot":
            raise ValueError(f"Director job '{job_id}' is not a generate_echo_shot job.")

        existing_status = str(job.get("status") or "")
        if existing_status in {"completed", "failed"}:
            result_url = job.get("result_url")
            return {
                "status": existing_status,
                "operation": "generate_echo_shot",
                "work_id": work_id,
                "job_id": job_id,
                "duplicate": True,
                "result_urls": [result_url]
                if isinstance(result_url, str) and result_url.strip()
                else [],
                "updated_shots": [],
            }

        status = str(callback_payload.get("status") or "completed")
        if status not in {"completed", "failed"}:
            raise ValueError("Echo callback status must be 'completed' or 'failed'.")

        state = self._load_state(work_id)
        result = (
            self._normalize_echo_callback_result(callback_payload, job)
            if status == "completed"
            else None
        )
        default_shot_ids = self._target_shot_ids(job.get("target"))
        target_shot_id = (
            int(result["shot_id"])
            if isinstance(result, dict)
            else (default_shot_ids[0] if default_shot_ids else None)
        )
        if target_shot_id is None:
            raise ValueError(
                "Echo callback could not determine shot_id from callback or job target."
            )

        shot = self._load_shot(work_id, target_shot_id)
        if not shot:
            raise ValueError(f"Shot {target_shot_id} does not exist in work {work_id}.")

        current_status = str(shot.get("status") or "")
        if current_status in {"prompt_ready", "revised_prompt_ready", "planned"}:
            job["status"] = status
            job["completed_at"] = callback_payload.get("completed_at") or _now_iso()
            remote = job.get("remote")
            if not isinstance(remote, dict):
                remote = {}
                job["remote"] = remote
            remote["callback_received_at"] = _now_iso()
            remote["ignored_stale_callback"] = True
            if callback_payload.get("remote_task_id"):
                remote["remote_task_id"] = callback_payload.get("remote_task_id")
            job["callback_payload"] = callback_payload
            self._save_job(work_id, job_id, job)
            self._clear_pending_remote_job(state, job_id)
            self._save_state(work_id, state)
            return {
                "status": "ignored",
                "operation": "generate_echo_shot",
                "work_id": work_id,
                "job_id": job_id,
                "shot_id": target_shot_id,
                "reason": f"shot is {current_status} after replan; callback ignored",
            }

        job["status"] = status
        job["completed_at"] = callback_payload.get("completed_at") or _now_iso()
        remote = job.get("remote")
        if not isinstance(remote, dict):
            remote = {}
            job["remote"] = remote
        remote["callback_received_at"] = _now_iso()
        if callback_payload.get("remote_task_id"):
            remote["remote_task_id"] = callback_payload.get("remote_task_id")
        job["callback_payload"] = callback_payload

        echo = shot.get("echo")
        if not isinstance(echo, dict):
            echo = {}
            shot["echo"] = echo

        updated_shots: list[dict[str, Any]] = []
        result_urls: list[str] = []
        if status == "completed" and isinstance(result, dict):
            result_url = result["result_url"]
            shot["status"] = "generated"
            shot.pop("generation_error", None)
            shot["artifact_url"] = result_url
            shot["last_job_id"] = job_id
            echo.update(
                {
                    "status": "completed",
                    "result_url": result_url,
                    "completed_at": job.get("completed_at"),
                    "callback_received_at": remote.get("callback_received_at"),
                    "remote_task_id": remote.get("remote_task_id"),
                }
            )
            for key in ("video_id",):
                if result.get(key) is not None:
                    echo[key] = result[key]
            job["result_url"] = result_url
            updated_shots.append(
                {
                    "shot_id": target_shot_id,
                    "shot_key": _shot_key(target_shot_id),
                    "result_url": result_url,
                }
            )
            result_urls.append(result_url)
        else:
            error_message = callback_payload.get("error") or "Remote Echo generation failed."
            job["error"] = error_message
            shot = self._mark_shot_generation_error(
                work_id,
                target_shot_id,
                error_message=error_message,
                job_id=job_id,
            )
            echo = shot.get("echo")
            if isinstance(echo, dict):
                echo.update(
                    {
                        "callback_received_at": remote.get("callback_received_at"),
                        "remote_task_id": remote.get("remote_task_id"),
                    }
                )
                shot["echo"] = echo
                self._save_shot(work_id, target_shot_id, shot)

        if status == "completed":
            self._save_shot(work_id, target_shot_id, shot)
        shots = state.setdefault("shots", {})
        if isinstance(shots, dict):
            shots[_shot_key(target_shot_id)] = self._state_shot_entry(shot)
        self._save_job(work_id, job_id, job)
        self._clear_pending_remote_job(state, job_id)
        self._sync_stage_from_state(state)
        self._save_state(work_id, state)
        self._refresh_fact(work_id, state)

        if status == "completed":
            message = prompts.text(
                "director.callback.generate_echo_shot.completed",
                work_id=work_id,
                shot_key=_shot_key(target_shot_id),
                result_url=result_urls[0],
            )
        else:
            message = prompts.text(
                "director.callback.generate_echo_shot.failed",
                work_id=work_id,
                shot_key=_shot_key(target_shot_id),
                error=job.get("error"),
            )
        return {
            "status": status,
            "operation": "generate_echo_shot",
            "work_id": work_id,
            "job_id": job_id,
            "result_urls": result_urls,
            "updated_shots": updated_shots,
            "injection_message": message,
            "session_key": callback_payload.get("session_key"),
            "channel": callback_payload.get("channel"),
            "chat_id": callback_payload.get("chat_id"),
        }

    @staticmethod
    def _normalize_merge_callback_result(callback_payload: dict[str, Any]) -> dict[str, str | None]:
        raw_result = callback_payload.get("result")
        sources: list[dict[str, Any]] = [callback_payload]
        if isinstance(raw_result, dict):
            sources.insert(0, raw_result)

        stored_locator = DirectorTool._extract_remote_stored_locator(*sources)

        artifact_path = None
        for source in sources:
            candidate = source.get("artifact_path") or source.get("output_path")
            if isinstance(candidate, str) and candidate.strip():
                artifact_path = candidate.strip()
                break

        normalized_path = artifact_path
        normalized_url = stored_locator
        if not normalized_path and not normalized_url:
            raise ValueError(
                "Merge callback requires a public asset_urls URL, artifact_path, "
                "or artifact_url/result_url, "
                "either at the top level or inside result={...}."
            )
        return {
            "artifact_path": normalized_path,
            "artifact_url": normalized_url,
        }

    def apply_merge_callback_payload(self, callback_payload: dict[str, Any]) -> dict[str, Any]:
        work_id = callback_payload.get("work_id")
        job_id = callback_payload.get("job_id")
        if not isinstance(work_id, str) or not work_id.strip():
            raise ValueError("Merge callback missing work_id.")
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("Merge callback missing job_id.")

        job = self._load_job(work_id, job_id)
        if not job:
            raise ValueError(f"Director job '{job_id}' was not found for work '{work_id}'.")
        if job.get("kind") != "merge_shot":
            raise ValueError(f"Director job '{job_id}' is not a merge_shot job.")

        existing_status = str(job.get("status") or "")
        if existing_status in {"completed", "failed"}:
            final_output = job.get("artifact_url") or job.get("artifact_path")
            return {
                "status": existing_status,
                "operation": "merge_shot",
                "work_id": work_id,
                "job_id": job_id,
                "duplicate": True,
                "final_output": final_output,
                "final_output_path": job.get("artifact_path"),
                "final_output_url": job.get("artifact_url"),
                "media": [final_output]
                if isinstance(final_output, str) and final_output.strip()
                else [],
            }

        status = str(callback_payload.get("status") or "completed")
        if status not in {"completed", "failed"}:
            raise ValueError("Merge callback status must be 'completed' or 'failed'.")

        state = self._load_state(work_id)
        result = (
            self._normalize_merge_callback_result(callback_payload)
            if status == "completed"
            else None
        )

        job["status"] = status
        job["completed_at"] = callback_payload.get("completed_at") or _now_iso()
        remote = job.get("remote")
        if not isinstance(remote, dict):
            remote = {}
            job["remote"] = remote
        remote["callback_received_at"] = _now_iso()
        if callback_payload.get("remote_task_id"):
            remote["remote_task_id"] = callback_payload.get("remote_task_id")
        job["callback_payload"] = callback_payload

        final_output_path = None
        final_output_url = None
        if status == "completed" and isinstance(result, dict):
            state.pop("generation_error", None)
            final_output_path = result["artifact_path"]
            final_output_url = result["artifact_url"]
            state["final_output_path"] = final_output_path
            state["final_output_url"] = final_output_url
            if final_output_path is not None:
                job["artifact_path"] = final_output_path
            if final_output_url is not None:
                job["artifact_url"] = final_output_url
        else:
            error_message = callback_payload.get("error") or "Remote merge failed."
            job["error"] = error_message
            state["generation_error"] = error_message
            state["stage"] = "failed"

        self._save_job(work_id, job_id, job)
        self._clear_pending_remote_job(state, job_id)
        self._sync_stage_from_state(state)
        self._save_state(work_id, state)
        self._refresh_fact(work_id, state)

        final_output = final_output_url or final_output_path
        if status == "completed":
            message = prompts.text(
                "director.callback.merge_shot.completed",
                work_id=work_id,
                final_output=final_output,
            )
        else:
            message = prompts.text(
                "director.callback.merge_shot.failed",
                work_id=work_id,
                job_id=job_id,
                error=job.get("error"),
            )
        return {
            "status": status,
            "operation": "merge_shot",
            "work_id": work_id,
            "job_id": job_id,
            "final_output": final_output,
            "final_output_path": final_output_path,
            "final_output_url": final_output_url,
            "media": [final_output]
            if isinstance(final_output, str) and final_output.strip()
            else [],
            "injection_message": message,
            "session_key": callback_payload.get("session_key"),
            "channel": callback_payload.get("channel"),
            "chat_id": callback_payload.get("chat_id"),
        }

    def _next_action(self, state: dict[str, Any]) -> str:
        if not state.get("story_confirmed"):
            return ""
        goal = state.get("goal", {}) if isinstance(state.get("goal"), dict) else {}
        if goal.get("shot_count") in (None, 0):
            return ""
        shot_count = int(goal.get("shot_count") or 0)
        shots = self._shot_entries(state)
        if len(shots) < shot_count:
            if self._session_auto_generate() or bool(state.get("auto_generate")):
                return ""
            return prompts.text("director.next_action.storyboard_ready")
        if any(
            item.get("status") in {"planned", "prompt_ready", "revised_prompt_ready", "queued"}
            for item in shots
        ):
            return prompts.text("director.next_action.start_generation")
        if state.get("review_completed_at"):
            return prompts.text("director.next_action.review_complete")
        if not state.get("final_output_path"):
            return prompts.text("director.next_action.merge_ready")
        return prompts.text("director.next_action.work_complete")

    def _confirm(self, work_id: str) -> dict[str, Any]:
        state = self._load_state(work_id)
        self._sync_stage_from_state(state)
        self._save_state(work_id, state)
        fact = self._refresh_fact(work_id, state)
        goal = state.get("goal", {}) if isinstance(state.get("goal"), dict) else {}
        shots = self._shot_entries(state)
        approved = sum(1 for item in shots if item.get("status") in {"review_pass", "approved"})
        generated = sum(1 for item in shots if item.get("status") == "generated")
        return {
            "work_id": work_id,
            "stage": state.get("stage", self._DEFAULT_STAGE),
            "story_confirmed": bool(state.get("story_confirmed")),
            "goal": goal,
            "story_exists": self._paths(work_id)["story"].read_text(encoding="utf-8").strip() != "",
            "shot_total": len(shots),
            "shot_generated": generated,
            "shot_approved": approved,
            "next_recommended_action": self._next_action(state),
            "fact_md": fact,
        }


@tool_parameters(
    tool_parameters_schema(
        goal=StringSchema("Brief description of the user's intended video or story work"),
        title=StringSchema("Optional short title for the work"),
        continue_policy=StringSchema(
            "How to handle an unfinished existing work: ask, resume, or new",
            enum=["ask", "resume", "new"],
        ),
        required=["goal"],
    )
)
class StartDirectorTool(DirectorTool):
    @property
    def name(self) -> str:
        return "start_director"

    async def execute(
        self,
        goal: str,
        title: str | None = None,
        continue_policy: str = "ask",
        **kwargs: Any,
    ) -> str:
        # Check current active work first, then scan history for unfinished works
        existing_work_id: str | None = None
        existing_state: dict[str, Any] | None = None

        active_id = self._active_work_id()
        if active_id:
            state = self._load_state(active_id)
            if state and self._is_unfinished(state):
                existing_work_id = active_id
                existing_state = state

        if not existing_work_id:
            for hist_id in reversed(self._session_work_history()):
                if hist_id == active_id:
                    continue
                state = self._load_state(hist_id)
                if state and self._is_unfinished(state):
                    existing_work_id = hist_id
                    existing_state = state
                    break

        if existing_work_id and continue_policy == "ask":
            return _json_dump(
                {
                    "status": "needs_confirmation",
                    "message": "An unfinished director work already exists for this session.",
                    "existing_work_id": existing_work_id,
                    "stage": existing_state.get("stage") if existing_state else None,
                    "goal_brief": existing_state.get("goal_brief") if existing_state else None,
                    "next_step": "Ask the user whether to continue the existing work or create a new one.",
                }
            )
        if existing_work_id and continue_policy == "resume":
            self._set_active_work(existing_work_id)
            return _json_dump(
                {
                    "status": "resumed",
                    "work_id": existing_work_id,
                    "stage": existing_state.get("stage") if existing_state else None,
                    "goal_brief": existing_state.get("goal_brief") if existing_state else None,
                }
            )

        slug_source = title or goal[:48]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        work_id = f"work-{stamp}-{_slugify(slug_source, fallback='video')}"
        state = self._ensure_work_files(work_id, title=title, goal=goal)
        self._set_active_work(work_id)
        return _json_dump(
            {
                "status": "created",
                "work_id": work_id,
                "work_dir": str(self._paths(work_id)["work_dir"]),
                "stage": state.get("stage"),
                "goal_brief": goal,
            }
        )


@tool_parameters(
    tool_parameters_schema(
        work_id=StringSchema(
            "Optional explicit work ID; defaults to the active work", nullable=True
        ),
        shot_count=IntegerSchema(description="Target number of shots", minimum=1, nullable=True),
        shot_duration_sec=IntegerSchema(
            description="Nominal duration per shot in seconds",
            minimum=1,
            nullable=True,
        ),
        generation_mode=StringSchema(
            "Shot generation mode",
            enum=["sequential", "parallel"],
            nullable=True,
        ),
    )
)
class SetDirectorGoalTool(DirectorTool):
    @property
    def name(self) -> str:
        return "set_director_goal"

    async def execute(
        self,
        work_id: str | None = None,
        shot_count: int | None = None,
        shot_duration_sec: int | None = None,
        generation_mode: str | None = None,
        **kwargs: Any,
    ) -> str:
        resolved_work_id, _ = self._resolve_work_id(work_id)
        if not resolved_work_id:
            return "Error: No active director work. Call start_director first."
        if all(
            value is None
            for value in (shot_count, shot_duration_sec, generation_mode)
        ):
            return "Error: At least one goal field must be provided."
        state = self._load_state(resolved_work_id)
        goal = state.setdefault("goal", {})
        if not isinstance(goal, dict):
            goal = {}
            state["goal"] = goal
        try:
            previous_shot_count = int(goal.get("shot_count") or 0)
        except (TypeError, ValueError):
            previous_shot_count = 0
        if shot_count is not None:
            goal["shot_count"] = shot_count
            lock_reference_image(state)
            self._lock_session_reference_image()
            try:
                new_shot_count = int(shot_count)
            except (TypeError, ValueError):
                new_shot_count = 0
            if (
                previous_shot_count <= 0
                and new_shot_count > 0
                and not self._session_auto_generate()
                and not bool(state.get("auto_generate"))
            ):
                state[SHOT_COUNT_NEXT_STEP_HINT_PENDING_KEY] = True
        if shot_duration_sec is not None:
            goal["shot_duration_sec"] = shot_duration_sec
        if generation_mode is not None:
            goal["generation_mode"] = generation_mode
        self._sync_stage_from_state(state)
        self._save_state(resolved_work_id, state)
        self._refresh_fact(resolved_work_id, state)
        return _json_dump(
            {
                "status": "ok",
                "work_id": resolved_work_id,
                "goal": goal,
                "stage": state.get("stage"),
            }
        )


@tool_parameters(
    tool_parameters_schema(
        work_id=StringSchema(
            "Optional explicit work ID; defaults to the active work", nullable=True
        ),
        include_shots=BooleanSchema(description="Include per-shot summary rows", default=True),
        include_jobs=BooleanSchema(description="Include recent job rows", default=True),
        limit=IntegerSchema(description="Maximum shots/jobs to return", minimum=1, maximum=200),
    )
)
class GetWorkplaceStatusTool(DirectorTool):
    @property
    def name(self) -> str:
        return "get_workplace_status"

    async def execute(
        self,
        work_id: str | None = None,
        include_shots: bool = True,
        include_jobs: bool = True,
        limit: int = 20,
        **kwargs: Any,
    ) -> str:
        resolved_work_id, work_dir = self._resolve_work_id(work_id)
        if not resolved_work_id or not work_dir:
            return "Error: No active director work. Call start_director first."
        state = self._load_state(resolved_work_id)
        self._sync_stage_from_state(state)
        shots = self._shot_entries(state)
        goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
        payload: dict[str, Any] = {
            "work_id": resolved_work_id,
            "work_dir": str(work_dir),
            "stage": state.get("stage"),
            "story_confirmed": bool(state.get("story_confirmed")),
            "goal_brief": state.get("goal_brief"),
            "goal": state.get("goal", {}),
            "final_output_path": state.get("final_output_path"),
            "final_output_url": state.get("final_output_url"),
            "reference_image_present": reference_image_present(state.get("reference_image")),
            "reference_image_locked": is_reference_image_locked(state),
            "auto_generate": bool(state.get("auto_generate")),
            "auto_generate_shot_count": self._effective_auto_generate_shot_count(goal),
            "reference_image_needs_story_rewrite": self._session_reference_needs_rewrite(),
            "counts": self._status_counts(state),
            "pending_remote_jobs": self._pending_remote_jobs(state),
            "story_path": str(self._paths(resolved_work_id)["story"]),
            "story_profile_path": str(self._paths(resolved_work_id)["story_profile"]),
            "fact_path": str(self._paths(resolved_work_id)["fact"]),
            "next_recommended_action": self._next_action(state),
            # Only assets with a textual profile are exposed to the agent.
            # Binary media stays in the local workspace and is resolved only
            # after a human approves the recommendation.
            "memory_assets": self._memory_asset_catalog(resolved_work_id),
        }
        if include_shots:
            payload["shots"] = [
                {
                    "shot_id": item.get("shot_id"),
                    "shot_key": item.get("shot_key"),
                    "status": item.get("status"),
                    "summary": item.get("summary"),
                    "cut": bool(item.get("cut", True)),
                    "has_shot_spec": bool(item.get("has_shot_spec")),
                    "has_artifact": bool(item.get("artifact_path") or item.get("artifact_url")),
                    "artifact_path": item.get("artifact_path"),
                    "artifact_url": item.get("artifact_url"),
                    "last_review": item.get("last_review"),
                    "review_notes": item.get("review_notes") or "",
                }
                for item in shots[:limit]
            ]
        if include_jobs:
            jobs_dir = self._paths(resolved_work_id)["jobs"]
            jobs: list[dict[str, Any]] = []
            for path in sorted(jobs_dir.glob("*.json"), reverse=True)[:limit]:
                data = self._read_json(path, {})
                if isinstance(data, dict):
                    jobs.append(
                        {
                            "job_id": data.get("job_id"),
                            "kind": data.get("kind"),
                            "status": data.get("status"),
                            "target": data.get("target"),
                            "created_at": data.get("created_at"),
                        }
                    )
            payload["jobs"] = jobs
        return _json_dump(payload)


@tool_parameters(
    tool_parameters_schema(
        work_id=StringSchema(
            "Optional explicit work ID; defaults to the active work", nullable=True
        ),
    )
)
class GetStoryTool(DirectorTool):
    @property
    def name(self) -> str:
        return "get_story"

    async def execute(self, work_id: str | None = None, **kwargs: Any) -> str:
        resolved_work_id, _ = self._resolve_work_id(work_id)
        if not resolved_work_id:
            return "Error: No active director work. Call start_director first."
        paths = self._paths(resolved_work_id)
        return _json_dump(
            {
                "work_id": resolved_work_id,
                "story_md": paths["story"].read_text(encoding="utf-8"),
                "story_profile": self._load_story_profile(resolved_work_id),
            }
        )


@tool_parameters(
    tool_parameters_schema(
        topic=StringSchema(
            "Guidance topic to load, e.g. 'shot-sequence-patterns' or 'shot-prompt-writer'"
        ),
        required=["topic"],
    )
)
class GetGuidanceTool(DirectorTool):
    @property
    def name(self) -> str:
        return "get_guidance"

    async def execute(self, topic: str = "", **kwargs: Any) -> str:
        manager = PEManager.instance()
        active = manager.active_for_session(self._session_key.get())
        path = manager.resolve_reference(topic, name=active)
        if path is None:
            available = ", ".join(manager.list_references(name=active)) or "(none)"
            return prompts.text(
                "director.guidance.not_found", topic=topic, available=available
            )
        return path.read_text(encoding="utf-8")


@tool_parameters(
    tool_parameters_schema(
        work_id=StringSchema(
            "Optional explicit work ID; defaults to the active work", nullable=True
        ),
        story_md=StringSchema("Story markdown or screenplay text"),
        story_profile=ObjectSchema(
            description=(
                "Structured story profile JSON; use for shot mapping and beat lookup. "
                "When provided, must include a non-empty summary and beats "
                "(array of {shot_id, summary})."
            ),
            additional_properties=True,
            nullable=True,
        ),
        confirmed=BooleanSchema(
            description=(
                "Whether the screenplay is locked for generation. "
                "Set to true only after the user explicitly confirms the screenplay in chat."
            ),
            default=False,
        ),
        summary=StringSchema("Optional short story summary to cache in state", nullable=True),
        required=["story_md"],
    )
)
class WriteStoryTool(DirectorTool):
    @property
    def name(self) -> str:
        return "write_story"

    async def execute(
        self,
        story_md: str,
        work_id: str | None = None,
        story_profile: dict[str, Any] | None = None,
        confirmed: bool = False,
        summary: str | None = None,
        **kwargs: Any,
    ) -> str:
        resolved_work_id, _ = self._resolve_work_id(work_id)
        if not resolved_work_id:
            return "Error: No active director work. Call start_director first."
        paths = self._paths(resolved_work_id)
        if isinstance(story_profile, dict):
            profile_error = _story_profile_validation_error(story_profile)
            if profile_error:
                return profile_error
            previous_profile = self._load_story_profile(resolved_work_id)
            prepared_profile = dict(story_profile)
            _preserve_story_profile_language(prepared_profile, previous_profile)
            if "language" not in prepared_profile:
                from nanobot.session.generation_settings import get_generation_settings
                from nanobot.session.manager import SessionManager

                session = SessionManager(self.workspace).get_or_create(self._session_key.get())
                metadata = session.metadata if isinstance(session.metadata, dict) else {}
                settings = get_generation_settings(metadata)
                _apply_story_profile_language(
                    prepared_profile,
                    str(settings.get("language") or ""),
                )
            language_error = _story_profile_language_validation_error(prepared_profile)
            if language_error:
                return language_error
            screenplay_error = _story_md_language_validation_error(story_md, prepared_profile)
            if screenplay_error:
                return screenplay_error
            self._save_story_profile(resolved_work_id, prepared_profile)
            story_profile = prepared_profile
        else:
            screenplay_error = _story_md_language_validation_error(
                story_md,
                self._load_story_profile(resolved_work_id),
            )
            if screenplay_error:
                return screenplay_error
        self._write_text(paths["story"], story_md)
        self._clear_reference_image_story_rewrite_flag()
        if confirmed:
            profile_error = _story_profile_validation_error(
                self._load_story_profile(resolved_work_id),
            )
            if profile_error:
                return (
                    "Error: Cannot confirm story without a valid story_profile. "
                    "Call write_story with story_profile including a non-empty summary "
                    "and at least one beat in beats."
                )
        state = self._load_state(resolved_work_id)
        if summary:
            state["latest_story_summary"] = summary
        elif isinstance(story_profile, dict):
            state["latest_story_summary"] = story_profile["summary"].strip()
        if confirmed:
            state["story_confirmed"] = True
            goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
            try:
                if int(goal.get("shot_count") or 0) > 0:
                    lock_reference_image(state)
                    self._lock_session_reference_image()
            except (TypeError, ValueError):
                pass
        # Agent has reconciled the user's story edit — clear the pending flag.
        state.pop("story_pending_agent_review", None)
        self._sync_stage_from_state(state)
        self._save_state(resolved_work_id, state)
        confirm = self._confirm(resolved_work_id)
        return _json_dump(
            {
                "status": "ok",
                "work_id": resolved_work_id,
                "story_confirmed": bool(state.get("story_confirmed")),
                "stage": state.get("stage"),
                "confirmation": confirm,
            }
        )


@tool_parameters(
    tool_parameters_schema(
        work_id=StringSchema(
            "Optional explicit work ID; defaults to the active work", nullable=True
        ),
    )
)
class GetFactTool(DirectorTool):
    @property
    def name(self) -> str:
        return "get_fact"

    async def execute(self, work_id: str | None = None, **kwargs: Any) -> str:
        resolved_work_id, _ = self._resolve_work_id(work_id)
        if not resolved_work_id:
            return "Error: No active director work. Call start_director first."
        return self._paths(resolved_work_id)["fact"].read_text(encoding="utf-8")


@tool_parameters(
    tool_parameters_schema(
        work_id=StringSchema(
            "Optional explicit work ID; defaults to the active work", nullable=True
        ),
    )
)
class ConfirmFactTool(DirectorTool):
    @property
    def name(self) -> str:
        return "confirm_fact"

    async def execute(self, work_id: str | None = None, **kwargs: Any) -> str:
        resolved_work_id, _ = self._resolve_work_id(work_id)
        if not resolved_work_id:
            return "Error: No active director work. Call start_director first."
        return _json_dump(self._confirm(resolved_work_id))


@tool_parameters(
    tool_parameters_schema(
        work_id=StringSchema(
            "Optional explicit work ID; defaults to the active work", nullable=True
        ),
        shot_id=IntegerSchema(description="1-based shot number", minimum=1),
        required=["shot_id"],
    )
)
class GetShotTool(DirectorTool):
    @property
    def name(self) -> str:
        return "get_shot"

    async def execute(self, shot_id: int, work_id: str | None = None, **kwargs: Any) -> str:
        resolved_work_id, _ = self._resolve_work_id(work_id)
        if not resolved_work_id:
            return "Error: No active director work. Call start_director first."
        shot = self._load_shot(resolved_work_id, shot_id)
        if not shot:
            return f"Error: Shot {shot_id} does not exist in work {resolved_work_id}."
        return _json_dump(shot)


def build_create_shot_prompt_parameters() -> dict[str, Any]:
    """Build create_shot_prompt's JSON schema at call time so PE switches take effect."""
    return tool_parameters_schema(
        work_id=StringSchema(
            "Optional explicit work ID; defaults to the active work", nullable=True
        ),
        shot_id=IntegerSchema(description="1-based shot number", minimum=1),
        cut=BooleanSchema(
            description="Whether this shot starts a fresh cut. false means generate as a continuation from the previous shot tail frame."
        ),
        caption=StringSchema(prompts.text("director.shot_caption.description")),
        status=StringSchema(
            "Optional explicit shot status",
            enum=[
                "planned",
                "prompt_ready",
                "revised_prompt_ready",
                "queued",
                "generated",
                "error",
                "review_pass",
                "review_fail",
                "approved",
            ],
            nullable=True,
        ),
        required=["shot_id", "cut", "caption"],
    )


class CreateShotPromptTool(DirectorTool):
    @property
    def name(self) -> str:
        return "create_shot_prompt"

    @property
    def parameters(self) -> dict[str, Any]:
        return build_create_shot_prompt_parameters()

    async def execute(
        self,
        shot_id: int,
        work_id: str | None = None,
        cut: bool = True,
        caption: str | None = None,
        status: str | None = None,
        **kwargs: Any,
    ) -> str:
        resolved_work_id, _ = self._resolve_work_id(work_id)
        if not resolved_work_id:
            return "Error: No active director work. Call start_director first."
        if not _allow_workflow_operation("create_shot_prompt"):
            return _workflow_gate_error("create_shot_prompt")
        if not isinstance(caption, str) or not caption.strip():
            return "Error: caption is required."
        story_profile = self._load_story_profile(resolved_work_id)
        language_error = _caption_language_validation_error(caption, story_profile)
        if language_error:
            return language_error
        shot = self._load_shot(resolved_work_id, shot_id)
        is_revised_prompt = self._is_revised_prompt_update(shot)
        shot["shot_id"] = shot_id
        shot["shot_key"] = _shot_key(shot_id)
        shot["cut"] = cut
        shot["caption"] = caption.strip()
        shot.pop("shot_spec", None)
        shot.pop("summary", None)
        shot["summary"] = self._summary_from_shot(shot)
        shot.pop("prompt", None)
        shot.pop("negative_prompt", None)
        for key in (
            "artifact_url",
            "artifact_path",
            "generation_error",
            "last_job_id",
            "echo",
            "remote_result",
            "last_review",
            "review_notes",
        ):
            shot.pop(key, None)
        if status is not None:
            shot["status"] = status
        else:
            shot["status"] = "revised_prompt_ready" if is_revised_prompt else "prompt_ready"
        state = self._load_state(resolved_work_id)
        sync_shot_echo_duration(shot, resolve_echo_duration_seconds(shot, state))
        self._save_shot(resolved_work_id, shot_id, shot)

        shots = state.setdefault("shots", {})
        if not isinstance(shots, dict):
            shots = {}
            state["shots"] = shots
        shots[shot["shot_key"]] = self._state_shot_entry(shot)
        self._sync_stage_from_state(state)
        self._save_state(resolved_work_id, state)
        confirm = self._confirm(resolved_work_id)
        return _json_dump(
            {
                "status": "ok",
                "work_id": resolved_work_id,
                "shot_id": shot_id,
                "shot_key": shot["shot_key"],
                "shot_status": shot["status"],
                "confirmation": confirm,
            }
        )


@tool_parameters(
    tool_parameters_schema(
        work_id=StringSchema(
            "Optional explicit work ID; defaults to the active work", nullable=True
        ),
        shot_id=IntegerSchema(description="1-based shot number", minimum=1),
        verdict=StringSchema(
            "Review result for the shot",
            enum=["accept", "revise"],
        ),
        review_source=StringSchema(
            "Who provided the review result",
            enum=["human", "vlm"],
        ),
        feedback=StringSchema(
            "Required when verdict='revise'; concise feedback for the next revision round",
            nullable=True,
        ),
        required=["shot_id", "verdict"],
    )
)
class ReviewShotTool(DirectorTool):
    @property
    def name(self) -> str:
        return "review_shot"

    async def execute(
        self,
        shot_id: int,
        verdict: str,
        work_id: str | None = None,
        review_source: str = "human",
        feedback: str | None = None,
        **kwargs: Any,
    ) -> str:
        resolved_work_id, _ = self._resolve_work_id(work_id)
        if not resolved_work_id:
            return "Error: No active director work. Call start_director first."
        if not _allow_workflow_operation("review_shot"):
            return _workflow_gate_error("review_shot")
        try:
            shot, state = self._apply_shot_review(
                resolved_work_id,
                shot_id,
                verdict=verdict,
                review_source=review_source,
                feedback=feedback,
            )
        except ValueError as exc:
            return f"Error: {exc}"
        return _json_dump(
            {
                "status": "ok",
                "work_id": resolved_work_id,
                "shot_id": shot_id,
                "shot_status": shot.get("status"),
                "stage": state.get("stage"),
                "review_notes": shot.get("review_notes") or "",
                "last_review": shot.get("last_review"),
            }
        )


@tool_parameters(
    tool_parameters_schema(
        work_id=StringSchema(
            "Optional explicit work ID; defaults to the active work", nullable=True
        ),
        shot_id=IntegerSchema(description="1-based shot number", minimum=1),
        reference_shot_ids=ArraySchema(
            IntegerSchema(
                description="Earlier logical shot ID used as visual reference", minimum=1
            ),
            description="Logical prior shot IDs selected as Echo references for this shot.",
        ),
        selection_note=StringSchema(
            "Optional short note explaining why these references were selected",
            nullable=True,
        ),
        required=["shot_id", "reference_shot_ids"],
    )
)
class SetShotReferencesTool(DirectorTool):
    @property
    def name(self) -> str:
        return "set_shot_references"

    def apply_set_references(
        self,
        work_id: str,
        shot_id: int,
        reference_shot_ids: list[int],
        selection_note: str | None = None,
    ) -> dict[str, Any]:
        shot = self._load_shot(work_id, shot_id)
        if not shot:
            raise ValueError(f"Shot {shot_id} does not exist in work {work_id}.")
        if not isinstance(shot.get("caption"), str) or not shot.get("caption", "").strip():
            raise ValueError(f"Shot {shot_id} has no caption yet. Call create_shot_prompt first.")
        normalized = self._normalize_reference_shot_ids(
            shot_id,
            reference_shot_ids,
            cut=bool(shot.get("cut", True)),
        )
        shot["planned_reference_shot_ids"] = normalized
        if isinstance(selection_note, str) and selection_note.strip():
            shot["reference_selection_note"] = selection_note.strip()
        self._save_shot(work_id, shot_id, shot)
        state = self._load_state(work_id)
        shots = state.setdefault("shots", {})
        if isinstance(shots, dict):
            shots[_shot_key(shot_id)] = self._state_shot_entry(shot)
        self._save_state(work_id, state)
        return shot

    async def execute(
        self,
        shot_id: int,
        reference_shot_ids: list[int],
        work_id: str | None = None,
        selection_note: str | None = None,
        **kwargs: Any,
    ) -> str:
        resolved_work_id, _ = self._resolve_work_id(work_id)
        if not resolved_work_id:
            return "Error: No active director work. Call start_director first."
        if not _allow_workflow_operation("set_shot_references"):
            return _workflow_gate_error("set_shot_references")
        try:
            shot = self.apply_set_references(
                resolved_work_id,
                shot_id,
                reference_shot_ids,
                selection_note=selection_note,
            )
        except ValueError as exc:
            return f"Error: {exc}"
        return _json_dump(
            {
                "status": "ok",
                "work_id": resolved_work_id,
                "shot_id": shot_id,
                "planned_reference_shot_ids": shot.get("planned_reference_shot_ids") or [],
                "reference_selection_note": shot.get("reference_selection_note") or "",
            }
        )


@tool_parameters(
    tool_parameters_schema(
        work_id=StringSchema(
            "Optional explicit work ID; defaults to the active work", nullable=True
        ),
        shot_id=IntegerSchema(description="Target 1-based shot number", minimum=1),
        recommendations=ArraySchema(
            ObjectSchema(
                image_asset_id=StringSchema(
                    "Profile-bearing Memory Workspace asset used for the slot image"
                ),
                audio_asset_id=StringSchema(
                    "Optional profile-bearing asset used for slot audio", nullable=True
                ),
                reason=StringSchema("Short reason this slot helps the target shot"),
                required=["image_asset_id", "reason"],
                additional_properties=False,
            ),
            description="Ordered recommendation draft; zero to seven slots.",
            max_items=7,
        ),
        required=["shot_id", "recommendations"],
    )
)
class SetShotMemoryRecommendationsTool(DirectorTool):
    """Let the agent propose slots without granting generation approval."""

    @property
    def name(self) -> str:
        return "set_shot_memory_recommendations"

    async def execute(
        self,
        shot_id: int,
        recommendations: list[dict[str, Any]],
        work_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        resolved_work_id, _ = self._resolve_work_id(work_id)
        if not resolved_work_id:
            return "Error: No active director work. Call start_director first."
        if not _allow_workflow_operation("set_shot_memory_recommendations"):
            return _workflow_gate_error("set_shot_memory_recommendations")
        if len(recommendations) > 7:
            return "Error: recommendations cannot exceed 7 slots."
        shot = self._load_shot(resolved_work_id, shot_id)
        if not shot:
            return f"Error: Shot {shot_id} does not exist in work {resolved_work_id}."
        catalog = {
            item["asset_id"]: item for item in self._memory_asset_catalog(resolved_work_id)
        }
        normalized: list[dict[str, Any]] = []
        seen_images: set[str] = set()
        for raw in recommendations:
            if not isinstance(raw, dict):
                return "Error: each recommendation must be an object."
            image_id = str(raw.get("image_asset_id") or "").strip()
            audio_id = str(raw.get("audio_asset_id") or "").strip() or None
            reason = str(raw.get("reason") or "").strip()
            if not image_id or image_id not in catalog:
                return f"Error: unknown or unprofiled image asset '{image_id}'."
            if catalog[image_id].get("media_type") == "audio":
                return f"Error: asset '{image_id}' has no image."
            if image_id in seen_images:
                return f"Error: duplicate image asset '{image_id}'."
            if audio_id is not None and audio_id not in catalog:
                return f"Error: unknown or unprofiled audio asset '{audio_id}'."
            if audio_id is not None and catalog[audio_id].get("media_type") == "image":
                return f"Error: asset '{audio_id}' has no audio."
            if not reason:
                return "Error: every recommendation needs a reason."
            seen_images.add(image_id)
            normalized.append({
                "image_asset_id": image_id,
                **({"audio_asset_id": audio_id} if audio_id else {}),
                "reason": reason[:500],
            })
        shot["recommended_memory_slot_refs"] = normalized
        shot["memory_recommendation_source"] = "agent"
        shot["memory_recommendation_updated_at"] = _now_iso()
        self._save_shot(resolved_work_id, shot_id, shot)
        return _json_dump({
            "status": "ok",
            "work_id": resolved_work_id,
            "shot_id": shot_id,
            "recommended_memory_slot_refs": normalized,
            "approval": "pending_human",
        })


@tool_parameters(
    tool_parameters_schema(
        work_id=StringSchema(
            "Optional explicit work ID; defaults to the active work", nullable=True
        ),
        shot_id=IntegerSchema(description="1-based shot number", minimum=1),
        reference_shot_ids=ArraySchema(
            IntegerSchema(
                description="Earlier logical shot ID used as visual reference", minimum=1
            ),
            description=(
                "Logical prior shot IDs the agent selected as Echo references. "
                "Almost every shot MUST reference at least one earlier shot for visual continuity — "
                "an empty list is allowed ONLY for shot_id=1, or when the shot introduces a completely new scene "
                "with entirely new characters that have zero visual overlap with any previous shot. "
                "When in doubt, include at least the most recent shot that shares a character, environment, or prop."
            ),
        ),
        selection_note=StringSchema(
            "Optional short note explaining why these references were selected",
            nullable=True,
        ),
        required=["shot_id", "reference_shot_ids"],
    )
)
class GenerateEchoShotTool(DirectorTool):
    @property
    def name(self) -> str:
        return "generate_echo_shot"

    def apply_generate(
        self,
        work_id: str,
        shot_id: int,
        reference_shot_ids: list[int],
        selection_note: str | None = None,
        condition_image_url: str | None = None,
        i2v_prompt: str | None = None,
    ) -> dict[str, Any]:
        state = self._load_state(work_id)
        existing_shot = self._load_shot(work_id, shot_id)
        if (
            shot_id > 1
            and str(state.get("stage") or "") == "awaiting_memory_build"
            and not existing_shot.get("memory_slots_user_configured")
            and not state.get("auto_generate")
        ):
            raise ValueError(
                "Build Memory must be reviewed and applied before generating this shot."
            )
        caption = existing_shot.get("caption")
        # Skip language validation when an I2V prompt is supplied — the prompt
        # has already been rewritten by rewrite_prompt_for_i2v with the correct
        # first-frame contract sentence, and the underlying caption may contain
        # technical tokens from PE re-captioning that the validator flags.
        if isinstance(caption, str) and not i2v_prompt:
            language_error = _caption_language_validation_error(
                caption,
                self._load_story_profile(work_id),
            )
            if language_error:
                raise ValueError(language_error.removeprefix("Error: "))

        duration_value = resolve_echo_duration_seconds(existing_shot, state)
        num_frames = sync_shot_echo_duration(existing_shot, duration_value)
        goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
        width = goal.get("width")
        height = goal.get("height")
        prompt_text = i2v_prompt.strip() if i2v_prompt else caption.strip()

        # Normalize references early so invalid refs fail before submission.
        normalized_reference_ids = self._normalize_reference_shot_ids(
            shot_id, reference_shot_ids, cut=bool(existing_shot.get("cut", True))
        )

        memory_slots = self._build_memory_slots(
            existing_shot.get("approved_memory_slots"), normalized_reference_ids,
            work_id=work_id,
        )

        request_payload = self._build_r2v_payload(
            work_id,
            shot_id,
            prompt=prompt_text,
            num_frames=num_frames,
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
            condition_image_url=condition_image_url,
            memory_slots=memory_slots,
        )

        state["stage"] = "shot_generating"
        job_id = _job_id("echo", work_id, _shot_key(shot_id))
        try:
            job = self._submit_r2v_request(
                work_id,
                job_id,
                request_payload,
                target=_shot_key(shot_id),
            )
        except EchoGeneratorBusyError:
            raise
        except EchoGeneratorUnavailableError:
            raise
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

        self._write_json(self._job_path(work_id, job_id), job)
        self._clear_pending_remote_jobs_for_target(state, "generate_echo_shot", _shot_key(shot_id))
        self._register_pending_remote_job(state, job)

        existing_shot["status"] = "queued"
        existing_shot.pop("generation_error", None)
        existing_shot["last_job_id"] = job_id
        existing_shot["reference_shot_ids"] = normalized_reference_ids
        if selection_note is not None:
            existing_shot["reference_selection_note"] = selection_note
        echo = existing_shot.get("echo")
        if not isinstance(echo, dict):
            echo = {}
            existing_shot["echo"] = echo
        echo.update(
            {
                "status": "queued",
                "reference_shot_ids": normalized_reference_ids,
                "selection_note": selection_note,
                "request_payload_path": job.get("request_payload_path"),
                "request_envelope_path": job.get("request_envelope_path"),
                "remote_task_id": (
                    job.get("remote", {}).get("remote_task_id")
                    if isinstance(job.get("remote"), dict)
                    else None
                ),
                "version_id": (
                    job.get("remote", {}).get("version_id")
                    if isinstance(job.get("remote"), dict)
                    else None
                ),
            }
        )
        self._save_shot(work_id, shot_id, existing_shot)

        shots = state.setdefault("shots", {})
        if isinstance(shots, dict):
            shots[_shot_key(shot_id)] = self._state_shot_entry(existing_shot)
        self._save_state(work_id, state)
        self._refresh_fact(work_id, state)
        return job

    def apply_generate_continuous(
        self,
        work_id: str,
        shot_id: int,
        condition_image_url: str,
        reference_shot_ids: list[int],
        selection_note: str | None = None,
        i2v_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Submit a shot for I2V generation using the previous shot's tail frame as condition image.

        Rewrites the current shot's prompt for I2V format, then submits to the Echo
        backend with ``condition_image_url`` pointing to the tail-frame image.
        """
        state = self._load_state(work_id)
        existing_shot = self._load_shot(work_id, shot_id)
        if (
            shot_id > 1
            and str(state.get("stage") or "") == "awaiting_memory_build"
            and not existing_shot.get("memory_slots_user_configured")
            and not state.get("auto_generate")
        ):
            raise ValueError(
                "Build Memory must be reviewed and applied before generating this shot."
            )
        caption = existing_shot.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Shot {shot_id} has no caption yet. Call create_shot_prompt first.")

        # Defensive caption-language check (mirrors apply_generate).
        language_error = _caption_language_validation_error(
            caption,
            self._load_story_profile(work_id),
        )
        if language_error:
            raise ValueError(language_error.removeprefix("Error: "))

        story_profile = self._load_story_profile(work_id)
        caption_language = str(
            story_profile.get("caption_language")
            or story_profile.get("language")
            or ""
        )

        # The WebSocket continuation path performs the multimodal rewrite
        # (ordinary PE + I2V skill + the extracted tail frame). Keep the
        # deterministic helper as a fallback for existing internal callers.
        rewritten_prompt = (
            i2v_prompt.strip()
            if isinstance(i2v_prompt, str) and i2v_prompt.strip()
            else rewrite_prompt_for_i2v(caption.strip(), caption_language)
        )

        # Persist the I2V prompt on the shot record that will be saved.
        existing_shot["i2v_prompt"] = rewritten_prompt

        duration_value = resolve_echo_duration_seconds(existing_shot, state)
        num_frames = sync_shot_echo_duration(existing_shot, duration_value)
        goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
        width = goal.get("width")
        height = goal.get("height")

        # Normalize references early so invalid refs fail before submission.
        normalized_reference_ids = self._normalize_reference_shot_ids(
            shot_id, reference_shot_ids, cut=bool(existing_shot.get("cut", True))
        )

        memory_slots = self._build_memory_slots(
            existing_shot.get("approved_memory_slots"), normalized_reference_ids,
            work_id=work_id,
        )

        request_payload = self._build_r2v_payload(
            work_id,
            shot_id,
            prompt=rewritten_prompt,
            num_frames=num_frames,
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
            condition_image_url=condition_image_url,
            memory_slots=memory_slots,
        )

        state["stage"] = "shot_generating"
        job_id = _job_id("echo", work_id, _shot_key(shot_id))
        try:
            job = self._submit_r2v_request(
                work_id,
                job_id,
                request_payload,
                target=_shot_key(shot_id),
            )
        except EchoGeneratorBusyError:
            raise
        except EchoGeneratorUnavailableError:
            raise
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

        self._write_json(self._job_path(work_id, job_id), job)
        self._clear_pending_remote_jobs_for_target(state, "generate_echo_shot", _shot_key(shot_id))
        self._register_pending_remote_job(state, job)

        existing_shot["status"] = "queued"
        existing_shot.pop("generation_error", None)
        existing_shot["last_job_id"] = job_id
        existing_shot["reference_shot_ids"] = normalized_reference_ids
        if selection_note is not None:
            existing_shot["reference_selection_note"] = selection_note
        echo = existing_shot.get("echo")
        if not isinstance(echo, dict):
            echo = {}
            existing_shot["echo"] = echo
        echo.update(
            {
                "status": "queued",
                "reference_shot_ids": normalized_reference_ids,
                "selection_note": selection_note,
                "request_payload_path": job.get("request_payload_path"),
                "request_envelope_path": job.get("request_envelope_path"),
                "remote_task_id": (
                    job.get("remote", {}).get("remote_task_id")
                    if isinstance(job.get("remote"), dict)
                    else None
                ),
                "version_id": (
                    job.get("remote", {}).get("version_id")
                    if isinstance(job.get("remote"), dict)
                    else None
                ),
            }
        )
        self._save_shot(work_id, shot_id, existing_shot)

        shots = state.setdefault("shots", {})
        if isinstance(shots, dict):
            shots[_shot_key(shot_id)] = self._state_shot_entry(existing_shot)
        self._save_state(work_id, state)
        self._refresh_fact(work_id, state)
        return job

    async def execute(
        self,
        shot_id: int,
        reference_shot_ids: list[int],
        work_id: str | None = None,
        selection_note: str | None = None,
        **kwargs: Any,
    ) -> str:
        resolved_work_id, _ = self._resolve_work_id(work_id)
        if not resolved_work_id:
            return "Error: No active director work. Call start_director first."
        if not _allow_workflow_operation("generate_echo_shot"):
            return _workflow_gate_error("generate_echo_shot")

        # 检查是否开启首尾衔接 → I2V 生成
        shot = self._load_shot(resolved_work_id, shot_id)
        use_continuous = (
            bool(shot.get("continuous_enabled"))
            and shot_id > 1
            and not bool(self._load_state(resolved_work_id).get("auto_generate"))
        )

        try:
            if use_continuous:
                previous_shot_id = shot_id - 1
                prev_shot = self._load_shot(resolved_work_id, previous_shot_id)
                video_url = prev_shot.get("artifact_url") or (
                    prev_shot.get("echo") or {}
                ).get("result_url")
                if not video_url:
                    return (
                        f"Error: previous shot {previous_shot_id} has no video "
                        f"artifact; cannot extract tail frame for continuous generation"
                    )
                logger.info(
                    "agent continuous-generate: shot_id={} using previous shot {} "
                    "tail frame, extracting from video_url={}",
                    shot_id, previous_shot_id, video_url,
                )
                condition_image_url = await asyncio.to_thread(
                    self._extract_and_publish_tail_frame,
                    resolved_work_id,
                    previous_shot_id,
                    video_url,
                )
                if not condition_image_url:
                    return (
                        f"Error: failed to extract tail frame from shot "
                        f"{previous_shot_id}"
                    )
                logger.info(
                    "agent continuous-generate: tail frame ready, "
                    "shot_id={} condition_image_url={}",
                    shot_id, condition_image_url,
                )
                # 确保 previous_shot_id 在 reference_shot_ids 中
                if previous_shot_id not in reference_shot_ids:
                    reference_shot_ids = sorted(
                        set(reference_shot_ids) | {previous_shot_id}
                    )
                job = await asyncio.to_thread(
                    self.apply_generate_continuous,
                    resolved_work_id,
                    shot_id,
                    condition_image_url,
                    reference_shot_ids,
                    selection_note=selection_note,
                )
            else:
                first_frame_url = None
                if shot_id == 1:
                    first_frame_url = self._state_first_frame_url(
                        self._load_state(resolved_work_id)
                    )
                if first_frame_url:
                    logger.info(
                        "agent generate_echo_shot: shot_id=1 using state.reference_image "
                        "url={}",
                        first_frame_url,
                    )
                    profile = self._load_story_profile(resolved_work_id)
                    language = ""
                    if isinstance(profile, dict):
                        language = str(
                            profile.get("caption_language") or profile.get("language") or ""
                        )
                    caption = str(shot.get("caption") or "").strip()
                    i2v_prompt = (
                        rewrite_prompt_for_i2v(caption, language) if caption else None
                    )
                    job = await asyncio.to_thread(
                        self.apply_generate,
                        resolved_work_id,
                        shot_id,
                        reference_shot_ids,
                        selection_note=selection_note,
                        condition_image_url=first_frame_url,
                        i2v_prompt=i2v_prompt,
                    )
                else:
                    job = await asyncio.to_thread(
                        self.apply_generate,
                        resolved_work_id,
                        shot_id,
                        reference_shot_ids,
                        selection_note=selection_note,
                    )
        except EchoGeneratorBusyError as exc:
            return f"Error: {exc}"
        except EchoGeneratorUnavailableError as exc:
            return f"Error: {exc}"
        except ValueError as exc:
            return f"Error: {exc}"
        return _json_dump(job)


@tool_parameters(
    tool_parameters_schema(
        work_id=StringSchema(
            "Optional explicit work ID; defaults to the active work", nullable=True
        ),
        shot_ids=ArraySchema(
            IntegerSchema(description="1-based shot number"),
            description="Optional explicit shot list; defaults to all known shots in order",
            nullable=True,
        ),
    )
)
class MergeShotTool(DirectorTool):
    @property
    def name(self) -> str:
        return "merge_shot"

    async def execute(
        self,
        work_id: str | None = None,
        shot_ids: list[int] | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            job = await asyncio.to_thread(
                self.apply_merge,
                work_id=work_id,
                shot_ids=shot_ids,
            )
        except (RuntimeError, ValueError) as exc:
            return f"Error: {exc}"
        return _json_dump(job)

    def apply_merge(
        self,
        work_id: str | None = None,
        shot_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        resolved_work_id, _ = self._resolve_work_id(work_id)
        if not resolved_work_id:
            raise ValueError("No active director work. Call start_director first.")
        if not _allow_workflow_operation("merge_shot"):
            raise ValueError(_workflow_gate_error("merge_shot"))
        state = self._load_state(resolved_work_id)
        state["stage"] = "merging"
        state.pop("generation_error", None)
        available = self._shot_entries(state)
        if not available:
            raise ValueError("No shots exist yet. Create shot prompts first.")
        selected_ids = shot_ids or [int(item["shot_id"]) for item in available]
        selected_shots = []
        for shot_id in selected_ids:
            shot = self._load_shot(resolved_work_id, shot_id)
            if not shot:
                raise ValueError(f"Shot {shot_id} does not exist in work {resolved_work_id}.")
            selected_shots.append(shot)
        job_id = _job_id("merge", resolved_work_id, "final")
        payload = self._build_merge_payload(
            resolved_work_id,
            selected_ids,
            selected_shots,
        )
        job = self._submit_remote_request(
            resolved_work_id,
            job_id,
            payload,
            target="final",
            operation="merge_shot",
        )
        self._write_json(self._job_path(resolved_work_id, job_id), job)
        self._clear_pending_remote_jobs_for_target(state, "merge_shot", "final")
        self._register_pending_remote_job(state, job)
        state.pop("merge_confirmation_requested_at", None)
        state["latest_merge_job_id"] = job_id
        self._sync_stage_from_state(state)
        self._save_state(resolved_work_id, state)
        self._refresh_fact(resolved_work_id, state)
        return job


@tool_parameters(
    tool_parameters_schema(
        job_id=StringSchema("Director job ID to inspect"),
        work_id=StringSchema(
            "Optional explicit work ID; defaults to the active work or a repo-wide search",
            nullable=True,
        ),
        required=["job_id"],
    )
)
class GetDirectorJobTool(DirectorTool):
    @property
    def name(self) -> str:
        return "get_director_job"

    async def execute(self, job_id: str, work_id: str | None = None, **kwargs: Any) -> str:
        candidates: list[Path] = []
        resolved_work_id, _ = self._resolve_work_id(work_id)
        if resolved_work_id:
            candidates.append(self._job_path(resolved_work_id, job_id))
        else:
            self._ensure_root()
            for jobs_dir in self.works_root.glob("*/jobs"):
                candidates.append(jobs_dir / f"{job_id}.json")
        for path in candidates:
            if path.exists():
                data = self._read_json(path, {})
                if isinstance(data, dict):
                    return _json_dump(data)
        return f"Error: Director job '{job_id}' was not found."


def apply_echo_generate_shot_callback(
    workspace: Path,
    callback_payload: dict[str, Any],
    *,
    tools_config: Any | None = None,
) -> dict[str, Any]:
    """Apply one generate_echo_shot remote callback to the director workspace.

    Expected callback payload shape:
    - required: `work_id`, `job_id`
    - optional: `status` (`completed` or `failed`, defaults to `completed`)
    - completed result:
      - either top-level `shot_id` + a public `asset_urls` entry (preferred) or `result_url`
      - or `result={"shot_id": 8, "asset_urls": {"primary": {"url": "https://..."}}}` / `result_url`
    - injection routing:
      - `session_key`, `channel`, `chat_id`
    """
    tool = GenerateEchoShotTool(workspace=workspace, tools_config=tools_config)
    return tool.apply_echo_callback_payload(callback_payload)


def apply_merge_shot_callback(
    workspace: Path,
    callback_payload: dict[str, Any],
    *,
    tools_config: Any | None = None,
) -> dict[str, Any]:
    """Apply one merge_shot remote callback to the director workspace.

    Expected callback payload shape:
    - required: `work_id`, `job_id`
    - optional: `status` (`completed` or `failed`, defaults to `completed`)
    - completed result:
      - preferred: the first public URL in top-level or `result.asset_urls`
      - fallback: `artifact_path`, `artifact_url`, `result_url`, or `result={...}` equivalents
    - injection routing:
      - `session_key`, `channel`, `chat_id`
    """
    tool = MergeShotTool(workspace=workspace, tools_config=tools_config)
    return tool.apply_merge_callback_payload(callback_payload)
