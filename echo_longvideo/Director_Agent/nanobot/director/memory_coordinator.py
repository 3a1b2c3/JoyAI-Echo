"""Persist and project shot memories for the human review gate."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from nanobot.director.memory_review import MemoryReviewConflict
from nanobot.director.memory_selector import MemoryVlmSelector
from nanobot.director.r2v_memory_workflow import (
    Publisher,
    publish_memory_record,
    publish_memory_records,
)
from nanobot.storage.files import configured_file_publisher
from nanobot.utils.helpers import write_json_atomic

AudioExtractor = Callable[[Path, Path], None]
_MEMORY_REVIEW_LOCK = threading.Lock()
_MAX_VLM_PROMPT_IMAGES = 16
_SHOT_READ_ATTEMPTS = 20
_SHOT_READ_DELAY_SEC = 0.01
_COMMON_MEDIA_BIN_DIRS = (
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
    Path("/usr/bin"),
)


def _resolve_media_binary(name: str) -> str:
    """Resolve ffmpeg tools even when a detached Gateway has a minimal PATH."""
    configured = os.environ.get(f"NANOBOT_{name.upper()}_BIN", "").strip()
    if configured:
        return configured
    discovered = shutil.which(name)
    if discovered:
        return discovered
    for directory in _COMMON_MEDIA_BIN_DIRS:
        candidate = directory / name
        if candidate.is_file():
            return str(candidate)
    return name


def merge_approved_memories(
    bank: dict[str, dict[str, Any]],
    proposed: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge an approved proposal without degrading confirmed visual memories."""
    merged = deepcopy(bank)
    for memory_id, candidate in proposed.items():
        existing = merged.get(memory_id)
        if existing and existing.get("visual_status") == "confirmed":
            if candidate.get("visual_status") != "confirmed":
                continue
            if float(candidate.get("confidence") or 0) <= float(
                existing.get("confidence") or 0
            ):
                continue
        replacement = deepcopy(candidate)
        if existing:
            for key in ("local_audio_path", "audio_path", "audio_source_shot_id"):
                if not replacement.get(key) and existing.get(key):
                    replacement[key] = existing[key]
        merged[memory_id] = replacement
    return merged


def materialize_character_memories(
    *,
    selections: dict[str, Any],
    candidates: list[dict[str, Any]],
    shot_id: int,
    video_path: Path,
    memory_dir: Path,
    audio_extractor: AudioExtractor,
) -> dict[str, dict[str, Any]]:
    """Copy selected frames and bind each new visual to this shot's audio."""
    by_index = {int(item["candidate_index"]): item for item in candidates}
    rows = selections.get("selections")
    if not isinstance(rows, list):
        return {}
    memory_dir.mkdir(parents=True, exist_ok=True)
    source_audio = memory_dir / f"_source_shot_{shot_id:03d}.wav"
    if rows:
        audio_extractor(video_path, source_audio)
    proposed: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        memory_id = str(raw.get("character_id") or "").strip()
        try:
            candidate = by_index[int(raw["candidate_index"])]
            confidence = float(raw["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        if not memory_id:
            continue
        image = memory_dir / f"{memory_id}.jpg"
        audio = memory_dir / f"{memory_id}.wav"
        shutil.copyfile(str(candidate["path"]), image)
        shutil.copyfile(source_audio, audio)
        proposed[memory_id] = {
            "memory_id": memory_id,
            "kind": "character",
            "candidate_index": int(candidate["candidate_index"]),
            "frame_index": int(candidate["frame_index"]),
            "timestamp_sec": float(candidate["timestamp_sec"]),
            "confidence": confidence,
            "target_only": bool(raw.get("target_only", False)),
            "visible_character_ids": list(raw.get("visible_character_ids") or []),
            "reasoning": str(raw.get("reasoning") or ""),
            "visual_status": "confirmed" if raw.get("target_only") else "provisional",
            "source_shot_id": int(shot_id),
            "audio_source_shot_id": int(shot_id),
            "local_image_path": str(image.resolve()),
            "local_audio_path": str(audio.resolve()),
        }
    source_audio.unlink(missing_ok=True)
    return proposed


def _review_selection(memory_id: str, record: dict[str, Any], *, kind: str) -> dict[str, Any]:
    selection = {
        "memory_id": memory_id,
        "kind": kind,
        "candidate_index": int(record.get("candidate_index") or 0),
        "frame_index": int(record.get("frame_index") or 0),
        "timestamp_sec": float(record.get("timestamp_sec") or 0),
        "confidence": float(record.get("confidence") or 0),
        "visual_status": str(record.get("visual_status") or (
            "representative" if kind == "previous_shot" else "provisional")),
        "source_shot_id": int(record.get("source_shot_id") or 0),
        "audio_source_shot_id": int(record.get("audio_source_shot_id")
                                    or record.get("source_shot_id") or 0),
        "reasoning": str(record.get("reasoning") or ""),
    }
    for key in ("local_image_path", "local_audio_path", "image_path", "audio_path"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            selection[key] = value.strip()
    return selection


def build_memory_review(
    *,
    shot_id: int,
    attempt: int,
    candidate_count: int,
    bank: dict[str, dict[str, Any]],
    previous_shot: dict[str, Any] | None,
    review_id: str,
    rejected_candidate_indices: list[int] | None = None,
    history: list[dict[str, Any]] | None = None,
    updated_at: str = "",
) -> dict[str, Any]:
    """Build a proposal containing every stored ID plus PREVIOUS_SHOT."""
    selections = [
        _review_selection(memory_id, record, kind="character")
        for memory_id, record in bank.items()
        if isinstance(record, dict)
        and (record.get("local_image_path") or record.get("image_path"))
    ]
    if previous_shot and (previous_shot.get("local_image_path")
                          or previous_shot.get("image_path")):
        selections.append(_review_selection(
            "PREVIOUS_SHOT", previous_shot, kind="previous_shot"))
    return {
        "review_id": review_id,
        "status": "awaiting_review",
        "shot_id": int(shot_id),
        "attempt": int(attempt),
        "candidate_count": int(candidate_count),
        "rejected_candidate_indices": sorted(set(rejected_candidate_indices or [])),
        "selections": selections,
        "history": deepcopy(history or []),
        "error": None,
        "updated_at": updated_at,
    }


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return deepcopy(default)


def _write_json(path: Path, value: Any) -> None:
    write_json_atomic(path, value)


def _load_shot_for_memory(shot_path: Path, shot_id: int) -> dict[str, Any]:
    """Read a shot JSON, retrying through concurrent truncated writes."""
    for attempt in range(_SHOT_READ_ATTEMPTS):
        shot: dict[str, Any] | None = None
        try:
            if shot_path.is_file():
                raw = shot_path.read_text(encoding="utf-8")
                if raw.strip():
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        shot = parsed
        except (OSError, json.JSONDecodeError):
            shot = None
        if isinstance(shot, dict):
            try:
                loaded_id = int(shot.get("shot_id") or 0)
            except (TypeError, ValueError):
                loaded_id = 0
            if loaded_id == shot_id:
                return shot
        if attempt + 1 < _SHOT_READ_ATTEMPTS:
            time.sleep(_SHOT_READ_DELAY_SEC)
    raise ValueError(f"shot {shot_id} not found for memory selection")


def mark_memory_review_selecting(
    *,
    workspace: Path,
    work_id: str,
    shot_id: int,
) -> dict[str, Any]:
    """Persist selecting before workplace push so auto-generate does not race."""
    shot_path = (
        workspace / "director" / "works" / work_id / "shots" / f"shot_{shot_id:03d}.json"
    )
    shot = _load_shot_for_memory(shot_path, shot_id)
    review = shot.get("memory_review")
    review = review if isinstance(review, dict) else {}
    status = str(review.get("status") or "")
    if status not in {"awaiting_review", "approved", "selecting", "reselecting"}:
        review["status"] = "selecting"
        review["shot_id"] = int(shot_id)
        shot["memory_review"] = review
        _write_json(shot_path, shot)
    return shot


def _ordered_character_ids(caption: str) -> list[str]:
    return list(dict.fromkeys(re.findall(
        r"(?<![A-Za-z0-9_-])ID_[A-Za-z0-9_-]+(?![A-Za-z0-9_-])",
        caption,
    )))


def initialize_memory_review_method_prompt(
    *,
    workspace: Path,
    work_id: str,
    shot_id: int,
    error: str | None = None,
) -> dict[str, Any]:
    """Pause after generation and ask how memories should be selected."""
    work_dir = workspace / "director" / "works" / work_id
    shot_path = work_dir / "shots" / f"shot_{shot_id:03d}.json"
    state_path = work_dir / "state.json"
    bank_path = work_dir / "memory" / "memory_bank.json"
    shot = _load_shot_for_memory(shot_path, shot_id)
    caption = str(shot.get("caption") or shot.get("summary") or "")
    character_ids = _ordered_character_ids(caption)
    bank = _read_json(bank_path, {})
    bank = bank if isinstance(bank, dict) else {}
    missing_character_ids = [
        memory_id
        for memory_id in character_ids
        if not isinstance(bank.get(memory_id), dict)
        or not (
            bank[memory_id].get("local_image_path")
            or bank[memory_id].get("image_path")
        )
    ]
    review_character_ids = missing_character_ids or character_ids
    review = {
        "review_id": f"memory-review-{uuid.uuid4().hex}",
        "status": "awaiting_method",
        "shot_id": int(shot_id),
        "attempt": 1,
        "candidate_count": 0,
        "rejected_candidate_indices": [],
        "required_memory_ids": review_character_ids + ["PREVIOUS_SHOT"],
        "manual_selected_ids": [],
        "retained_memory_ids": [],
        "selection_mode": None,
        "selections": [],
        "history": [],
        "error": error.strip() if isinstance(error, str) and error.strip() else None,
        "updated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z"),
    }
    shot["memory_review"] = review
    _write_json(shot_path, shot)
    state = _read_json(state_path, {})
    if isinstance(state, dict):
        state["stage"] = "awaiting_memory_review"
        _write_json(state_path, state)
    return review


class ManualMemorySelector:
    """Create non-final placeholders after the user chooses manual mode."""

    @staticmethod
    def _middle(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidates:
            raise ValueError("manual memory selection has no source frames")
        return candidates[len(candidates) // 2]

    def select_characters(
        self,
        *,
        character_ids: list[str],
        candidates: list[dict[str, Any]],
        **_: Any,
    ) -> dict[str, Any]:
        middle = self._middle(candidates)
        return {
            "reasoning": "Waiting for manual identity-frame selection.",
            "selections": [
                {
                    "character_id": memory_id,
                    "candidate_index": int(middle["candidate_index"]),
                    "confidence": 0.0,
                    "target_only": False,
                    "visible_character_ids": list(character_ids),
                    "reasoning": "Drag through the video and choose this identity memory.",
                }
                for memory_id in character_ids
            ],
        }

    def decide_scene_transition(self, **_: Any) -> dict[str, Any]:
        return {
            "scene_transition": False,
            "reasoning": "Scene continuity will be selected manually.",
        }

    def select_representative(
        self,
        *,
        candidates: list[dict[str, Any]],
        **_: Any,
    ) -> dict[str, Any]:
        middle = self._middle(candidates)
        return {
            "candidate_index": int(middle["candidate_index"]),
            "confidence": 0.0,
            "reasoning": "Drag through the video and choose the scene memory.",
        }


def _next_planned_shot(work_dir: Path, shot_id: int) -> dict[str, Any] | None:
    """Return the first later shot that has enough screenplay text to compare."""
    for path in sorted((work_dir / "shots").glob("shot_*.json")):
        candidate = _read_json(path, {})
        if not isinstance(candidate, dict):
            continue
        try:
            candidate_id = int(candidate.get("shot_id") or 0)
        except (TypeError, ValueError):
            continue
        caption = str(candidate.get("caption") or candidate.get("summary") or "").strip()
        if candidate_id > shot_id and caption:
            return candidate
    return None


def prepare_memory_review(
    *,
    workspace: Path,
    work_id: str,
    shot_id: int,
    selector: Any,
    video_fetcher: Callable[[str, Path], None],
    frame_sampler: Callable[[Path, Path, int], list[dict[str, Any]]],
    audio_extractor: AudioExtractor,
    review_id_factory: Callable[[], str],
    now: Callable[[], str],
    candidate_count: int = 6,
    target_memory_id: str | None = None,
    publisher: Publisher | None = None,
) -> dict[str, Any]:
    """Create and persist a real review proposal for a completed R2V shot."""
    work_dir = workspace / "director" / "works" / work_id
    shot_path = work_dir / "shots" / f"shot_{shot_id:03d}.json"
    state_path = work_dir / "state.json"
    bank_path = work_dir / "memory" / "memory_bank.json"
    shot = _load_shot_for_memory(shot_path, shot_id)
    state = _read_json(state_path, {})
    bank = _read_json(bank_path, {})
    if str(shot.get("status") or "") not in {"generated", "review_pass", "approved"}:
        raise ValueError(f"shot {shot_id} is not generated")
    if not isinstance(bank, dict):
        bank = {}

    old_review = shot.get("memory_review")
    old_review = old_review if isinstance(old_review, dict) else {}
    if target_memory_id:
        review_ids = {
            str(item.get("memory_id") or "")
            for item in old_review.get("selections", [])
            if isinstance(item, dict)
        }
        if target_memory_id not in review_ids:
            raise ValueError(
                f"memory {target_memory_id} is not in the current review"
            )
    attempt = int(old_review.get("attempt") or 0) + 1
    rejected = {
        int(value) for value in old_review.get("rejected_candidate_indices", [])
        if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
    }
    history = old_review.get("history")
    history = history if isinstance(history, list) else []

    memory_root = work_dir / "memory"
    video_path = memory_root / "videos" / f"shot_{shot_id:03d}.mp4"
    artifact = str(shot.get("artifact_url") or shot.get("artifact_path") or "").strip()
    if not artifact:
        raise ValueError(f"shot {shot_id} has no video artifact")
    if not video_path.is_file():
        video_fetcher(artifact, video_path)
    candidate_dir = memory_root / "candidates" / f"shot_{shot_id:03d}"
    candidates = frame_sampler(video_path, candidate_dir, candidate_count)
    if not candidates:
        raise ValueError(f"shot {shot_id} produced no memory candidates")

    caption = str(shot.get("caption") or shot.get("summary") or "")
    character_ids = _ordered_character_ids(caption)
    proposal_base = (
        deepcopy(old_review.get("proposed_bank"))
        if target_memory_id and isinstance(old_review.get("proposed_bank"), dict)
        else deepcopy(bank)
    )
    missing_character_ids = [
        memory_id
        for memory_id in character_ids
        if not isinstance(bank.get(memory_id), dict)
        or not (
            bank[memory_id].get("local_image_path")
            or bank[memory_id].get("image_path")
        )
    ]
    if target_memory_id == "PREVIOUS_SHOT":
        selection_character_ids = []
    elif target_memory_id:
        selection_character_ids = [target_memory_id]
    else:
        selection_character_ids = missing_character_ids or character_ids
    if selection_character_ids:
        try:
            character_selection = selector.select_characters(
                shot_id=shot_id,
                caption=caption,
                character_ids=selection_character_ids,
                candidates=candidates,
                rejected_candidate_indices=rejected,
            )
        except Exception as exc:
            raise RuntimeError(
                f"memory character selection failed for shot {shot_id}: {exc}"
            ) from exc
    else:
        character_selection = {"reasoning": "no character IDs", "selections": []}
    selection_rows = character_selection.get("selections")
    if not isinstance(selection_rows, list):
        selection_rows = []
        character_selection["selections"] = selection_rows
    selected_ids = {
        str(item.get("character_id") or "")
        for item in selection_rows
        if isinstance(item, dict)
    }
    unselected_ids = [
        memory_id
        for memory_id in selection_character_ids
        if memory_id not in selected_ids
    ]
    if unselected_ids:
        raise RuntimeError(
            "memory character selection returned no frame for required IDs: "
            + ", ".join(unselected_ids)
        )
    proposal_dir = memory_root / "proposals" / f"shot_{shot_id:03d}_attempt_{attempt:02d}"
    proposed = materialize_character_memories(
        selections=character_selection,
        candidates=candidates,
        shot_id=shot_id,
        video_path=video_path,
        memory_dir=proposal_dir,
        audio_extractor=audio_extractor,
    )
    proposed_bank = merge_approved_memories(proposal_base, proposed)
    changed_memory_ids = {
        memory_id: record
        for memory_id, record in proposed_bank.items()
        if memory_id not in bank or record != bank[memory_id]
    }

    next_shot = _next_planned_shot(work_dir, shot_id)
    transition = False
    transition_reasoning = "next shot unavailable; preserve continuity by default"
    if next_shot is not None:
        next_shot_id = int(next_shot["shot_id"])
        next_caption = str(next_shot.get("caption") or next_shot.get("summary") or "")
        try:
            decision = selector.decide_scene_transition(
                previous_shot_id=shot_id,
                previous_caption=caption,
                next_shot_id=next_shot_id,
                next_caption=next_caption,
            )
            transition = bool(decision["scene_transition"])
            transition_reasoning = str(decision["reasoning"])
        except Exception as exc:
            raise RuntimeError(
                f"memory scene-transition selection failed for shot {shot_id}: {exc}"
            ) from exc

    # The active mode always carries one representative continuity frame.
    # Scene-transition analysis remains metadata and does not remove this slot.
    previous = (
        deepcopy(old_review.get("previous_shot"))
        if target_memory_id and target_memory_id != "PREVIOUS_SHOT"
        and isinstance(old_review.get("previous_shot"), dict)
        else None
    )
    if candidates and (
        target_memory_id is None or target_memory_id == "PREVIOUS_SHOT"
    ):
        try:
            representative_selection = selector.select_representative(
                shot_id=shot_id,
                caption=caption,
                candidates=candidates,
                rejected_candidate_indices=rejected,
            )
        except Exception as exc:
            raise RuntimeError(
                f"memory representative selection failed for shot {shot_id}: {exc}"
            ) from exc
        by_index = {int(item["candidate_index"]): item for item in candidates}
        representative_index = int(representative_selection["candidate_index"])
        if representative_index not in by_index:
            raise RuntimeError(
                "memory representative selection returned invalid candidate index "
                f"{representative_index} for shot {shot_id}"
            )
        representative_candidate = by_index[representative_index]
        representative_image = proposal_dir / "PREVIOUS_SHOT.jpg"
        representative_audio = proposal_dir / "PREVIOUS_SHOT.wav"
        shutil.copyfile(str(representative_candidate["path"]), representative_image)
        audio_extractor(video_path, representative_audio)
        previous = {
            "memory_id": "PREVIOUS_SHOT",
            "kind": "previous_shot",
            "candidate_index": int(representative_candidate["candidate_index"]),
            "frame_index": int(representative_candidate["frame_index"]),
            "timestamp_sec": float(representative_candidate["timestamp_sec"]),
            "confidence": float(representative_selection["confidence"]),
            "reasoning": str(representative_selection["reasoning"]),
            "visual_status": "representative",
            "source_shot_id": shot_id,
            "audio_source_shot_id": shot_id,
            "local_image_path": str(representative_image.resolve()),
            "local_audio_path": str(representative_audio.resolve()),
        }
    if publisher is not None:
        proposed_bank = publish_memory_records(proposed_bank, publisher=publisher)
        changed_bank = {
            memory_id: proposed_bank[memory_id]
            for memory_id in changed_memory_ids
            if memory_id in proposed_bank
        }
        if previous is not None:
            previous = publish_memory_record(
                previous,
                prefix=f"shot_{shot_id:03d}_PREVIOUS_SHOT",
                publisher=publisher,
            )
    else:
        changed_bank = changed_memory_ids
    review = build_memory_review(
        shot_id=shot_id, attempt=attempt, candidate_count=len(candidates),
        bank=changed_bank, previous_shot=previous, review_id=review_id_factory(),
        rejected_candidate_indices=sorted(rejected), history=history, updated_at=now())
    review["proposed_bank"] = proposed_bank
    review["previous_shot"] = previous
    review["scene_transition"] = transition
    review["scene_transition_reasoning"] = transition_reasoning
    review["selection_warnings"] = []
    review["candidate_paths"] = candidates
    review["selection_mode"] = (
        "manual" if isinstance(selector, ManualMemorySelector) else "vlm"
    )
    review["retained_memory_ids"] = (
        []
        if isinstance(selector, ManualMemorySelector)
        else [
            str(item.get("memory_id") or "")
            for item in review.get("selections", [])
            if isinstance(item, dict) and item.get("memory_id")
        ]
    )
    # Re-read latest shot/state from disk before writing to avoid overwriting
    # concurrent changes (e.g. user accept during a reselect VLM window).
    try:
        latest_shot = _load_shot_for_memory(shot_path, shot_id)
    except ValueError:
        latest_shot = shot
    latest_state = _read_json(state_path, {})
    if isinstance(latest_shot, dict):
        latest_shot["memory_review"] = review
        _write_json(shot_path, latest_shot)
    else:
        shot["memory_review"] = review
        _write_json(shot_path, shot)
    if isinstance(latest_state, dict):
        latest_state["stage"] = "awaiting_memory_review"
        _write_json(state_path, latest_state)
    else:
        _write_json(state_path, state)
    return review


def download_video(locator: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source = Path(locator)
    if source.is_file():
        shutil.copyfile(source, target)
        return
    request = urllib.request.Request(
        locator,
        headers={"Accept": "video/mp4,*/*", "User-Agent": "EchoMemoryAgent/1.0"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        with target.open("wb") as output:
            shutil.copyfileobj(response, output)
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError(f"downloaded empty video: {locator}")


def sample_video_frames(video: Path, output: Path, count: int) -> list[dict[str, Any]]:
    probe = subprocess.run(
        [
            _resolve_media_binary("ffprobe"),
            "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,duration:format=duration",
            "-of", "json", str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = json.loads(probe.stdout)
    stream = (metadata.get("streams") or [{}])[0]
    duration = float(
        stream.get("duration") or (metadata.get("format") or {}).get("duration") or 0
    )
    numerator, _, denominator = str(stream.get("r_frame_rate") or "25/1").partition("/")
    fps = float(numerator) / max(float(denominator or 1), 1.0)
    if duration <= 0 or fps <= 0:
        raise RuntimeError("invalid generated video metadata")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        timestamp = duration * (0.05 + 0.90 * index / max(count - 1, 1))
        image = output / f"candidate_{index:02d}.jpg"
        result = subprocess.run(
            [
                _resolve_media_binary("ffmpeg"),
                "-loglevel", "error", "-y", "-ss", f"{timestamp:.6f}",
                "-i", str(video), "-frames:v", "1", "-vf", "scale=640:-2",
                "-q:v", "3", str(image),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and image.is_file():
            rows.append({
                "candidate_index": index,
                "frame_index": max(0, int(round(timestamp * fps))),
                "timestamp_sec": round(timestamp, 6),
                "path": str(image.resolve()),
            })
    return rows


def extract_video_frame(
    video: Path,
    target: Path,
    timestamp_sec: float,
) -> tuple[float, int]:
    """Extract one exact review frame and return its clamped time and index."""
    probe = subprocess.run(
        [
            _resolve_media_binary("ffprobe"),
            "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,duration:format=duration",
            "-of", "json", str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = json.loads(probe.stdout)
    stream = (metadata.get("streams") or [{}])[0]
    duration = float(
        stream.get("duration") or (metadata.get("format") or {}).get("duration") or 0
    )
    numerator, _, denominator = str(stream.get("r_frame_rate") or "25/1").partition("/")
    fps = float(numerator) / max(float(denominator or 1), 1.0)
    if duration <= 0 or fps <= 0:
        raise RuntimeError("invalid generated video metadata")
    timestamp = min(max(float(timestamp_sec), 0.0), max(duration - (0.5 / fps), 0.0))
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            _resolve_media_binary("ffmpeg"),
            "-loglevel", "error", "-y", "-ss", f"{timestamp:.6f}",
            "-i", str(video), "-frames:v", "1", "-vf", "scale=640:-2",
            "-q:v", "3", str(target),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError(f"failed to extract manual memory frame: {result.stderr[:1000]}")
    return round(timestamp, 6), max(0, int(round(timestamp * fps)))


def select_manual_memory_frame(
    *,
    workspace: Path,
    work_id: str,
    shot_id: int,
    review_id: str,
    attempt: int,
    memory_id: str,
    timestamp_sec: float,
    video_fetcher: Callable[[str, Path], None] = download_video,
    frame_extractor: Callable[[Path, Path, float], tuple[float, int]] = extract_video_frame,
    now: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Replace one VLM proposal image with a user-selected source-video frame."""
    with _MEMORY_REVIEW_LOCK:
        work_dir = workspace / "director" / "works" / work_id
        shot_path = work_dir / "shots" / f"shot_{shot_id:03d}.json"
        shot = _load_shot_for_memory(shot_path, shot_id)
        review = shot.get("memory_review")
        if not isinstance(review, dict):
            raise ValueError(f"shot {shot_id} has no memory review")
        if (
            str(review.get("review_id") or "") != review_id
            or int(review.get("attempt") or 0) != int(attempt)
        ):
            raise MemoryReviewConflict("stale memory review attempt")
        status = str(review.get("status") or "")
        if status not in {"awaiting_review", "error"}:
            raise MemoryReviewConflict(
                f"memory frame cannot be selected from {status}"
            )
        selections = [
            item for item in review.get("selections", []) if isinstance(item, dict)
        ]
        if not any(str(item.get("memory_id") or "") == memory_id for item in selections):
            raise MemoryReviewConflict(f"memory {memory_id} is not in this review")

        video_path = work_dir / "memory" / "videos" / f"shot_{shot_id:03d}.mp4"
        if not video_path.is_file():
            artifact = str(
                shot.get("artifact_url") or shot.get("artifact_path") or ""
            ).strip()
            if not artifact:
                raise ValueError(f"shot {shot_id} has no video artifact")
            video_fetcher(artifact, video_path)

        safe_memory_id = re.sub(r"[^A-Za-z0-9._-]+", "_", memory_id).strip("._")
        safe_memory_id = safe_memory_id or "memory"
        image_path = (
            work_dir
            / "memory"
            / "proposals"
            / f"shot_{shot_id:03d}_attempt_{attempt:02d}"
            / f"{safe_memory_id}_manual.jpg"
        )
        selected_time, frame_index = frame_extractor(
            video_path, image_path, float(timestamp_sec)
        )

        if memory_id == "PREVIOUS_SHOT":
            source = review.get("previous_shot")
            kind = "previous_shot"
        else:
            proposed_bank = review.get("proposed_bank")
            source = proposed_bank.get(memory_id) if isinstance(proposed_bank, dict) else None
            kind = "character"
        if not isinstance(source, dict):
            raise MemoryReviewConflict(f"memory {memory_id} has no proposal record")

        replacement = deepcopy(source)
        for key in ("local_image_path", "image_path", "image_url", "image"):
            replacement.pop(key, None)
        replacement.update(
            {
                "memory_id": memory_id,
                "kind": kind,
                "candidate_index": -1,
                "frame_index": frame_index,
                "timestamp_sec": selected_time,
                "confidence": 1.0,
                "reasoning": "Selected manually from the source video.",
                "visual_status": (
                    "representative" if kind == "previous_shot" else "confirmed"
                ),
                "source_shot_id": shot_id,
                "local_image_path": str(image_path.resolve()),
            }
        )
        if memory_id == "PREVIOUS_SHOT":
            review["previous_shot"] = replacement
        else:
            proposed_bank = review.get("proposed_bank")
            if not isinstance(proposed_bank, dict):
                proposed_bank = {}
                review["proposed_bank"] = proposed_bank
            proposed_bank[memory_id] = replacement

        review["selections"] = [
            _review_selection(memory_id, replacement, kind=kind)
            if str(item.get("memory_id") or "") == memory_id
            else item
            for item in selections
        ]
        selected_ids = review.setdefault("manual_selected_ids", [])
        if not isinstance(selected_ids, list):
            selected_ids = []
            review["manual_selected_ids"] = selected_ids
        if memory_id not in selected_ids:
            selected_ids.append(memory_id)
        retained_ids = review.setdefault("retained_memory_ids", [])
        if not isinstance(retained_ids, list):
            retained_ids = []
            review["retained_memory_ids"] = retained_ids
        if memory_id not in retained_ids:
            retained_ids.append(memory_id)
        review["status"] = "awaiting_review"
        review["error"] = None
        review["updated_at"] = (
            now()
            if now is not None
            else datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
                "+00:00", "Z"
            )
        )
        shot["memory_review"] = review
        _write_json(shot_path, shot)
        return review


def extract_video_audio(video: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            _resolve_media_binary("ffmpeg"),
            "-loglevel", "error", "-y", "-i", str(video), "-vn",
            "-acodec", "pcm_s16le", "-ar", "48000", str(target),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not target.is_file() or target.stat().st_size <= 44:
        raise RuntimeError(f"failed to extract memory audio: {result.stderr[:1000]}")


def run_memory_review_from_config(
    *,
    workspace: Path,
    work_id: str,
    shot_id: int,
    target_memory_id: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    vlm_model: str | None = None,
    candidate_count: int = 24,
    selection_mode: str = "vlm",
) -> dict[str, Any]:
    """Production callback entrypoint using an explicitly resolved VLM route."""
    resolved_api_base = (api_base or "").strip()
    resolved_api_key = (api_key or "").strip()
    resolved_model = (vlm_model or "").strip()
    if selection_mode not in {"manual", "vlm"}:
        raise ValueError("selection_mode must be manual or vlm")
    if selection_mode == "vlm" and not all(
        (resolved_api_base, resolved_api_key, resolved_model)
    ):
        raise RuntimeError(
            "memory review requires tools.memoryReview.provider/model and "
            "the referenced provider apiKey/apiBase"
        )
    with _MEMORY_REVIEW_LOCK:
        # JD's OpenAI-compatible Qwen VLM route rejects prompts containing
        # more than 16 images. Keep the richer 24-frame strip for manual
        # scrubbing, but cap automatic review before frames are sampled and
        # embedded into the VLM prompt.
        effective_candidate_count = (
            min(candidate_count, _MAX_VLM_PROMPT_IMAGES)
            if selection_mode == "vlm"
            else candidate_count
        )
        selector = (
            ManualMemorySelector()
            if selection_mode == "manual"
            else MemoryVlmSelector(
                api_base=resolved_api_base,
                api_key=resolved_api_key,
                model=resolved_model,
            )
        )
        review = prepare_memory_review(
            workspace=workspace,
            work_id=work_id,
            shot_id=shot_id,
            selector=selector,
            video_fetcher=download_video,
            frame_sampler=sample_video_frames,
            audio_extractor=extract_video_audio,
            review_id_factory=lambda: f"memory-review-{uuid.uuid4().hex}",
            now=lambda: datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z"),
            candidate_count=effective_candidate_count,
            target_memory_id=target_memory_id,
            publisher=(
                configured_file_publisher(work_id)
                if selection_mode == "vlm"
                else None
            ),
        )
        if selection_mode == "manual":
            review["selection_mode"] = selection_mode
            review["required_memory_ids"] = [
                str(item.get("memory_id") or "")
                for item in review.get("selections", [])
                if isinstance(item, dict) and item.get("memory_id")
            ]
            review["manual_selected_ids"] = []
            review["retained_memory_ids"] = []
            shot_path = (
                workspace / "director" / "works" / work_id / "shots"
                / f"shot_{shot_id:03d}.json"
            )
            shot = _load_shot_for_memory(shot_path, shot_id)
            shot["memory_review"] = review
            _write_json(shot_path, shot)
        return review
