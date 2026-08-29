"""Approve selected memories and prepare exactly one next R2V shot."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable

from nanobot.storage.files import configured_file_publisher
from nanobot.utils.helpers import write_json_atomic

Publisher = Callable[[str, str], str]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    write_json_atomic(path, value)


def stage_after_memory_advance(state: dict[str, Any], *, has_next: bool) -> str:
    """Keep merge in flight. Memory approval must not clobber a pending merge."""
    pending = state.get("pending_remote_jobs")
    kinds: set[str] = set()
    if isinstance(pending, dict):
        kinds = {
            str(item.get("kind"))
            for item in pending.values()
            if isinstance(item, dict)
        }
    if "merge_shot" in kinds or str(state.get("stage") or "") == "merging":
        return "merging"
    if "generate_echo_shot" in kinds:
        return "shot_generating"
    return "shot_generating" if has_next else "shot_reviewing"


def publish_memory_record(
    record: dict[str, Any],
    *,
    prefix: str,
    publisher: Publisher,
) -> dict[str, Any]:
    published = dict(record)
    image = str(record.get("local_image_path") or "").strip()
    audio = str(record.get("local_audio_path") or "").strip()
    source_shot_id = int(record.get("source_shot_id") or 0)
    frame_index = int(record.get("frame_index") or 0)
    object_prefix = (
        f"shot_{source_shot_id:03d}/"
        f"{prefix}_frame_{frame_index:06d}"
    )
    if image:
        published["image_path"] = publisher(image, f"{object_prefix}.jpg")
    if audio:
        published["audio_path"] = publisher(audio, f"{object_prefix}.wav")
    published.pop("local_image_path", None)
    published.pop("local_audio_path", None)
    if getattr(publisher, "delete_local_after_upload", False):
        for path in (image, audio):
            if path:
                Path(path).unlink(missing_ok=True)
    return published


def publish_memory_records(
    records: dict[str, dict[str, Any]],
    *,
    publisher: Publisher,
) -> dict[str, dict[str, Any]]:
    """Publish a character bank while preserving its metadata and key order."""
    return {
        memory_id: publish_memory_record(record, prefix=memory_id, publisher=publisher)
        for memory_id, record in records.items()
        if isinstance(record, dict)
    }


def _character_ids(caption: str) -> list[str]:
    return list(dict.fromkeys(re.findall(
        r"(?<![A-Za-z0-9_-])ID_[A-Za-z0-9_-]+(?![A-Za-z0-9_-])",
        caption,
    )))


def _memory_slots(
    bank: dict[str, dict[str, Any]],
    caption: str,
    previous: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for memory_id in _character_ids(caption):
        record = bank.get(memory_id)
        if not isinstance(record, dict) or not record.get("image_path"):
            continue
        slot = {
            "image_url": record["image_path"],
            "metadata": {
                "id": memory_id,
                "workspace_asset_id": _automatic_asset_id(
                    record, memory_id=memory_id, kind="character"
                ),
                "source": "character_memory_bank",
                "visual_status": record.get("visual_status", "confirmed"),
                "source_shot_id": record.get("source_shot_id"),
                "frame_index": record.get("frame_index"),
                "timestamp_sec": record.get("timestamp_sec"),
                "confidence": record.get("confidence"),
            },
        }
        if record.get("audio_path"):
            slot["audio_url"] = record["audio_path"]
            slot["metadata"]["audio_source_shot_id"] = record.get(
                "audio_source_shot_id")
        else:
            slot["audio_mode"] = "empty"
        slots.append(slot)
    if previous and previous.get("image_path"):
        slot = {
            "image_url": previous["image_path"],
            "metadata": {
                "id": "PREVIOUS_SHOT",
                "workspace_asset_id": _automatic_asset_id(
                    previous, memory_id="PREVIOUS_SHOT", kind="previous_shot"
                ),
                "source": "previous_shot_representative",
                "visual_status": "representative",
                "source_shot_id": previous.get("source_shot_id"),
                "frame_index": previous.get("frame_index"),
                "timestamp_sec": previous.get("timestamp_sec"),
                "confidence": previous.get("confidence"),
            },
        }
        if previous.get("audio_path"):
            slot["audio_url"] = previous["audio_path"]
            slot["metadata"]["audio_source_shot_id"] = previous.get(
                "audio_source_shot_id")
        else:
            slot["audio_mode"] = "empty"
        slots.append(slot)
    if len(slots) > 7:
        raise ValueError("active character memory count exceeds 7")
    return slots


def _automatic_asset_id(
    raw: dict[str, Any], *, memory_id: str, kind: str
) -> str:
    """Match the stable id used by the WebSocket Memory Workspace."""
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


def _remote_review_selections(
    selections: Any,
    bank: dict[str, dict[str, Any]],
    previous: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for raw in selections if isinstance(selections, list) else []:
        if not isinstance(raw, dict):
            continue
        memory_id = str(raw.get("memory_id") or "").strip()
        source = previous if memory_id == "PREVIOUS_SHOT" else bank.get(memory_id)
        item = dict(raw)
        item.pop("local_image_path", None)
        item.pop("local_audio_path", None)
        if isinstance(source, dict):
            for key in ("image_path", "audio_path"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    item[key] = value.strip()
        projected.append(item)
    return projected


def _cleanup_local_review_artifacts(
    work: Path,
    shot_id: int,
    review: dict[str, Any],
) -> None:
    review.pop("candidate_paths", None)
    for attempt in review.get("history", []):
        if not isinstance(attempt, dict):
            continue
        for item in attempt.get("selections", []):
            if isinstance(item, dict):
                item.pop("local_image_path", None)
                item.pop("local_audio_path", None)
    memory = work / "memory"
    (memory / "videos" / f"shot_{shot_id:03d}.mp4").unlink(missing_ok=True)
    shutil.rmtree(memory / "candidates" / f"shot_{shot_id:03d}", ignore_errors=True)
    proposals = memory / "proposals"
    for path in proposals.glob(f"shot_{shot_id:03d}_attempt_*"):
        shutil.rmtree(path, ignore_errors=True)


def approve_review_and_prepare_next(
    *,
    workspace: Path,
    work_id: str,
    shot_id: int,
    publisher: Publisher | None = None,
) -> int | None:
    work = workspace / "director" / "works" / work_id
    current_path = work / "shots" / f"shot_{shot_id:03d}.json"
    current = _read(current_path)
    review = current.get("memory_review")
    if not isinstance(review, dict) or review.get("status") != "approved":
        raise ValueError("memory review must be approved before advancing")
    proposed = review.get("proposed_bank")
    if not isinstance(proposed, dict):
        raise ValueError("approved memory review has no proposed bank")

    selections = [
        item for item in review.get("selections", []) if isinstance(item, dict)
    ]
    previous_raw = review.get("previous_shot")
    if "retained_memory_ids" in review:
        retained_ids = {
            str(value)
            for value in review.get("retained_memory_ids", [])
            if value
        }
        bank_path = work / "memory" / "memory_bank.json"
        current_bank = _read(bank_path) if bank_path.is_file() else {}
        retained_proposed = dict(current_bank)
        for item in selections:
            memory_id = str(item.get("memory_id") or "")
            record = proposed.get(memory_id)
            if (
                memory_id
                and memory_id != "PREVIOUS_SHOT"
                and memory_id in retained_ids
                and isinstance(record, dict)
            ):
                retained_proposed[memory_id] = record
        proposed = retained_proposed
        selections = [
            item
            for item in selections
            if str(item.get("memory_id") or "") in retained_ids
        ]
        if "PREVIOUS_SHOT" not in retained_ids:
            previous_raw = None

    publisher = publisher or configured_file_publisher(work_id)
    published_bank = publish_memory_records(proposed, publisher=publisher)
    previous = (
        publish_memory_record(
            previous_raw,
            prefix=f"shot_{shot_id:03d}_PREVIOUS_SHOT",
            publisher=publisher,
        )
        if isinstance(previous_raw, dict)
        else None
    )
    review["proposed_bank"] = published_bank
    review["previous_shot"] = previous
    review["selections"] = _remote_review_selections(
        selections, published_bank, previous
    )
    if getattr(publisher, "delete_local_after_upload", False):
        _cleanup_local_review_artifacts(work, shot_id, review)
    current["memory_review"] = review
    _write(current_path, current)
    _write(work / "memory" / "memory_bank.json", published_bank)
    if previous:
        _write(work / "memory" / "previous_shot.json", previous)
    else:
        (work / "memory" / "previous_shot.json").unlink(missing_ok=True)

    next_path: Path | None = None
    next_shot: dict[str, Any] | None = None
    for path in sorted((work / "shots").glob("shot_*.json")):
        candidate = _read(path)
        candidate_id = int(candidate.get("shot_id") or 0)
        if candidate_id <= shot_id:
            continue
        if str(candidate.get("status") or "") in {
            "queued", "generated", "review_pass", "approved"
        }:
            continue
        next_path = path
        next_shot = candidate
        break

    state_path = work / "state.json"
    state = _read(state_path)
    if next_path is None or next_shot is None:
        state["stage"] = stage_after_memory_advance(state, has_next=False)
        _write(state_path, state)
        return None

    next_caption = str(next_shot.get("caption") or "")
    # Extraction creates a recommendation draft only. In interactive mode the
    # user must explicitly apply that draft in Build Memory before generation.
    # An agent recommendation may replace this conservative identity-based
    # draft later, but neither path may silently approve it.
    if not next_shot.get("memory_slots_user_configured"):
        recommended = _memory_slots(published_bank, next_caption, previous)
        next_shot["recommended_memory_slots"] = recommended
        next_shot["recommended_memory_display_slots"] = recommended
        next_shot["memory_recommendation_source"] = (
            "profile_fallback"
            if state.get("auto_generate")
            else "pending_agent"
        )
        next_shot.pop("approved_memory_slots", None)
        next_shot.pop("approved_memory_display_slots", None)
    _write(next_path, next_shot)
    advanced_stage = stage_after_memory_advance(state, has_next=True)
    state["stage"] = (
        "awaiting_memory_build"
        if advanced_stage == "shot_generating"
        else advanced_stage
    )
    _write(state_path, state)
    return int(next_shot["shot_id"])


def auto_approve_review_and_prepare_next(
    *,
    workspace: Path,
    work_id: str,
    shot_id: int,
    publisher: Publisher | None = None,
) -> int | None:
    """Approve a fresh Memory proposal and prepare its next shot exactly once."""
    work = workspace / "director" / "works" / work_id
    current_path = work / "shots" / f"shot_{shot_id:03d}.json"
    current = _read(current_path)
    review = current.get("memory_review")
    if not isinstance(review, dict):
        raise ValueError("generated shot has no memory review")

    if review.get("auto_advance_complete") is True:
        next_id = review.get("auto_advance_next_shot_id")
        return int(next_id) if next_id is not None else None
    if review.get("status") == "awaiting_review":
        review["status"] = "approved"
        current["memory_review"] = review
        _write(current_path, current)
    elif review.get("status") != "approved":
        raise ValueError("memory review is not ready for automatic approval")

    next_id = approve_review_and_prepare_next(
        workspace=workspace,
        work_id=work_id,
        shot_id=shot_id,
        publisher=publisher,
    )
    if next_id is not None:
        next_path = work / "shots" / f"shot_{next_id:03d}.json"
        next_shot = _read(next_path)
        recommended = next_shot.get("recommended_memory_slots")
        if isinstance(recommended, list):
            next_shot["approved_memory_slots"] = recommended
            next_shot["approved_memory_display_slots"] = next_shot.get(
                "recommended_memory_display_slots", recommended
            )
            next_shot["memory_slots_auto_approved"] = True
            _write(next_path, next_shot)
        state_path = work / "state.json"
        state = _read(state_path)
        if str(state.get("stage") or "") == "awaiting_memory_build":
            state["stage"] = "shot_generating"
            _write(state_path, state)
    current = _read(current_path)
    review = current.get("memory_review")
    if isinstance(review, dict):
        review["auto_advance_complete"] = True
        review["auto_advance_next_shot_id"] = next_id
        current["memory_review"] = review
        _write(current_path, current)
    return next_id
