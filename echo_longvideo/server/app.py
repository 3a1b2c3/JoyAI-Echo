"""Local Echo 1.5 inference scheduler and media assembly API.

The control plane schedules work from in-memory queues and mirrors job state to
SQLite for restart recovery. GPU workers admit work from live GPU/RAM telemetry,
stage conditioning and generation weights, execute the shared local pipeline,
index finished artifacts, and notify callbacks.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from r2v_schema import MAX_MEMORY_SLOTS
from .state import JobJournal

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_AUDIO_BYTES = 100 * 1024 * 1024
_DATA_MIME_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _callback_retry_at(attempts_before_failure: int) -> str:
    delay_seconds = min(2 ** max(attempts_before_failure, 0), 30)
    return (datetime.now(UTC) + timedelta(seconds=delay_seconds)).isoformat()


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _dit_residency_policy(value: str) -> str:
    policy = value.strip().lower()
    if policy not in {"auto", "resident", "swap"}:
        raise ValueError("ECHO_DIT_RESIDENCY must be auto, resident, or swap")
    return policy


def _decode_data_url(value: str, *, kind: str, max_bytes: int) -> tuple[bytes, str]:
    """Decode a bounded base64 data URL and return its bytes and extension."""

    header, separator, encoded = value.partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        raise ValueError(f"{kind} data URL must use base64 encoding")
    mime_type = header[5:-7].strip().lower()
    expected_prefix = "image/" if kind == "image" else "audio/"
    if not mime_type.startswith(expected_prefix):
        raise ValueError(f"{kind} data URL has incompatible MIME type: {mime_type}")
    extension = _DATA_MIME_EXTENSIONS.get(mime_type)
    if not extension:
        raise ValueError(f"unsupported {kind} data URL MIME type: {mime_type}")
    if len(encoded) > ((max_bytes + 2) // 3) * 4 + 4:
        raise ValueError(f"{kind} data URL exceeds {max_bytes} decoded bytes")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError(f"invalid base64 in {kind} data URL") from error
    if not payload or len(payload) > max_bytes:
        raise ValueError(f"{kind} data URL must contain 1..{max_bytes} bytes")
    return payload, extension


def _config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"server config section {name!r} must be a mapping")
    return value


def load_server_config(path: str | Path) -> dict[str, Any]:
    config_path = _repo_path(str(path))
    with config_path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source) or {}
    if not isinstance(config, dict):
        raise ValueError("server config root must be a mapping")
    for section in ("inference", "runtime", "server"):
        _config_section(config, section)
    return config


def _environment_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def apply_server_config(path: str | Path) -> dict[str, Any]:
    """Load deployment settings while preserving explicit environment overrides."""

    config = load_server_config(path)
    inference = _config_section(config, "inference")
    runtime = _config_section(config, "runtime")
    server = _config_section(config, "server")
    mappings = (
        ("ECHO_INFERENCE_CONFIG", inference, "config"),
        ("ECHO_CHECKPOINT", inference, "checkpoint"),
        ("ECHO_CONDITIONING_CACHE_DIR", inference, "conditioning_cache_dir"),
        ("ECHO_GPU_IDS", runtime, "gpu_ids"),
        ("ECHO_DISABLE_INFERENCE_WORKERS", runtime, "disable_inference_workers"),
        ("ECHO_DIT_RESIDENCY", runtime, "dit_residency"),
        ("ECHO_GPU_HEADROOM_FRACTION", runtime, "gpu_headroom_fraction"),
        ("ECHO_RAM_HEADROOM_FRACTION", runtime, "ram_headroom_fraction"),
        ("ECHO_MODEL_IDLE_SECONDS", runtime, "model_idle_seconds"),
        ("ECHO_ARTIFACT_DB_PATH", server, "artifact_db_path"),
        ("ECHO_R2V_TIMEOUT_SECONDS", server, "request_timeout_seconds"),
        ("ECHO_R2V_POLL_SECONDS", server, "poll_seconds"),
        ("ECHO_PUBLIC_BASE_URL", server, "public_base_url"),
        ("ECHO_R2V_QUEUE_CAPACITY", server, "queue_capacity"),
        ("ECHO_CALLBACK_MAX_ATTEMPTS", server, "callback_max_attempts"),
        ("ECHO_MEDIA_ROOT", server, "media_root"),
        ("ECHO_FFMPEG_PATH", server, "ffmpeg_path"),
        ("ECHO_MERGE_TIMEOUT_SECONDS", server, "merge_timeout_seconds"),
        ("ECHO_MERGE_MAX_INPUT_BYTES", server, "merge_max_input_bytes"),
    )
    for environment_name, section, key in mappings:
        value = section.get(key)
        if value is not None:
            os.environ.setdefault(environment_name, _environment_value(value))
    return config


class MemorySlot(BaseModel):
    """One ordered memory slot accepted by the production R2V endpoint."""

    shot_id: str | None = None
    image_url: str | None = None
    image_mode: str | None = None
    audio_url: str | None = None
    audio_mode: str | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "MemorySlot":
        self.shot_id = (self.shot_id or "").strip() or None
        self.image_url = (self.image_url or "").strip() or None
        self.image_mode = (self.image_mode or "").strip().lower() or None
        self.audio_url = (self.audio_url or "").strip() or None
        audio_mode = (self.audio_mode or "").strip().lower()

        if self.image_mode:
            raise ValueError("image_mode is not supported")
        if bool(self.shot_id) == bool(self.image_url):
            raise ValueError("exactly one of shot_id or image_url is required")
        if audio_mode not in {"", "empty"}:
            raise ValueError("audio_mode currently supports only 'empty'")
        self.audio_mode = audio_mode or None
        if self.shot_id and (self.audio_url or self.audio_mode):
            raise ValueError("shot_id cannot be combined with audio_url or audio_mode")
        if self.audio_url and self.audio_mode == "empty":
            raise ValueError("audio_url conflicts with audio_mode='empty'")
        if self.image_url and not self.audio_url and self.audio_mode is None:
            self.audio_mode = "empty"
        return self


class R2VGenerateRequest(BaseModel):
    work_id: str = Field(min_length=1)
    shot_id: str = Field(min_length=1)
    job_id: str | None = None
    prompt: str = Field(min_length=1)
    memory_slots: list[MemorySlot] = Field(max_length=MAX_MEMORY_SLOTS)
    condition_img: str | None = None
    callback_url: str | None = None
    callback_context: dict[str, Any] = Field(default_factory=dict)
    num_frames: int | None = Field(default=None, gt=0)
    duration_sec: float | None = Field(default=None, gt=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    seed: int | None = None

    @field_validator("work_id", "shot_id", "prompt")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class MergeShotInput(BaseModel):
    """One ordered video input for a merge job."""

    version_id: str | None = None
    video_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("video_url", "artifact_url", "result_url"),
    )

    @model_validator(mode="after")
    def validate_source(self) -> "MergeShotInput":
        self.version_id = (self.version_id or "").strip() or None
        self.video_url = (self.video_url or "").strip() or None
        if not self.version_id and not self.video_url:
            raise ValueError("version_id or video_url is required")
        return self


class MergeRequest(BaseModel):
    """Ordered shot list assembled into one server-owned MP4 artifact."""

    work_id: str = Field(min_length=1)
    job_id: str | None = None
    shots: list[MergeShotInput] = Field(min_length=1)
    callback_url: str | None = None
    callback_context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("work_id")
    @classmethod
    def strip_work_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


@dataclass(frozen=True)
class Settings:
    artifact_db_path: Path
    request_timeout: float
    poll_interval: float
    public_base_url: str
    queue_capacity: int
    callback_attempts: int
    media_root: Path
    ffmpeg_binary: str
    merge_timeout: float
    merge_max_input_bytes: int
    inference_config: Path
    checkpoint: str | None
    conditioning_cache_dir: Path
    gpu_ids: str
    disable_inference_workers: bool
    dit_residency: str
    gpu_headroom_fraction: float
    ram_headroom_fraction: float
    model_idle_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            artifact_db_path=_repo_path(
                os.environ.get("ECHO_ARTIFACT_DB_PATH", "data/artifacts.sqlite3")
            ),
            request_timeout=float(os.environ.get("ECHO_R2V_TIMEOUT_SECONDS", "30")),
            poll_interval=max(float(os.environ.get("ECHO_R2V_POLL_SECONDS", "5")), 0.1),
            public_base_url=os.environ.get(
                "ECHO_PUBLIC_BASE_URL", "http://127.0.0.1:8221"
            ).strip().rstrip("/"),
            queue_capacity=max(int(os.environ.get("ECHO_R2V_QUEUE_CAPACITY", "1000")), 1),
            callback_attempts=max(int(os.environ.get("ECHO_CALLBACK_MAX_ATTEMPTS", "0")), 0),
            media_root=_repo_path(os.environ.get("ECHO_MEDIA_ROOT", "data/media")),
            ffmpeg_binary=os.environ.get("ECHO_FFMPEG_PATH", "ffmpeg").strip() or "ffmpeg",
            merge_timeout=max(float(os.environ.get("ECHO_MERGE_TIMEOUT_SECONDS", "3600")), 1),
            merge_max_input_bytes=max(
                int(os.environ.get("ECHO_MERGE_MAX_INPUT_BYTES", str(2 * 1024**3))),
                1,
            ),
            inference_config=_repo_path(
                os.environ.get("ECHO_INFERENCE_CONFIG", "configs/inference.bf16.yaml")
            ),
            checkpoint=os.environ.get("ECHO_CHECKPOINT", "").strip() or None,
            conditioning_cache_dir=_repo_path(
                os.environ.get("ECHO_CONDITIONING_CACHE_DIR", "data/conditioning_cache")
            ),
            gpu_ids=os.environ.get("ECHO_GPU_IDS", "0").strip(),
            disable_inference_workers=os.environ.get(
                "ECHO_DISABLE_INFERENCE_WORKERS", "0"
            ).strip().lower()
            in {"1", "true", "yes"},
            dit_residency=_dit_residency_policy(
                os.environ.get("ECHO_DIT_RESIDENCY", "auto")
            ),
            gpu_headroom_fraction=min(
                max(float(os.environ.get("ECHO_GPU_HEADROOM_FRACTION", "0.05")), 0),
                0.5,
            ),
            ram_headroom_fraction=min(
                max(float(os.environ.get("ECHO_RAM_HEADROOM_FRACTION", "0.10")), 0),
                0.5,
            ),
            model_idle_seconds=max(
                float(os.environ.get("ECHO_MODEL_IDLE_SECONDS", "0")), 0
            ),
        )


class JobStore:
    """In-memory FIFO mirrored to a durable restart journal."""

    def __init__(
        self,
        capacity: int,
        *,
        journal: JobJournal | None = None,
        journal_kind: str = "r2v",
    ) -> None:
        self.capacity = capacity
        self.journal = journal
        self.journal_kind = journal_kind
        self._jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def initialize(self) -> None:
        if self.journal is None:
            return
        self.journal.initialize()
        restored = self.journal.load(self.journal_kind)
        with self._lock:
            self._jobs.clear()
            for job in restored:
                if job.get("status") == "running":
                    job.update(
                        status="queued",
                        stage="recovered",
                        gpu_id=None,
                        resource_json=None,
                        updated_at=_now(),
                    )
                    self.journal.save(self.journal_kind, job)
                self._jobs[str(job["version_id"])] = job
            self._prune_terminal()

    def _persist(self, job: dict[str, Any]) -> None:
        if self.journal is not None:
            self.journal.save(self.journal_kind, job)

    def _prune_terminal(self) -> None:
        while len(self._jobs) > self.capacity * 2:
            terminal_id = next(
                (
                    version_id
                    for version_id, job in self._jobs.items()
                    if job["status"] in {"succeeded", "failed"}
                    and (
                        not job.get("callback_url")
                        or job.get("callback_status") == "delivered"
                    )
                ),
                None,
            )
            if terminal_id is None:
                return
            self._jobs.pop(terminal_id, None)

    def enqueue(
        self,
        version_id: str,
        request: R2VGenerateRequest,
        payload: dict[str, Any],
        callback_url: str | None,
    ) -> str:
        now = _now()
        request_json = _canonical_json(payload)
        with self._lock:
            if request.job_id:
                existing = next(
                    (
                        job
                        for job in self._jobs.values()
                        if job.get("agent_job_id") == request.job_id
                    ),
                    None,
                )
                if existing is None and self.journal is not None:
                    existing = self.journal.get_by_agent_job_id(
                        self.journal_kind, request.job_id
                    )
                if existing is not None:
                    if existing["request_json"] != request_json:
                        raise ValueError(
                            "job_id is already associated with a different R2V request"
                        )
                    existing.update(
                        callback_url=callback_url or existing.get("callback_url"),
                        callback_context_json=json.dumps(
                            request.callback_context, ensure_ascii=False
                        ),
                        updated_at=now,
                    )
                    self._persist(existing)
                    return str(existing["version_id"])
            pending = sum(
                job["status"] in {"queued", "running"} for job in self._jobs.values()
            )
            if pending >= self.capacity:
                raise OverflowError("R2V submission queue is full")
            self._jobs[version_id] = {
                "version_id": version_id,
                "work_id": request.work_id,
                "shot_id": request.shot_id,
                "agent_job_id": request.job_id,
                "request_json": request_json,
                "callback_url": callback_url,
                "callback_context_json": json.dumps(
                    request.callback_context, ensure_ascii=False
                ),
                "callback_status": None,
                "callback_response": None,
                "callback_error": None,
                "callback_attempts": 0,
                "callback_next_at": None,
                "status": "queued",
                "stage": "queued",
                "result_json": None,
                "gpu_id": None,
                "resource_json": None,
                "error": None,
                "attempts": 0,
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "completed_at": None,
            }
            self._persist(self._jobs[version_id])
            self._prune_terminal()
            return version_id

    def claim_batch(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                job for job in self._jobs.values() if job["status"] == "queued"
            ][: max(int(limit), 1)]
            if not rows:
                return []
            now = _now()
            for job in rows:
                job.update(
                    status="running",
                    stage="claimed",
                    attempts=int(job["attempts"]) + 1,
                    started_at=job["started_at"] or now,
                    updated_at=now,
                    error=None,
                )
                self._persist(job)
            return [dict(job) for job in rows]

    def claim_next(self) -> dict[str, Any] | None:
        jobs = self.claim_batch(1)
        return jobs[0] if jobs else None

    def update_stage(
        self,
        version_id: str,
        stage: str,
        *,
        gpu_id: int | None = None,
        resources: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs[version_id]
            job.update(
                status="running",
                stage=stage,
                gpu_id=gpu_id,
                resource_json=json.dumps(resources, ensure_ascii=False)
                if resources
                else None,
                updated_at=_now(),
            )
            self._persist(job)

    def complete(self, version_id: str, result: dict[str, Any]) -> None:
        now = _now()
        with self._lock:
            self._jobs[version_id].update(
                status="succeeded",
                stage="succeeded",
                result_json=json.dumps(result, ensure_ascii=False),
                updated_at=now,
                completed_at=now,
                error=None,
            )
            self._persist(self._jobs[version_id])
            self._prune_terminal()

    def fail(self, version_id: str, error: str) -> None:
        now = _now()
        with self._lock:
            self._jobs[version_id].update(
                status="failed",
                stage="failed",
                error=error[:4000],
                updated_at=now,
                completed_at=now,
            )
            self._persist(self._jobs[version_id])
            self._prune_terminal()

    def callback_candidates(self, max_attempts: int) -> list[dict[str, Any]]:
        now = _now()
        with self._lock:
            return [
                dict(job)
                for job in self._jobs.values()
                if job["status"] in {"succeeded", "failed"}
                and job["callback_url"]
                and job["callback_status"] != "delivered"
                and (max_attempts == 0 or job["callback_attempts"] < max_attempts)
                and (job["callback_next_at"] is None or job["callback_next_at"] <= now)
            ]

    def record_callback(
        self,
        version_id: str,
        *,
        delivered: bool,
        response: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs[version_id]
            attempts = int(job["callback_attempts"])
            job.update(
                callback_status="delivered" if delivered else "error",
                callback_response=response[:4000] if response else None,
                callback_error=error[:1000] if error else None,
                callback_attempts=attempts + 1,
                callback_next_at=None
                if delivered
                else _callback_retry_at(attempts),
                updated_at=_now(),
            )
            self._persist(job)

    def get(self, version_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(version_id)
            if job is not None:
                return dict(job)
        if self.journal is not None:
            return self.journal.get(self.journal_kind, version_id)
        return None

    def queue_position(self, version_id: str) -> int | None:
        job = self.get(version_id)
        if job is None or job["status"] != "queued":
            return None
        with self._lock:
            queued = [
                candidate["version_id"]
                for candidate in self._jobs.values()
                if candidate["status"] == "queued"
            ]
        return queued.index(version_id) + 1

    def counts(self) -> dict[str, int]:
        result = {name: 0 for name in ("queued", "running", "succeeded", "failed")}
        with self._lock:
            for job in self._jobs.values():
                result[str(job["status"])] += 1
        return result


class ArtifactStore:
    """Small metadata index for generated and merged video artifacts."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    work_id TEXT NOT NULL,
                    shot_id TEXT,
                    kind TEXT NOT NULL,
                    role TEXT NOT NULL,
                    url TEXT NOT NULL,
                    local_path TEXT,
                    size_bytes INTEGER,
                    sha256 TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(version_id, role)
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(artifacts)").fetchall()
            }
            if "shot_id" not in columns:
                connection.execute("ALTER TABLE artifacts ADD COLUMN shot_id TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_work_id "
                "ON artifacts(work_id, created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_shot "
                "ON artifacts(work_id, shot_id, created_at)"
            )
            for row in connection.execute(
                "SELECT artifact_id, metadata_json FROM artifacts WHERE shot_id IS NULL"
            ).fetchall():
                metadata = _json_loads(row["metadata_json"], {})
                request = metadata.get("request") if isinstance(metadata, dict) else None
                shot_id = (
                    request.get("shot_id")
                    if isinstance(request, dict)
                    else metadata.get("shot_id") if isinstance(metadata, dict) else None
                )
                if shot_id:
                    connection.execute(
                        "UPDATE artifacts SET shot_id = ? WHERE artifact_id = ?",
                        (str(shot_id), row["artifact_id"]),
                    )

    def upsert(
        self,
        *,
        version_id: str,
        work_id: str,
        shot_id: str | None = None,
        kind: str,
        role: str,
        url: str,
        local_path: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _now()
        artifact_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"echo:{version_id}:{role}"))
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, version_id, work_id, shot_id, kind, role, url, local_path,
                    size_bytes, sha256, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(version_id, role) DO UPDATE SET
                    work_id = excluded.work_id,
                    shot_id = excluded.shot_id,
                    kind = excluded.kind,
                    url = excluded.url,
                    local_path = excluded.local_path,
                    size_bytes = excluded.size_bytes,
                    sha256 = excluded.sha256,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    artifact_id,
                    version_id,
                    work_id,
                    shot_id,
                    kind,
                    role,
                    url,
                    local_path,
                    size_bytes,
                    sha256,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return self.get(artifact_id) or {}

    def get(self, artifact_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
        return self._public(dict(row)) if row else None

    def for_version(self, version_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM artifacts WHERE version_id = ? ORDER BY created_at, role",
                (version_id,),
            ).fetchall()
        return [self._public(dict(row)) for row in rows]

    def latest_local_for_shot(self, work_id: str, shot_id: str) -> dict[str, Any] | None:
        """Return the newest generated primary artifact for one logical shot."""

        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM artifacts
                WHERE work_id = ? AND shot_id = ? AND kind = 'r2v' AND role = 'primary'
                ORDER BY created_at DESC LIMIT 1
                """,
                (work_id, shot_id),
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _public(row: dict[str, Any]) -> dict[str, Any]:
        row["metadata"] = _json_loads(row.pop("metadata_json", None), {})
        # Filesystem layout is an implementation detail and is never returned.
        row.pop("local_path", None)
        return row


class RequestAssetResolver:
    """Materialize inline assets and prior-shot references for local inference."""

    def __init__(self, settings: Settings, artifacts: ArtifactStore) -> None:
        self.settings = settings
        self.artifacts = artifacts
        self.root = settings.media_root / "request_assets"
        self._lock = threading.Lock()

    def _write_inline(self, value: str, *, kind: str) -> str:
        limit = MAX_IMAGE_BYTES if kind == "image" else MAX_AUDIO_BYTES
        payload, extension = _decode_data_url(value, kind=kind, max_bytes=limit)
        digest = sha256(payload).hexdigest()
        target = self.root / "inline" / f"{digest}{extension}"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_bytes(payload)
            try:
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return str(target.resolve())

    def _extract_reference_assets(
        self,
        *,
        work_id: str,
        shot_id: str,
    ) -> tuple[str, str]:
        artifact = self.artifacts.latest_local_for_shot(work_id, shot_id)
        if artifact is None:
            raise LookupError(
                f"memory shot is not available: work_id={work_id} shot_id={shot_id}"
            )
        source = Path(str(artifact.get("local_path") or "")).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"memory shot artifact is missing: work_id={work_id} shot_id={shot_id}"
            )
        binary = shutil.which(self.settings.ffmpeg_binary)
        if binary is None:
            raise RuntimeError(
                f"FFmpeg executable was not found: {self.settings.ffmpeg_binary}"
            )

        version_id = str(artifact["version_id"])
        output_dir = self.root / "memory" / version_id
        frame_path = output_dir / "representative.png"
        audio_path = output_dir / "audio.wav"
        with self._lock:
            output_dir.mkdir(parents=True, exist_ok=True)
            if not frame_path.is_file():
                temporary = output_dir / f".representative.{uuid.uuid4().hex}.png"
                try:
                    subprocess.run(
                        [
                            binary,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            str(source),
                            "-vf",
                            "thumbnail",
                            "-frames:v",
                            "1",
                            str(temporary),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=self.settings.merge_timeout,
                    )
                    temporary.replace(frame_path)
                finally:
                    temporary.unlink(missing_ok=True)
            if not audio_path.is_file():
                temporary = output_dir / f".audio.{uuid.uuid4().hex}.wav"
                try:
                    subprocess.run(
                        [
                            binary,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            str(source),
                            "-vn",
                            "-ac",
                            "2",
                            "-ar",
                            "48000",
                            "-c:a",
                            "pcm_s16le",
                            str(temporary),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=self.settings.merge_timeout,
                    )
                    temporary.replace(audio_path)
                finally:
                    temporary.unlink(missing_ok=True)
        return str(frame_path.resolve()), str(audio_path.resolve())

    def materialize_r2v_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a durable payload understood by the local inference loader."""

        materialized = json.loads(_canonical_json(payload))
        condition_img = materialized.get("condition_img")
        if isinstance(condition_img, str) and condition_img.startswith("data:"):
            materialized["condition_img"] = self._write_inline(
                condition_img, kind="image"
            )

        slots: list[dict[str, Any]] = []
        for raw_slot in materialized.get("memory_slots", []):
            slot = dict(raw_slot)
            reference_shot_id = str(slot.get("shot_id") or "").strip()
            if reference_shot_id:
                image_path, audio_path = self._extract_reference_assets(
                    work_id=str(materialized["work_id"]),
                    shot_id=reference_shot_id,
                )
                metadata = dict(slot.get("metadata") or {})
                metadata.update(
                    {
                        "source_shot_id": reference_shot_id,
                        "resolved_by": "local_server",
                    }
                )
                slot = {
                    "image_url": image_path,
                    "audio_url": audio_path,
                    "metadata": metadata,
                }
            else:
                image_url = slot.get("image_url")
                audio_url = slot.get("audio_url")
                if isinstance(image_url, str) and image_url.startswith("data:"):
                    slot["image_url"] = self._write_inline(image_url, kind="image")
                if isinstance(audio_url, str) and audio_url.startswith("data:"):
                    slot["audio_url"] = self._write_inline(audio_url, kind="audio")
            slots.append(slot)
        materialized["memory_slots"] = slots
        return materialized


class MergeStore:
    """In-memory merge FIFO mirrored to the durable restart journal."""

    def __init__(
        self,
        capacity: int,
        *,
        journal: JobJournal | None = None,
        journal_kind: str = "merge",
    ) -> None:
        self.capacity = capacity
        self.journal = journal
        self.journal_kind = journal_kind
        self._jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def initialize(self) -> None:
        if self.journal is None:
            return
        self.journal.initialize()
        restored = self.journal.load(self.journal_kind)
        with self._lock:
            self._jobs.clear()
            for job in restored:
                if job.get("status") == "running":
                    job.update(status="queued", updated_at=_now(), error=None)
                    self.journal.save(self.journal_kind, job)
                self._jobs[str(job["version_id"])] = job
            self._prune_terminal()

    def _persist(self, job: dict[str, Any]) -> None:
        if self.journal is not None:
            self.journal.save(self.journal_kind, job)

    def _prune_terminal(self) -> None:
        while len(self._jobs) > self.capacity * 2:
            terminal_id = next(
                (
                    version_id
                    for version_id, job in self._jobs.items()
                    if job["status"] in {"succeeded", "failed"}
                    and (
                        not job.get("callback_url")
                        or job.get("callback_status") == "delivered"
                    )
                ),
                None,
            )
            if terminal_id is None:
                return
            self._jobs.pop(terminal_id, None)

    def enqueue(
        self,
        version_id: str,
        request: MergeRequest,
        *,
        callback_url: str | None,
        public_base_url: str,
    ) -> str:
        now = _now()
        payload = request.model_dump(
            exclude_none=True,
            exclude={"callback_url", "callback_context", "job_id"},
        )
        payload["public_base_url"] = public_base_url
        request_json = _canonical_json(payload)
        with self._lock:
            if request.job_id:
                existing = next(
                    (
                        job
                        for job in self._jobs.values()
                        if job.get("agent_job_id") == request.job_id
                    ),
                    None,
                )
                if existing is None and self.journal is not None:
                    existing = self.journal.get_by_agent_job_id(
                        self.journal_kind, request.job_id
                    )
                if existing is not None:
                    if existing["request_json"] != request_json:
                        raise ValueError(
                            "job_id is already associated with a different merge request"
                        )
                    existing.update(
                        callback_url=callback_url or existing.get("callback_url"),
                        callback_context_json=json.dumps(
                            request.callback_context, ensure_ascii=False
                        ),
                        updated_at=now,
                    )
                    self._persist(existing)
                    return str(existing["version_id"])
            pending = sum(
                job["status"] in {"queued", "running"} for job in self._jobs.values()
            )
            if pending >= self.capacity:
                raise OverflowError("merge queue is full")
            self._jobs[version_id] = {
                "version_id": version_id,
                "work_id": request.work_id,
                "agent_job_id": request.job_id,
                "request_json": request_json,
                "callback_url": callback_url,
                "callback_context_json": json.dumps(
                    request.callback_context, ensure_ascii=False
                ),
                "callback_status": None,
                "callback_error": None,
                "callback_attempts": 0,
                "callback_next_at": None,
                "status": "queued",
                "video_url": None,
                "output_path": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "completed_at": None,
            }
            self._persist(self._jobs[version_id])
            self._prune_terminal()
            return version_id

    def claim_next(self) -> dict[str, Any] | None:
        with self._lock:
            row = next(
                (job for job in self._jobs.values() if job["status"] == "queued"),
                None,
            )
            if row is None:
                return None
            now = _now()
            row.update(
                status="running",
                started_at=row["started_at"] or now,
                updated_at=now,
                error=None,
            )
            self._persist(row)
            return dict(row)

    def complete(self, version_id: str, *, video_url: str, output_path: Path) -> None:
        now = _now()
        with self._lock:
            self._jobs[version_id].update(
                status="succeeded",
                video_url=video_url,
                output_path=str(output_path),
                updated_at=now,
                completed_at=now,
                error=None,
            )
            self._persist(self._jobs[version_id])
            self._prune_terminal()

    def fail(self, version_id: str, error: str) -> None:
        now = _now()
        with self._lock:
            self._jobs[version_id].update(
                status="failed",
                error=error[:4000],
                updated_at=now,
                completed_at=now,
            )
            self._persist(self._jobs[version_id])
            self._prune_terminal()

    def get(self, version_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(version_id)
            if job is not None:
                return dict(job)
        if self.journal is not None:
            return self.journal.get(self.journal_kind, version_id)
        return None

    def callback_candidates(self, max_attempts: int) -> list[dict[str, Any]]:
        now = _now()
        with self._lock:
            return [
                dict(job)
                for job in self._jobs.values()
                if job["status"] in {"succeeded", "failed"}
                and job["callback_url"]
                and job["callback_status"] != "delivered"
                and (max_attempts == 0 or job["callback_attempts"] < max_attempts)
                and (job["callback_next_at"] is None or job["callback_next_at"] <= now)
            ]

    def record_callback(self, version_id: str, *, delivered: bool, error: str = "") -> None:
        with self._lock:
            job = self._jobs[version_id]
            attempts = int(job["callback_attempts"])
            job.update(
                callback_status="delivered" if delivered else "error",
                callback_error=None if delivered else error[:1000] or None,
                callback_attempts=attempts + 1,
                callback_next_at=None
                if delivered
                else _callback_retry_at(attempts),
                updated_at=_now(),
            )
            self._persist(job)

    def counts(self) -> dict[str, int]:
        result = {name: 0 for name in ("queued", "running", "succeeded", "failed")}
        with self._lock:
            for job in self._jobs.values():
                result[str(job["status"])] += 1
        return result


class R2VQueueService:
    """In-memory local inference queue with one staged runtime per GPU."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.journal = JobJournal(settings.artifact_db_path)
        self.store = JobStore(settings.queue_capacity, journal=self.journal)
        self.artifacts = ArtifactStore(settings.artifact_db_path)
        self.assets = RequestAssetResolver(settings, self.artifacts)
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.runtimes: dict[int, Any] = {}
        self.worker_states: dict[int, dict[str, Any]] = {}
        self._state_lock = threading.Lock()
        self._conditioning_lock = threading.Lock()

    def start(self) -> None:
        self.artifacts.initialize()
        self.store.initialize()
        self.threads = [
            threading.Thread(target=self._callback_loop, name="r2v-callback", daemon=True)
        ]
        if not self.settings.disable_inference_workers:
            from .runtime import LocalModelRuntime, resolve_gpu_ids

            if not self.settings.inference_config.is_file():
                raise FileNotFoundError(
                    f"inference config not found: {self.settings.inference_config}"
                )
            requests_root = self.settings.media_root / "requests"
            output_root = self.settings.media_root / "r2v"
            for gpu_id in resolve_gpu_ids(self.settings.gpu_ids):
                runtime = LocalModelRuntime(
                    gpu_id=gpu_id,
                    config_path=self.settings.inference_config,
                    checkpoint=self.settings.checkpoint,
                    conditioning_cache_dir=self.settings.conditioning_cache_dir,
                    requests_root=requests_root,
                    output_root=output_root,
                    dit_residency=self.settings.dit_residency,
                    gpu_headroom_fraction=self.settings.gpu_headroom_fraction,
                    ram_headroom_fraction=self.settings.ram_headroom_fraction,
                )
                self.runtimes[gpu_id] = runtime
                self.worker_states[gpu_id] = {
                    "gpu_id": gpu_id,
                    "state": "starting",
                    "current_task": None,
                    "model_loaded": False,
                    "weights_loaded": False,
                    "model_location": "unloaded",
                    "resources": None,
                    "error": None,
                }
                self.threads.append(
                    threading.Thread(
                        target=self._worker_loop,
                        args=(gpu_id,),
                        name=f"r2v-gpu-{gpu_id}",
                        daemon=True,
                    )
                )
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.wake_event.set()
        for thread in self.threads:
            thread.join(timeout=5)

    def enqueue(
        self,
        request: R2VGenerateRequest,
        payload: dict[str, Any],
        callback_url: str | None,
    ) -> str:
        version_id = str(uuid.uuid4())
        version_id = self.store.enqueue(version_id, request, payload, callback_url)
        self.wake_event.set()
        return version_id

    def materialize_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.assets.materialize_r2v_payload(payload)

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            workers = [dict(value) for _, value in sorted(self.worker_states.items())]
        return {
            "enabled": not self.settings.disable_inference_workers,
            "workers": workers,
            "queue_backend": "memory",
            "recovery_backend": "sqlite",
            "precision_config": str(self.settings.inference_config),
            "dit_residency": self.settings.dit_residency,
            "model_idle_seconds": self.settings.model_idle_seconds,
            "gpu_headroom_fraction": self.settings.gpu_headroom_fraction,
            "ram_headroom_fraction": self.settings.ram_headroom_fraction,
        }

    def _set_worker_state(self, gpu_id: int, **updates: Any) -> None:
        with self._state_lock:
            state = self.worker_states.setdefault(gpu_id, {"gpu_id": gpu_id})
            state.update(updates)

    @staticmethod
    def _weights_loaded(runtime: Any) -> bool:
        return bool(getattr(runtime, "weights_loaded", runtime.model_loaded))

    def _resource_ready(self, gpu_id: int) -> tuple[bool, dict[str, Any]]:
        from .runtime import probe_resources

        snapshot = probe_resources(gpu_id)
        payload = snapshot.as_dict()
        gpu_headroom = int(snapshot.gpu_total_bytes * self.settings.gpu_headroom_fraction)
        ram_headroom = int(snapshot.ram_total_bytes * self.settings.ram_headroom_fraction)
        runtime = self.runtimes[gpu_id]
        requirements = getattr(runtime, "admission_requirements", None)
        if requirements is not None:
            required_gpu, required_ram, plan = requirements(snapshot)
        else:
            required_gpu, required_ram, plan = gpu_headroom, ram_headroom, {
                "mode": "headroom_only"
            }
        ready = (
            snapshot.gpu_free_bytes >= required_gpu
            and snapshot.ram_available_bytes >= required_ram
        )
        payload["gpu_headroom_gib"] = round(gpu_headroom / 2**30, 3)
        payload["ram_headroom_gib"] = round(ram_headroom / 2**30, 3)
        payload["required_gpu_free_gib"] = round(required_gpu / 2**30, 3)
        payload["required_ram_available_gib"] = round(required_ram / 2**30, 3)
        payload["admission_plan"] = plan
        payload["ready"] = ready
        return ready, payload

    def _worker_loop(self, gpu_id: int) -> None:
        runtime = self.runtimes[gpu_id]
        while not self.stop_event.is_set():
            if self.store.counts()["queued"] == 0:
                if (
                    self._weights_loaded(runtime)
                    and self.settings.model_idle_seconds > 0
                    and time.monotonic() - runtime.last_used_at >= self.settings.model_idle_seconds
                ):
                    runtime.unload()
                self._set_worker_state(
                    gpu_id,
                    state=runtime.state if self._weights_loaded(runtime) else "idle",
                    current_task=None,
                    model_loaded=runtime.model_loaded,
                    weights_loaded=self._weights_loaded(runtime),
                    model_location=getattr(runtime, "model_location", "unloaded"),
                    error=None,
                )
                self.wake_event.wait(timeout=0.5)
                self.wake_event.clear()
                continue
            try:
                ready, resources = self._resource_ready(gpu_id)
            except Exception as exc:  # noqa: BLE001
                self._set_worker_state(gpu_id, state="resource_error", error=str(exc))
                self.stop_event.wait(self.settings.poll_interval)
                continue
            if not ready:
                if self._weights_loaded(runtime) and not runtime.model_loaded:
                    # Decoder-only leftovers cannot make forward progress when
                    # a cold generator load does not fit. Drop them and re-probe.
                    runtime.unload()
                elif self._weights_loaded(runtime) and (
                    resources["ram_available_gib"] < resources["ram_headroom_gib"]
                ):
                    runtime.unload()
                self._set_worker_state(
                    gpu_id,
                    state="waiting_resources",
                    current_task=None,
                    model_loaded=runtime.model_loaded,
                    weights_loaded=self._weights_loaded(runtime),
                    model_location=getattr(runtime, "model_location", "unloaded"),
                    resources=resources,
                    error=None,
                )
                self.stop_event.wait(self.settings.poll_interval)
                continue
            job = self.store.claim_next()
            if job is None:
                continue
            self._run_job(gpu_id, job, resources)
        runtime.unload()

    def _request_file(self, payload: dict[str, Any]) -> Path:
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        path = self.settings.media_root / "requests" / f"{digest}.json"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temporary.write_text(canonical + "\n", encoding="utf-8")
            try:
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return path

    def _fail_job(self, gpu_id: int, job: dict[str, Any], error: str) -> None:
        runtime = self.runtimes[gpu_id]
        version_id = str(job["version_id"])
        self.store.fail(version_id, error)
        self._set_worker_state(
            gpu_id,
            state="failed",
            current_task=version_id,
            model_loaded=runtime.model_loaded,
            weights_loaded=self._weights_loaded(runtime),
            model_location=getattr(runtime, "model_location", "unloaded"),
            error=error,
        )
        failed = self.store.get(version_id)
        if failed is not None:
            self.deliver_callback(failed)

    def _run_job(
        self,
        gpu_id: int,
        job: dict[str, Any],
        resources: dict[str, Any],
    ) -> None:
        runtime = self.runtimes[gpu_id]
        version_id = str(job["version_id"])
        try:
            self.store.update_stage(
                version_id, "validating", gpu_id=gpu_id, resources=resources
            )
            payload = _json_loads(job["request_json"], {})
            request_file = self._request_file(payload)
            request = runtime.load_request(request_file)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._fail_job(gpu_id, job, f"{type(exc).__name__}: {exc}")
            return

        def update_condition_stage(stage: str) -> None:
            try:
                _, current_resources = self._resource_ready(gpu_id)
            except Exception:  # noqa: BLE001
                current_resources = resources
            self.store.update_stage(
                version_id,
                stage,
                gpu_id=gpu_id,
                resources=current_resources,
            )
            self._set_worker_state(
                gpu_id,
                state=stage,
                current_task=version_id,
                model_loaded=runtime.model_loaded,
                weights_loaded=self._weights_loaded(runtime),
                model_location=getattr(runtime, "model_location", "unloaded"),
                resources=current_resources,
                error=None,
            )

        try:
            cache_loader = getattr(runtime, "load_cached_conditions", None)
            bundles = (
                cache_loader([request_file], [request], update_condition_stage)
                if cache_loader
                else None
            )
            if bundles is None:
                # Gemma is the largest transient conditioning component. Only
                # cache misses are serialized; cache hits remain parallel.
                with self._conditioning_lock:
                    from .runtime import probe_resources

                    snapshot = probe_resources(gpu_id)
                    policy_selector = getattr(
                        runtime, "conditioning_generator_policy", None
                    )
                    generator_policy = (
                        str(policy_selector(snapshot))
                        if policy_selector is not None
                        else "release"
                    )
                    bundles = runtime.prepare_conditions(
                        [request_file],
                        [request],
                        update_condition_stage,
                        generator_policy=generator_policy,
                    )
            bundle = bundles[request_file]
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            if "out of memory" in error.lower():
                runtime.unload()
            traceback.print_exc()
            self._fail_job(gpu_id, job, error)
            return

        self.store.update_stage(
            version_id, "ready", gpu_id=gpu_id, resources=resources
        )
        self._run_prepared_job(
            gpu_id,
            job,
            request_file,
            request,
            bundle,
            resources,
        )

    def _run_prepared_job(
        self,
        gpu_id: int,
        job: dict[str, Any],
        request_file: Path,
        request: Any,
        bundle: Any,
        resources: dict[str, Any],
    ) -> None:
        runtime = self.runtimes[gpu_id]
        version_id = str(job["version_id"])

        def update_stage(stage: str) -> None:
            try:
                _, current_resources = self._resource_ready(gpu_id)
            except Exception:  # noqa: BLE001
                current_resources = resources
            self.store.update_stage(
                version_id, stage, gpu_id=gpu_id, resources=current_resources
            )
            self._set_worker_state(
                gpu_id,
                state=stage,
                current_task=version_id,
                model_loaded=runtime.model_loaded,
                weights_loaded=self._weights_loaded(runtime),
                model_location=getattr(runtime, "model_location", "unloaded"),
                resources=current_resources,
                error=None,
            )

        try:
            update_stage("loading_generator")
            output_dir = (
                self.settings.media_root
                / "r2v"
                / request.work_id
                / request.shot_id
                / version_id
            )
            result = runtime.run(request_file, request, output_dir, bundle, update_stage)
            output_path = Path(result["output_path"]).resolve()
            relative = output_path.relative_to(self.settings.media_root.expanduser().resolve())
            video_url = f"{self.settings.public_base_url}/media/{quote(relative.as_posix())}"
            artifact = self.artifacts.upsert(
                version_id=version_id,
                work_id=str(job["work_id"]),
                shot_id=request.shot_id,
                kind="r2v",
                role="primary",
                url=video_url,
                local_path=str(output_path),
                size_bytes=output_path.stat().st_size,
                sha256=_file_sha256(output_path),
                metadata=result["metadata"],
            )
            local_result = {
                "video_id": artifact.get("artifact_id"),
                "video_url": video_url,
                "artifact_url": video_url,
                "output_path": str(output_path),
                "metadata": result["metadata"],
            }
            self.store.complete(version_id, local_result)
            completed = self.store.get(version_id)
            if completed is not None:
                self.deliver_callback(completed)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            if "out of memory" in error.lower():
                runtime.unload()
            elif runtime.model_loaded:
                runtime.state = "ready"
            traceback.print_exc()
            self._fail_job(gpu_id, job, error)
        finally:
            self._set_worker_state(
                gpu_id,
                state=runtime.state if self._weights_loaded(runtime) else "idle",
                current_task=None,
                model_loaded=runtime.model_loaded,
                weights_loaded=self._weights_loaded(runtime),
                model_location=getattr(runtime, "model_location", "unloaded"),
            )

    def deliver_callback(self, job: dict[str, Any]) -> None:
        callback_url = job.get("callback_url")
        if not callback_url or job.get("callback_status") == "delivered":
            return
        result = _json_loads(job.get("result_json"), {})
        completed = job["status"] == "succeeded"
        context = _json_loads(job.get("callback_context_json"), {})
        body: dict[str, Any] = {
            key: value
            for key, value in context.items()
            if key in {"session_key", "channel", "chat_id"} and value is not None
        }
        body.update(
            {
                "work_id": job["work_id"],
                "job_id": job.get("agent_job_id"),
                "status": "completed" if completed else "failed",
                "shot_id": job["shot_id"],
                "remote_task_id": job["version_id"],
            }
        )
        if completed:
            body.update(
                {
                    "video_id": result.get("video_id"),
                    "result_url": result.get("video_url"),
                    "artifact_url": result.get("video_url"),
                }
            )
            body = {key: value for key, value in body.items() if value is not None}
        else:
            body["error"] = job.get("error") or "generation failed"
        try:
            with httpx.Client(timeout=self.settings.request_timeout) as client:
                response = client.post(
                    callback_url,
                    json=body,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                )
            response.raise_for_status()
            self.store.record_callback(
                job["version_id"], delivered=True, response=getattr(response, "text", "")
            )
        except httpx.HTTPError as exc:
            self.store.record_callback(job["version_id"], delivered=False, error=str(exc))

    def _callback_loop(self) -> None:
        while not self.stop_event.wait(self.settings.poll_interval):
            for job in self.store.callback_candidates(self.settings.callback_attempts):
                if self.stop_event.is_set():
                    return
                self.deliver_callback(job)


class MergeQueueService:
    """Download ordered server-side artifacts and assemble them with FFmpeg."""

    def __init__(self, settings: Settings, r2v_service: R2VQueueService) -> None:
        self.settings = settings
        self.r2v_service = r2v_service
        self.store = MergeStore(
            settings.queue_capacity,
            journal=r2v_service.journal,
        )
        self.artifacts = r2v_service.artifacts
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.settings.media_root.expanduser().resolve().mkdir(parents=True, exist_ok=True)
        self.store.initialize()
        self.thread = threading.Thread(target=self._run_loop, name="video-merge", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.wake_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def enqueue(
        self,
        request: MergeRequest,
        *,
        callback_url: str | None,
        public_base_url: str,
    ) -> str:
        version_id = str(uuid.uuid4())
        version_id = self.store.enqueue(
            version_id,
            request,
            callback_url=callback_url,
            public_base_url=public_base_url,
        )
        self.wake_event.set()
        return version_id

    def _source_url(self, shot: dict[str, Any]) -> str:
        version_id = str(shot.get("version_id") or "").strip()
        if version_id:
            source_job = self.r2v_service.store.get(version_id)
            if source_job is None:
                raise RuntimeError(f"R2V version not found: {version_id}")
            if source_job.get("status") != "succeeded":
                raise RuntimeError(f"R2V version is not ready: {version_id}")
            result = _json_loads(source_job.get("result_json"), {})
            url = result.get("video_url")
        else:
            url = shot.get("video_url")
        if not isinstance(url, str) or not url.strip():
            raise RuntimeError("merge input does not resolve to a video URL")
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("merge video_url must be an absolute HTTP(S) URL")
        return url.strip()

    def _download(self, url: str, target: Path) -> None:
        total = 0
        with httpx.stream(
            "GET",
            url,
            timeout=self.settings.request_timeout,
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            with target.open("wb") as output:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    total += len(chunk)
                    if total > self.settings.merge_max_input_bytes:
                        raise RuntimeError(
                            f"merge input exceeds {self.settings.merge_max_input_bytes} bytes"
                        )
                    output.write(chunk)
        if total == 0:
            raise RuntimeError(f"merge input is empty: {url}")

    def _run_ffmpeg(self, inputs: list[Path], output: Path, work_dir: Path) -> None:
        binary = shutil.which(self.settings.ffmpeg_binary)
        if binary is None:
            raise RuntimeError(
                f"FFmpeg executable was not found: {self.settings.ffmpeg_binary}"
            )
        concat_file = work_dir / "inputs.txt"
        concat_file.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in inputs),
            encoding="utf-8",
        )
        common = [
            binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
        ]
        copy_result = subprocess.run(
            [*common, "-c", "copy", "-movflags", "+faststart", str(output)],
            capture_output=True,
            text=True,
            timeout=self.settings.merge_timeout,
            check=False,
        )
        if copy_result.returncode == 0 and output.is_file() and output.stat().st_size > 0:
            return
        output.unlink(missing_ok=True)
        encode_result = subprocess.run(
            [
                *common,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=self.settings.merge_timeout,
            check=False,
        )
        if encode_result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            detail = (encode_result.stderr or copy_result.stderr or "unknown FFmpeg error")[-2000:]
            raise RuntimeError(f"FFmpeg merge failed: {detail}")

    def _process(self, job: dict[str, Any]) -> None:
        payload = _json_loads(job.get("request_json"), {})
        shots = payload.get("shots")
        if not isinstance(shots, list) or not shots:
            raise RuntimeError("merge request has no shots")
        media_root = self.settings.media_root.expanduser().resolve()
        output_dir = media_root / "merges"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{job['version_id']}.mp4"
        output.unlink(missing_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="merge-", dir=media_root))
        try:
            inputs: list[Path] = []
            for index, shot in enumerate(shots, start=1):
                if not isinstance(shot, dict):
                    raise RuntimeError(f"merge shot {index} is invalid")
                target = work_dir / f"shot-{index:04d}.mp4"
                self._download(self._source_url(shot), target)
                inputs.append(target)
            self._run_ffmpeg(inputs, output, work_dir)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        base = str(payload.get("public_base_url") or "").rstrip("/")
        video_url = f"{base}/media/merges/{quote(output.name)}"
        digest = _file_sha256(output)
        self.artifacts.upsert(
            version_id=str(job["version_id"]),
            work_id=str(job["work_id"]),
            kind="merge",
            role="merged",
            url=video_url,
            local_path=str(output),
            size_bytes=output.stat().st_size,
            sha256=digest,
            metadata={"input_count": len(inputs)},
        )
        self.store.complete(str(job["version_id"]), video_url=video_url, output_path=output)

    def deliver_callback(self, job: dict[str, Any]) -> None:
        callback_url = str(job.get("callback_url") or "").strip()
        if not callback_url or job.get("callback_status") == "delivered":
            return
        context = _json_loads(job.get("callback_context_json"), {})
        body: dict[str, Any] = {
            key: value
            for key, value in context.items()
            if key in {"session_key", "channel", "chat_id"} and value is not None
        }
        completed = job.get("status") == "succeeded"
        body.update(
            {
                "work_id": job["work_id"],
                "job_id": job.get("agent_job_id"),
                "remote_task_id": job["version_id"],
                "status": "completed" if completed else "failed",
            }
        )
        if completed:
            body["result"] = {
                "artifact_url": job.get("video_url"),
                "result_url": job.get("video_url"),
            }
        else:
            body["error"] = job.get("error") or "merge failed"
        body = {key: value for key, value in body.items() if value is not None}
        try:
            with httpx.Client(timeout=self.settings.request_timeout) as client:
                response = client.post(callback_url, json=body)
            response.raise_for_status()
            self.store.record_callback(str(job["version_id"]), delivered=True)
        except httpx.HTTPError as exc:
            self.store.record_callback(
                str(job["version_id"]), delivered=False, error=str(exc)
            )

    def _run_loop(self) -> None:
        while not self.stop_event.is_set():
            job = self.store.claim_next()
            if job is not None:
                try:
                    self._process(job)
                except (OSError, RuntimeError, subprocess.SubprocessError, httpx.HTTPError) as exc:
                    self.store.fail(str(job["version_id"]), str(exc))
                completed = self.store.get(str(job["version_id"]))
                if completed is not None:
                    self.deliver_callback(completed)
                continue
            for candidate in self.store.callback_candidates(self.settings.callback_attempts):
                if self.stop_event.is_set():
                    return
                self.deliver_callback(candidate)
            self.wake_event.wait(timeout=0.5)
            self.wake_event.clear()


def _validate_callback_url(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=422, detail="callback URL must be absolute HTTP(S)")
    return value


def _validate_local_resource(value: str, field_name: str, *, kind: str) -> None:
    """Reject ambiguous paths before they enter the local GPU queue."""

    if value.startswith("data:"):
        limit = MAX_IMAGE_BYTES if kind == "image" else MAX_AUDIO_BYTES
        try:
            _decode_data_url(value, kind=kind, max_bytes=limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"{field_name}: {exc}") from exc
        return
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        if not parsed.netloc:
            raise HTTPException(
                status_code=422, detail=f"{field_name} must be an absolute HTTP(S) URL"
            )
        return
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise HTTPException(
                status_code=422, detail=f"{field_name} does not support remote file URLs"
            )
        local_path = Path(unquote(parsed.path))
    elif parsed.scheme:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{field_name} must use a data URL, HTTP(S), file://, "
                "or an absolute local path"
            ),
        )
    else:
        local_path = Path(value).expanduser()
    if not local_path.is_absolute():
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be absolute when submitted through server.py",
        )
    if not local_path.is_file():
        raise HTTPException(status_code=422, detail=f"{field_name} does not exist")


def _validate_local_r2v_request(body: R2VGenerateRequest) -> None:
    if body.condition_img:
        _validate_local_resource(body.condition_img, "condition_img", kind="image")
    for index, slot in enumerate(body.memory_slots):
        if slot.shot_id:
            continue
        assert slot.image_url is not None
        _validate_local_resource(
            slot.image_url,
            f"memory_slots[{index}].image_url",
            kind="image",
        )
        if slot.audio_url:
            _validate_local_resource(
                slot.audio_url,
                f"memory_slots[{index}].audio_url",
                kind="audio",
            )


def _service(request: Request) -> R2VQueueService:
    service = getattr(request.app.state, "r2v_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="R2V queue is not ready")
    return service


def _merge_service(request: Request) -> MergeQueueService:
    service = getattr(request.app.state, "merge_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="merge queue is not ready")
    return service


def _status_url(request: Request, settings: Settings, version_id: str) -> str:
    base = settings.public_base_url or str(request.base_url).rstrip("/")
    return f"{base}/version/{quote(version_id, safe='')}"


def _public_job(request: Request, service: R2VQueueService, job: dict[str, Any]) -> dict[str, Any]:
    payload = _json_loads(job.get("request_json"), {})
    local_result = _json_loads(job.get("result_json"), {})
    resources = _json_loads(job.get("resource_json"), {})
    result = {
        "accepted": True,
        "kind": "r2v",
        "task_id": job["version_id"],
        "version_id": job["version_id"],
        "remote_task_id": job["version_id"],
        "work_id": job["work_id"],
        "job_id": job.get("agent_job_id"),
        "shot_id": job["shot_id"],
        "status": job["status"],
        "stage": job.get("stage") or job["status"],
        "queue_position": service.store.queue_position(job["version_id"]),
        "status_url": _status_url(request, service.settings, job["version_id"]),
        "gpu_id": job.get("gpu_id"),
        "resources": resources or None,
        "video_id": local_result.get("video_id"),
        "video_url": local_result.get("video_url"),
        "width": payload.get("width"),
        "height": payload.get("height"),
        "memory_slots": payload.get("memory_slots", []),
        "error": job.get("error"),
        "callback_status": job.get("callback_status"),
        "callback_error": job.get("callback_error"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "artifacts": service.artifacts.for_version(str(job["version_id"])),
    }
    if local_result:
        result["result"] = local_result
    return result


def _public_merge_job(
    request: Request,
    service: MergeQueueService,
    job: dict[str, Any],
) -> dict[str, Any]:
    return {
        "accepted": True,
        "kind": "merge",
        "task_id": job["version_id"],
        "version_id": job["version_id"],
        "work_id": job["work_id"],
        "job_id": job.get("agent_job_id"),
        "status": job["status"],
        "status_url": _status_url(request, service.settings, str(job["version_id"])),
        "video_url": job.get("video_url"),
        "error": job.get("error"),
        "callback_status": job.get("callback_status"),
        "callback_error": job.get("callback_error"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "artifacts": service.artifacts.for_version(str(job["version_id"])),
    }


def _parse_merge_request(
    body: dict[str, Any],
    callback_header: str | None,
) -> tuple[MergeRequest, str | None]:
    payload_source = body.get("payload")
    payload = dict(payload_source) if isinstance(payload_source, dict) else dict(body)
    job = body.get("job")
    if isinstance(job, dict) and not payload.get("job_id"):
        payload["job_id"] = job.get("job_id")
    callback = body.get("callback")
    if isinstance(callback, dict):
        payload["callback_context"] = {
            key: callback.get(key) for key in ("session_key", "channel", "chat_id")
        }
        payload["callback_url"] = callback_header or callback.get("url")
    elif callback_header:
        payload["callback_url"] = callback_header
    try:
        parsed = MergeRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return parsed, _validate_callback_url(parsed.callback_url)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    previous_exception_handler = loop.get_exception_handler()

    def handle_loop_exception(
        current_loop: asyncio.AbstractEventLoop,
        context: dict[str, Any],
    ) -> None:
        exc = context.get("exception")
        message = str(context.get("message") or "")
        if (
            isinstance(exc, ConnectionResetError)
            and getattr(exc, "winerror", None) == 10054
            and "_call_connection_lost" in message
        ):
            return
        if previous_exception_handler is not None:
            previous_exception_handler(current_loop, context)
        else:
            current_loop.default_exception_handler(context)

    loop.set_exception_handler(handle_loop_exception)
    settings = Settings.from_env()
    r2v_service = R2VQueueService(settings)
    merge_service = MergeQueueService(settings, r2v_service)
    r2v_service.start()
    merge_service.start()
    app.state.r2v_service = r2v_service
    app.state.merge_service = merge_service
    try:
        yield
    finally:
        merge_service.stop()
        r2v_service.stop()
        app.state.r2v_service = None
        app.state.merge_service = None
        loop.set_exception_handler(previous_exception_handler)


app = FastAPI(
    title="Echo 1.5 Server",
    version="1.5",
    description="R2V generation, video merge, status, and health APIs for Echo Director.",
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    service = _service(request)
    merge_service = _merge_service(request)
    inference = service.status()
    r2v_counts = service.store.counts()
    workers = inference["workers"]
    worker_count = len(workers)
    gpu_busy = sum(worker.get("current_task") is not None for worker in workers)
    return {
        "status": "ok",
        "inference": inference,
        "queue": r2v_counts,
        "queues": {
            "r2v": r2v_counts,
            "merge": merge_service.store.counts(),
        },
        "scheduler": {
            "queues": {
                "inference": {
                    "workers": worker_count,
                    "busy": gpu_busy,
                    "idle": max(worker_count - gpu_busy, 0),
                    "pending": r2v_counts["queued"],
                }
            },
            "gpu_workers": worker_count,
            "gpu_busy": gpu_busy,
            "gpu_idle": max(worker_count - gpu_busy, 0),
            "active_tasks": r2v_counts["queued"] + r2v_counts["running"],
            "total_queued": r2v_counts["queued"],
        },
        "ffmpeg_available": shutil.which(service.settings.ffmpeg_binary) is not None,
    }


@app.post("/r2v")
async def generate_r2v(
    body: R2VGenerateRequest,
    request: Request,
    callback_header: str | None = Header(
        default=None,
        alias="X-Nanobot-Director-Callback-Url",
    ),
) -> dict[str, Any]:
    service = _service(request)
    _validate_local_r2v_request(body)
    callback_url = _validate_callback_url(callback_header or body.callback_url)
    payload = body.model_dump(
        exclude_none=True,
        exclude={"callback_url", "callback_context", "job_id"},
    )
    try:
        payload = service.materialize_request(payload)
        version_id = service.enqueue(body, payload, callback_url)
    except LookupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        status_code = 409 if "job_id is already associated" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except OverflowError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    job = service.store.get(version_id)
    assert job is not None
    return _public_job(request, service, job)


@app.post("/merge")
async def merge_videos(
    body: dict[str, Any],
    request: Request,
    callback_header: str | None = Header(
        default=None,
        alias="X-Nanobot-Director-Callback-Url",
    ),
) -> dict[str, Any]:
    service = _merge_service(request)
    merge_request, callback_url = _parse_merge_request(body, callback_header)
    public_base_url = service.settings.public_base_url or str(request.base_url).rstrip("/")
    try:
        version_id = service.enqueue(
            merge_request,
            callback_url=callback_url,
            public_base_url=public_base_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OverflowError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    job = service.store.get(version_id)
    assert job is not None
    return _public_merge_job(request, service, job)


@app.get("/version/{version_id}")
async def get_version(
    version_id: str,
    request: Request,
) -> dict[str, Any]:
    service = _service(request)
    job = service.store.get(version_id)
    if job is not None:
        return _public_job(request, service, job)
    merge_service = _merge_service(request)
    merge_job = merge_service.store.get(version_id)
    if merge_job is not None:
        return _public_merge_job(request, merge_service, merge_job)
    raise HTTPException(status_code=404, detail="version not found")


@app.get("/artifact/{artifact_id}")
async def get_artifact_metadata(
    artifact_id: str,
    request: Request,
) -> dict[str, Any]:
    service = _service(request)
    artifact = service.artifacts.get(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return artifact


@app.get("/media/{asset_path:path}")
async def get_media(asset_path: str, request: Request) -> FileResponse:
    service = _service(request)
    root = service.settings.media_root.expanduser().resolve()
    candidate = (root / asset_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="media not found") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="media not found")
    return FileResponse(candidate, media_type="video/mp4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Echo 1.5 local server")
    parser.add_argument(
        "--config",
        required=True,
        help="Server YAML that references an inference YAML",
    )
    parser.add_argument("--host", help="Override server.host")
    parser.add_argument("--port", type=int, help="Override server.port")
    args = parser.parse_args()

    config_path = _repo_path(args.config)
    config = apply_server_config(config_path)
    server_config = _config_section(config, "server")
    host = args.host or str(server_config.get("host", "127.0.0.1"))
    port = args.port or int(server_config.get("port", 8221))
    workers = int(server_config.get("workers", 1))
    if workers != 1:
        parser.error("server.workers must be 1 because GPU workers live in process memory")

    os.environ["ECHO_SERVER_CONFIG"] = str(config_path)
    import uvicorn

    uvicorn.run("server.app:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
