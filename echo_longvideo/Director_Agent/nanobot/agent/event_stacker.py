"""Event Stacker — record agent events as a structured event stream."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


_STACKER_DIR = ".event_stacker"


class EventStacker:
    _instance: EventStacker | None = None
    _lock = threading.Lock()

    def __init__(self, workspace: Path, *, max_traces: int = 100) -> None:
        self._workspace = workspace
        self._max_traces = max_traces
        self._dir = workspace / _STACKER_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._tls = threading.local()
        self._turn_counter = self._recover_max_turn_id()
        self._event_counter = 0

    def _recover_max_turn_id(self) -> int:
        max_id = 0
        try:
            for f in self._dir.glob("*.jsonl"):
                with open(f, "rb") as fh:
                    fh.seek(max(0, f.stat().st_size - 8192))
                    tail = fh.read().decode("utf-8", errors="replace")
                for line in reversed(tail.strip().split("\n")):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        tid = rec.get("turn_id", "")
                        if tid.startswith("turn_"):
                            num = int(tid.split("_", 1)[1])
                            max_id = max(max_id, num)
                            break
                    except (json.JSONDecodeError, ValueError):
                        continue
        except OSError:
            pass
        return max_id

    @classmethod
    def init(cls, workspace: Path, *, enabled: bool = False, max_traces: int = 100) -> None:
        with cls._lock:
            cls._instance = cls(workspace, max_traces=max_traces) if enabled else None

    @classmethod
    def begin_turn(cls, session_key: str | None, model: str | None) -> None:
        inst = cls._instance
        if inst is None:
            return
        key = session_key or "unknown"
        inst._tls.session_key = key
        with inst._lock:
            inst._turn_counter += 1
            inst._tls.turn_id = f"turn_{inst._turn_counter:04d}"
        inst._tls.phase = "turn_context"
        cls.emit("turn_start", {"model": model or ""})

    @classmethod
    def set_phase(cls, phase: str | None) -> None:
        inst = cls._instance
        if inst is None:
            return
        inst._tls.phase = phase

    @classmethod
    @contextmanager
    def phase(cls, name: str) -> Generator[None, None, None]:
        """Context manager: set phase for the duration, restore previous on exit."""
        inst = cls._instance
        if inst is None:
            yield
            return
        prev = getattr(inst._tls, "phase", None)
        inst._tls.phase = name
        try:
            yield
        finally:
            inst._tls.phase = prev

    @classmethod
    def log(cls, label: str, content: Any) -> None:
        """Shorthand for emit('context_part', ...) — mirrors PromptStacker.log API."""
        inst = cls._instance
        if inst is None:
            return
        current_phase = getattr(inst._tls, "phase", None)
        if current_phase is None:
            return
        text = content if isinstance(content, str) else json.dumps(
            content, ensure_ascii=False, default=str,
        )
        cls.emit("context_part", {
            "label": label,
            "phase": current_phase,
            "content": text,
            "char_count": len(text),
        })

    @classmethod
    def emit(cls, event_type: str, data: dict[str, Any] | None = None) -> None:
        inst = cls._instance
        if inst is None:
            return
        key = getattr(inst._tls, "session_key", None)
        if key is None:
            return
        turn_id = getattr(inst._tls, "turn_id", None)

        with inst._lock:
            inst._event_counter += 1
            seq = inst._event_counter

        record = {
            "seq": seq,
            "type": event_type,
            "turn_id": turn_id or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_key": key,
            "data": data or {},
        }

        safe_key = key.replace(":", "_").replace("/", "_")
        trace_file = inst._dir / f"{safe_key}.jsonl"
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @classmethod
    def end_turn(
        cls,
        stop_reason: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        cls.emit("turn_end", {
            "stop_reason": stop_reason or "",
            "usage": usage or {},
        })
        inst = cls._instance
        if inst is not None:
            inst._tls.phase = None
            inst._gc_traces()

    def _gc_traces(self) -> None:
        try:
            traces = sorted(self._dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
            while len(traces) > self._max_traces:
                traces.pop(0).unlink(missing_ok=True)
        except OSError:
            pass

    @classmethod
    def get_sessions(cls) -> list[dict[str, Any]]:
        inst = cls._instance
        if inst is None:
            return []
        results = []
        for f in sorted(inst._dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            stat = f.stat()
            event_count = 0
            last_record: dict[str, Any] = {}
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            event_count += 1
                            try:
                                last_record = json.loads(line)
                            except json.JSONDecodeError:
                                pass
            except OSError:
                pass
            results.append({
                "id": f.stem,
                "filename": f.name,
                "event_count": event_count,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "last_turn_id": last_record.get("turn_id", ""),
                "last_session_key": last_record.get("session_key", ""),
            })
        return results

    @classmethod
    def get_trace(cls, session_id: str) -> list[dict[str, Any]]:
        inst = cls._instance
        if inst is None:
            return []
        trace_file = inst._dir / f"{session_id}.jsonl"
        if not trace_file.is_file():
            return []
        try:
            trace_file.relative_to(inst._dir)
        except ValueError:
            return []
        records = []
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records
