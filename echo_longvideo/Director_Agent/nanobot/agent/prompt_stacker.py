"""Prompt Stacker — monitor and record prompt composition for each model call."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_STACKER_DIR = ".prompt_stacker"


@dataclass
class _SessionState:
    """Per-session recording state, isolated from other concurrent sessions."""

    base_parts: list[dict[str, Any]] = field(default_factory=list)
    iter_parts: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None
    phase: str = "context"


class PromptStacker:
    _instance: PromptStacker | None = None
    _lock = threading.Lock()

    def __init__(self, workspace: Path, *, max_traces: int = 100) -> None:
        self._workspace = workspace
        self._max_traces = max_traces
        self._dir = workspace / _STACKER_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, _SessionState] = {}
        self._turn_counter = self._recover_max_turn_id()
        self._tls = threading.local()

    @classmethod
    def init(cls, workspace: Path, *, enabled: bool = False, max_traces: int = 100) -> None:
        with cls._lock:
            if enabled:
                cls._instance = cls(workspace, max_traces=max_traces)
            else:
                cls._instance = None

    @classmethod
    def begin_turn(cls, session_key: str | None, model: str | None) -> None:
        """Called in loop.py BEFORE build_messages(). Resets state for this session's new turn."""
        inst = cls._instance
        if inst is None:
            return
        key = session_key or "unknown"
        inst._tls.session_key = key
        inst._sessions[key] = _SessionState(model=model, phase="context")

    @classmethod
    def log(cls, label: str, content: Any) -> None:
        """Log a prompt part for the current session (identified via thread-local)."""
        inst = cls._instance
        if inst is None:
            return
        key = getattr(inst._tls, "session_key", None)
        if key is None:
            return
        state = inst._sessions.get(key)
        if state is None:
            return
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
        entry = {
            "label": label,
            "content": text,
            "char_count": len(text),
        }
        if state.phase == "context":
            state.base_parts.append(entry)
        else:
            state.iter_parts.append(entry)

    @classmethod
    def begin_iteration(cls, iteration: int, model: str | None = None) -> None:
        """Called in runner.py at the start of each model call. Clears iter_parts only."""
        inst = cls._instance
        if inst is None:
            return
        key = getattr(inst._tls, "session_key", None)
        if key is None:
            return
        state = inst._sessions.get(key)
        if state is None:
            return
        state.iter_parts = []
        state.phase = "iteration"
        if model:
            state.model = model

    @classmethod
    def commit(
        cls,
        *,
        messages: list[dict[str, Any]] | None = None,
        response: Any = None,
        usage: dict[str, Any] | None = None,
        iteration: int = 0,
    ) -> None:
        """Write a complete record for one model call. Preserves base_parts for next iteration."""
        inst = cls._instance
        if inst is None:
            return

        key = getattr(inst._tls, "session_key", None)
        if key is None:
            return
        state = inst._sessions.get(key)
        if state is None:
            return

        with inst._lock:
            inst._turn_counter += 1
            turn_id = inst._turn_counter

        now = datetime.now(timezone.utc)
        safe_key = key.replace(":", "_").replace("/", "_")

        resp_data: dict[str, Any] = {}
        if response is not None:
            resp_data["content"] = getattr(response, "content", None) or ""
            tool_calls = getattr(response, "tool_calls", None)
            if tool_calls:
                resp_data["tool_calls"] = [
                    {"name": getattr(tc, "name", ""), "arguments": getattr(tc, "arguments", "")}
                    for tc in tool_calls
                ]
            reasoning = getattr(response, "reasoning_content", None)
            if reasoning:
                resp_data["reasoning_content"] = reasoning
        if usage:
            resp_data["usage"] = usage

        all_parts = list(state.base_parts) + list(state.iter_parts)

        record = {
            "id": f"turn_{turn_id:04d}",
            "timestamp": now.isoformat(),
            "session_key": key,
            "iteration": iteration,
            "model": state.model or "",
            "parts": all_parts,
            "messages_count": len(messages) if messages else 0,
            "messages": _full_messages(messages) if messages else [],
            "response": resp_data,
        }

        trace_file = inst._dir / f"{safe_key}.jsonl"
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        state.iter_parts = []
        inst._gc_traces()

    @classmethod
    def end_turn(cls, session_key: str | None = None) -> None:
        """Clean up session state after a turn completes. Called optionally."""
        inst = cls._instance
        if inst is None:
            return
        key = session_key or getattr(inst._tls, "session_key", None)
        if key:
            inst._sessions.pop(key, None)

    def _recover_max_turn_id(self) -> int:
        """Scan existing JSONL files to recover the highest turn counter."""
        max_id = 0
        try:
            for f in self._dir.glob("*.jsonl"):
                with open(f, "rb") as fh:
                    fh.seek(max(0, f.stat().st_size - 4096))
                    tail = fh.read().decode("utf-8", errors="replace")
                for line in reversed(tail.strip().split("\n")):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        tid = rec.get("id", "")
                        if tid.startswith("turn_"):
                            num = int(tid.split("_", 1)[1])
                            max_id = max(max_id, num)
                            break
                    except (json.JSONDecodeError, ValueError):
                        continue
        except OSError:
            pass
        return max_id

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
            last_line = ""
            try:
                with open(f, "rb") as fh:
                    fh.seek(max(0, stat.st_size - 4096))
                    last_line = fh.read().decode("utf-8", errors="replace").strip().rsplit("\n", 1)[-1]
            except OSError:
                pass
            turn_count = 0
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    turn_count = sum(1 for _ in fh)
            except OSError:
                pass
            last_record = {}
            if last_line:
                try:
                    last_record = json.loads(last_line)
                except json.JSONDecodeError:
                    pass
            results.append({
                "id": f.stem,
                "filename": f.name,
                "turn_count": turn_count,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "last_model": last_record.get("model", ""),
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


def _full_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str):
            full_text = content
            char_count = len(content)
        elif isinstance(content, list):
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            full_text = "\n".join(texts)
            char_count = len(full_text)
        else:
            full_text = str(content)
            char_count = len(full_text)

        preview = full_text[:200] + "..." if len(full_text) > 200 else full_text

        entry: dict[str, Any] = {
            "role": role,
            "preview": preview,
            "content": full_text,
            "char_count": char_count,
        }
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc.get("function", {}).get("name", ""), "id": tc.get("id", ""),
                 "arguments": tc.get("function", {}).get("arguments", "")}
                for tc in tool_calls
                if isinstance(tc, dict)
            ]
        if role == "tool":
            entry["tool_call_id"] = msg.get("tool_call_id", "")
            entry["tool_name"] = msg.get("name", "")
        result.append(entry)
    return result
