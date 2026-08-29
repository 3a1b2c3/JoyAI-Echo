"""Echo generator admission / traffic control.

Before submitting generation jobs, poll ``{echoGenerator.baseUrl}/health`` and
reject new work when the algorithm side is overloaded:

    workers  = scheduler.queues.inference.workers
    works    = scheduler.gpu_busy + scheduler.total_queued
    capacity = workers
    reject when works >= capacity
"""

from __future__ import annotations

import errno
import json
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from loguru import logger

from nanobot.config.schema import EchoGeneratorConfig

BUSY_MESSAGE = "The video generation service is busy. Please try again shortly."
UNAVAILABLE_MESSAGE = "The video generation service is temporarily unavailable. Please try again shortly."
_GENERATION_OPERATIONS = frozenset({"generate_echo_shot", "r2v_generate"})


class EchoGeneratorBusyError(RuntimeError):
    """Raised when the Echo generator should not accept more generation work."""

    def __init__(self, message: str = BUSY_MESSAGE, *, snapshot: "AdmissionSnapshot | None" = None):
        super().__init__(message)
        self.snapshot = snapshot


class EchoGeneratorUnavailableError(RuntimeError):
    """Raised when the Echo generator host cannot be reached."""

    def __init__(self, message: str = UNAVAILABLE_MESSAGE):
        super().__init__(message)


def is_connection_refused(exc: BaseException) -> bool:
    current: BaseException | None = exc
    for _ in range(5):
        if current is None:
            break
        if isinstance(current, ConnectionRefusedError):
            return True
        if getattr(current, "errno", None) == errno.ECONNREFUSED:
            return True
        reason = getattr(current, "reason", None)
        if isinstance(reason, BaseException) and reason is not current:
            current = reason
            continue
        current = current.__cause__ or current.__context__
    text = str(exc).lower()
    return "connection refused" in text or "errno 111" in text


@dataclass(frozen=True)
class AdmissionSnapshot:
    """Parsed health signals used for the admission decision."""

    workers: float | None
    capacity: float | None
    works: float | None
    busy: bool
    reason: str
    raw: dict[str, Any] | None = None


class EchoAdmissionController:
    """Traffic gate for Echo / JoyEcho generation endpoints."""

    def __init__(
        self,
        config: EchoGeneratorConfig | None = None,
        *,
        base_url: str | None = None,
        timeout_sec: float | None = None,
        fail_open: bool = True,
    ) -> None:
        cfg = config or EchoGeneratorConfig()
        resolved_base = (base_url if base_url is not None else cfg.base_url) or ""
        self.base_url = str(resolved_base).strip().rstrip("/")
        raw_timeout = timeout_sec if timeout_sec is not None else cfg.http_timeout_sec
        try:
            self.timeout_sec = max(1.0, float(raw_timeout))
        except (TypeError, ValueError):
            self.timeout_sec = 30.0
        self.fail_open = fail_open

    @classmethod
    def from_tools_config(cls, tools_config: Any, **kwargs: Any) -> "EchoAdmissionController":
        return cls(
            getattr(tools_config, "echo_generator", None) or EchoGeneratorConfig(),
            **kwargs,
        )

    def applies_to_operation(self, operation: str | None) -> bool:
        return (operation or "") in _GENERATION_OPERATIONS

    def fetch_health(self) -> dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("echoGenerator.baseUrl is not configured")
        url = f"{self.base_url}/health"
        req = urllib_request.Request(url, method="GET")
        with urllib_request.urlopen(req, timeout=self.timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected /health payload type: {type(data).__name__}")
        return data

    @staticmethod
    def _as_number(value: Any) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    @classmethod
    def _dig(cls, data: dict[str, Any], *path: str) -> Any:
        cur: Any = data
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    @classmethod
    def extract_signals(cls, health: dict[str, Any]) -> tuple[float | None, float | None]:
        """Resolve workers / works from /health.

        - workers = ``scheduler.queues.inference.workers``
        - works   = ``scheduler.gpu_busy`` + ``scheduler.total_queued``
        """
        workers = cls._as_number(
            cls._dig(health, "scheduler", "queues", "inference", "workers")
        )
        gpu_busy = cls._as_number(cls._dig(health, "scheduler", "gpu_busy"))
        total_queued = cls._as_number(cls._dig(health, "scheduler", "total_queued"))
        if gpu_busy is None or total_queued is None:
            works = None
        else:
            works = gpu_busy + total_queued
        return workers, works

    def evaluate(self, health: dict[str, Any]) -> AdmissionSnapshot:
        if str(health.get("status") or "").strip().lower() not in {"", "ok"}:
            return AdmissionSnapshot(
                workers=None,
                capacity=None,
                works=None,
                busy=True,
                reason="status_unhealthy",
                raw=health,
            )

        workers, works = self.extract_signals(health)
        if workers is None or works is None:
            return AdmissionSnapshot(
                workers=workers,
                capacity=None,
                works=works,
                busy=False,
                reason="missing_workers_or_works",
                raw=health,
            )

        capacity = workers
        busy = works >= capacity
        return AdmissionSnapshot(
            workers=workers,
            capacity=capacity,
            works=works,
            busy=busy,
            reason="at_or_over_capacity" if busy else "within_capacity",
            raw=health,
        )

    def check(self) -> AdmissionSnapshot:
        """Fetch health and decide. Raises ``EchoGeneratorBusyError`` when overloaded."""
        if not self.base_url:
            return AdmissionSnapshot(
                workers=None,
                capacity=None,
                works=None,
                busy=False,
                reason="base_url_not_configured",
            )
        try:
            health = self.fetch_health()
        except (
            urllib_error.URLError,
            urllib_error.HTTPError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            logger.warning(
                "Echo admission health check failed (fail_open={}): {}",
                self.fail_open,
                exc,
            )
            if is_connection_refused(exc):
                raise EchoGeneratorUnavailableError(UNAVAILABLE_MESSAGE) from exc
            if self.fail_open:
                return AdmissionSnapshot(
                    workers=None,
                    capacity=None,
                    works=None,
                    busy=False,
                    reason=f"health_check_failed:{exc}",
                )
            raise EchoGeneratorBusyError(BUSY_MESSAGE) from exc

        snapshot = self.evaluate(health)
        logger.info(
            "Echo admission check workers={} capacity={} works={} busy={} reason={}",
            snapshot.workers,
            snapshot.capacity,
            snapshot.works,
            snapshot.busy,
            snapshot.reason,
        )
        if snapshot.busy:
            raise EchoGeneratorBusyError(BUSY_MESSAGE, snapshot=snapshot)
        return snapshot

    def ensure_allowed(self, *, operation: str | None = None) -> AdmissionSnapshot:
        """Gate generation operations; no-op for unrelated operations."""
        if operation is not None and not self.applies_to_operation(operation):
            return AdmissionSnapshot(
                workers=None,
                capacity=None,
                works=None,
                busy=False,
                reason=f"skip_operation:{operation}",
            )
        return self.check()
