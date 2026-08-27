"""Director callback channel.

Receives remote director/Echo callbacks, applies workspace state changes, and
emits runtime/UI notifications without projecting the callback into the LLM
conversation.
"""

from __future__ import annotations

import asyncio
import hmac
import json
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from nanobot.agent.tools.director import (
    GenerateEchoShotTool,
    apply_echo_generate_shot_callback,
    apply_merge_shot_callback,
)
from nanobot.bus.events import OutboundMessage, RuntimeEvent
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base, ToolsConfig


class DirectorCallbackConfig(Base):
    """HTTP callback channel for remote director/Echo jobs."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 18791
    path_prefix: str = "/api/director"
    secret: str = ""


class DirectorCallbackChannel(BaseChannel):
    """Receive remote director callbacks as system-level runtime events."""

    name = "director_callback"
    display_name = "Director Callback"

    def __init__(
        self,
        config: Any,
        bus: MessageBus,
        *,
        workspace: Path | None = None,
        tools_config: ToolsConfig | None = None,
        memory_review_runner: Callable[..., Any] | None = None,
        memory_approval_runner: Callable[..., Any] | None = None,
        shot_generation_runner: Callable[..., Any] | None = None,
    ):
        if isinstance(config, dict):
            config = DirectorCallbackConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: DirectorCallbackConfig = config
        self.workspace = workspace or Path(".")
        self._tools_config = tools_config or ToolsConfig()
        self._runner: Any | None = None
        self._memory_review_lock = asyncio.Lock()
        if memory_review_runner is None:
            from nanobot.director.memory_coordinator import run_memory_review_from_config

            memory_review_runner = run_memory_review_from_config
        if memory_approval_runner is None:
            from nanobot.director.r2v_memory_workflow import (
                auto_approve_review_and_prepare_next,
            )

            memory_approval_runner = auto_approve_review_and_prepare_next
        self._memory_review_runner = memory_review_runner
        self._memory_approval_runner = memory_approval_runner
        self._shot_generation_runner = (
            shot_generation_runner or self._submit_memory_next_shot
        )

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return DirectorCallbackConfig().model_dump(by_alias=True)

    async def start(self) -> None:
        try:
            from aiohttp import web
        except ImportError:
            logger.error("director_callback requires aiohttp. Install with: pip install 'echo-director-agent[api]'")
            return

        self._running = True
        app = self.create_app()
        runner = web.AppRunner(app)
        self._runner = runner
        await runner.setup()
        site = web.TCPSite(runner, self.config.host, self.config.port)
        await site.start()
        logger.info(
            "Director callback channel listening on http://{}:{}{}",
            self.config.host,
            self.config.port,
            self._path_prefix(),
        )
        try:
            while self._running:
                await asyncio.sleep(1)
        finally:
            await runner.cleanup()
            self._runner = None

    async def stop(self) -> None:
        self._running = False
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception as exc:
                logger.warning("director_callback cleanup failed: {}", exc)
            self._runner = None

    async def send(self, msg: OutboundMessage) -> None:
        logger.debug("director_callback has no outbound delivery target: {}", msg.metadata)

    def create_app(self) -> Any:
        from aiohttp import web

        app = web.Application(client_max_size=20 * 1024 * 1024)
        prefix = self._path_prefix()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post(f"{prefix}/echo-generate-shot/callback", self._handle_echo_generate_callback)
        app.router.add_post(f"{prefix}/merge-shot/callback", self._handle_merge_callback)
        return app

    def _path_prefix(self) -> str:
        raw = (self.config.path_prefix or "/api/director").strip()
        if not raw.startswith("/"):
            raw = f"/{raw}"
        return raw.rstrip("/") or "/api/director"

    def _authorized(self, request: Any) -> bool:
        secret = (self.config.secret or "").strip()
        if not secret:
            return True
        supplied = (
            request.headers.get("X-Nanobot-Director-Secret")
            or request.headers.get("X-Nanobot-Auth")
            or ""
        ).strip()
        return bool(supplied) and hmac.compare_digest(supplied, secret)

    async def _parse_body(self, request: Any) -> dict[str, Any] | Any:
        from aiohttp import web

        if not self._authorized(request):
            return web.json_response({"error": "Unauthorized"}, status=401)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "Callback body must be a JSON object"}, status=400)
        return body

    async def _handle_health(self, request: Any) -> Any:
        from aiohttp import web

        return web.json_response({"status": "ok"})

    async def _handle_echo_generate_callback(self, request: Any) -> Any:
        return await self._handle_callback(
            request,
            operation="generate_echo_shot",
            apply_callback=apply_echo_generate_shot_callback,
            result_fields=("result_urls", "updated_shots"),
        )

    async def _handle_merge_callback(self, request: Any) -> Any:
        return await self._handle_callback(
            request,
            operation="merge_shot",
            apply_callback=apply_merge_shot_callback,
            result_fields=("final_output",),
        )

    async def _handle_callback(
        self,
        request: Any,
        *,
        operation: str,
        apply_callback: Callable[..., dict[str, Any]],
        result_fields: tuple[str, ...],
    ) -> Any:
        from aiohttp import web

        parsed = await self._parse_body(request)
        if isinstance(parsed, web.Response):
            return parsed

        try:
            result = apply_callback(
                self.workspace,
                parsed,
                tools_config=self._tools_config,
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception:
            logger.exception("director_callback: failed to apply {}", operation)
            return web.json_response({"error": f"Failed to apply {operation} callback"}, status=500)

        logger.info(
            "director_callback received operation={} work_id={} status={}",
            operation,
            result.get("work_id"),
            result.get("status"),
        )
        if result.get("duplicate"):
            return web.json_response(
                {
                    "status": "ok",
                    "operation": operation,
                    "job_id": result.get("job_id"),
                    "work_id": result.get("work_id"),
                    "duplicate": True,
                    "runtime_event": False,
                    "workplace_notified": False,
                }
            )
        self._schedule_memory_review(operation, parsed, result)
        await self._publish_runtime_event(operation, parsed, result)
        workplace_notified = await self._publish_workplace_updated(parsed, result)
        logger.info(
            "director_callback done operation={} workplace_notified={}",
            operation,
            workplace_notified,
        )

        response: dict[str, Any] = {
            "status": "ok",
            "operation": operation,
            "job_id": result.get("job_id"),
            "work_id": result.get("work_id"),
            "runtime_event": True,
            "workplace_notified": workplace_notified,
        }
        for field in result_fields:
            response[field] = result.get(field) or ([] if field.endswith("s") else None)
        return web.json_response(response)

    def _memory_review_enabled(self) -> bool:
        return self._tools_config.memory_review.enabled

    def _memory_review_auto_approve_enabled(self) -> bool:
        return self._tools_config.memory_review.auto_approve

    def _work_is_auto_generate(self, work_id: str) -> bool:
        if not work_id:
            return False
        state_path = self.workspace / "director" / "works" / work_id / "state.json"
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "director_callback failed to read auto_generate work_id={} error={}",
                work_id,
                exc,
            )
            return False
        return isinstance(data, dict) and bool(data.get("auto_generate"))

    def _schedule_memory_review(
        self, operation: str, body: dict[str, Any], result: dict[str, Any]
    ) -> None:
        if (
            operation != "generate_echo_shot"
            or str(result.get("status") or "") != "completed"
            or not self._memory_review_enabled()
        ):
            return
        updated = result.get("updated_shots")
        if not isinstance(updated, list) or not updated:
            return
        try:
            shot_id = int(updated[0]["shot_id"])
        except (KeyError, TypeError, ValueError):
            return
        work_id = str(result.get("work_id") or body.get("work_id") or "").strip()
        if work_id:
            auto_select = (
                self._work_is_auto_generate(work_id)
                or self._memory_review_auto_approve_enabled()
            )
            if not auto_select:
                try:
                    from nanobot.director.memory_coordinator import (
                        initialize_memory_review_method_prompt,
                    )

                    initialize_memory_review_method_prompt(
                        workspace=self.workspace,
                        work_id=work_id,
                        shot_id=shot_id,
                    )
                except Exception:
                    logger.exception(
                        "director_callback: failed to initialize memory method prompt "
                        "work_id={} shot_id={}",
                        work_id,
                        shot_id,
                    )
                return
            try:
                from nanobot.director.memory_coordinator import mark_memory_review_selecting

                mark_memory_review_selecting(
                    workspace=self.workspace,
                    work_id=work_id,
                    shot_id=shot_id,
                )
            except Exception:
                logger.exception(
                    "director_callback: failed to mark memory selecting "
                    "work_id={} shot_id={}",
                    work_id,
                    shot_id,
                )
            asyncio.create_task(
                self._run_memory_review(
                    body=body,
                    result=result,
                    work_id=work_id,
                    shot_id=shot_id,
                )
            )

    async def _run_memory_review(
        self,
        *,
        body: dict[str, Any],
        result: dict[str, Any],
        work_id: str,
        shot_id: int,
    ) -> None:
        try:
            async with self._memory_review_lock:
                review = await asyncio.to_thread(
                    self._memory_review_runner,
                    workspace=self.workspace,
                    work_id=work_id,
                    shot_id=shot_id,
                )
                if not isinstance(review, dict):
                    return
                auto_generate = self._work_is_auto_generate(work_id)
                if auto_generate or self._memory_review_auto_approve_enabled():
                    next_shot_id = await asyncio.to_thread(
                        self._memory_approval_runner,
                        workspace=self.workspace,
                        work_id=work_id,
                        shot_id=shot_id,
                    )
                    # Auto-generate lets the WebSocket workflow continue the next
                    # shot. Submitting here races generate_all.
                    if not auto_generate and next_shot_id is not None:
                        await asyncio.to_thread(
                            self._shot_generation_runner,
                            workspace=self.workspace,
                            work_id=work_id,
                            shot_id=int(next_shot_id),
                            body=body,
                            result=result,
                        )
        except Exception as exc:
            logger.exception(
                "director_callback: memory selection failed work_id={} shot_id={}",
                work_id,
                shot_id,
            )
            try:
                from nanobot.director.memory_coordinator import (
                    initialize_memory_review_method_prompt,
                )

                await asyncio.to_thread(
                    initialize_memory_review_method_prompt,
                    workspace=self.workspace,
                    work_id=work_id,
                    shot_id=shot_id,
                    error=f"Memory review failed: {exc}",
                )
            except Exception:
                logger.exception(
                    "director_callback: failed to persist memory selection error "
                    "work_id={} shot_id={}",
                    work_id,
                    shot_id,
                )
        finally:
            await self._publish_workplace_updated(body, result)

    def _submit_memory_next_shot(
        self,
        *,
        workspace: Path,
        work_id: str,
        shot_id: int,
        body: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Submit the Memory-prepared next shot unless it is already in flight."""
        shot_path = (
            workspace / "director" / "works" / work_id / "shots"
            / f"shot_{shot_id:03d}.json"
        )
        try:
            shot = json.loads(shot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"next shot {shot_id} is unavailable") from exc
        if not isinstance(shot, dict):
            raise ValueError(f"next shot {shot_id} is invalid")
        if str(shot.get("status") or "") in {
            "queued",
            "generated",
            "review_pass",
            "approved",
        }:
            return None

        references = shot.get("planned_reference_shot_ids") or []
        if not isinstance(references, list):
            references = []
        reference_ids = [int(value) for value in references]
        selection_note = str(shot.get("reference_selection_note") or "").strip() or None
        channel_name = str(result.get("channel") or body.get("channel") or "").strip()
        chat_id = str(result.get("chat_id") or body.get("chat_id") or "").strip()
        session_key = str(
            result.get("session_key") or body.get("session_key") or ""
        ).strip()
        tool = GenerateEchoShotTool(
            workspace=workspace,
            tools_config=self._tools_config,
        )
        if channel_name and chat_id:
            tool.set_context(
                channel_name,
                chat_id,
                effective_key=session_key or f"{channel_name}:{chat_id}",
            )
        return tool.apply_generate(
            work_id,
            shot_id,
            reference_ids,
            selection_note=selection_note,
        )

    async def _publish_runtime_event(
        self,
        operation: str,
        body: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        session_key = str(result.get("session_key") or body.get("session_key") or "").strip() or None
        channel = str(result.get("channel") or body.get("channel") or "").strip() or None
        chat_id = str(result.get("chat_id") or body.get("chat_id") or "").strip() or None
        await self.bus.publish_runtime(
            RuntimeEvent(
                kind="director_remote_result",
                source=self.name,
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                payload={
                    "operation": operation,
                    "work_id": result.get("work_id") or body.get("work_id"),
                    "job_id": result.get("job_id") or body.get("job_id"),
                    "status": body.get("status") or "completed",
                    "result": result,
                },
            )
        )

    async def _publish_workplace_updated(
        self,
        body: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:
        channel = str(result.get("channel") or body.get("channel") or "").strip()
        chat_id = str(result.get("chat_id") or body.get("chat_id") or "").strip()
        session_key = str(result.get("session_key") or body.get("session_key") or "").strip()
        if channel != "websocket" or not chat_id:
            logger.debug(
                "director_callback workplace push SKIPPED channel={} chat_id={}",
                channel or "-",
                chat_id or "-",
            )
            return False
        work_id = result.get("work_id") or body.get("work_id")
        media = result.get("video_paths") or result.get("result_urls") or result.get("media") or []
        logger.info(
            "director_callback workplace push SENT work_id={} session={} chat_id={} media_count={}",
            work_id or "-",
            session_key,
            chat_id,
            len(media) if isinstance(media, list) else 0,
        )
        await self.bus.publish_outbound(
            OutboundMessage(
                channel="websocket",
                chat_id=chat_id,
                content="",
                media=media if isinstance(media, list) else [],
                metadata={
                    "_workplace_event": "updated",
                    "session_key": session_key,
                    **({"work_id": work_id} if isinstance(work_id, str) and work_id.strip() else {}),
                },
            )
        )
        return True
