"""Context builder for assembling agent prompts."""

import base64
import json
import mimetypes
import platform
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.event_stacker import EventStacker
from nanobot.agent.memory import MemoryStore
from nanobot.agent.prompt_stacker import PromptStacker
from nanobot.agent.skills import SkillsLoader
from nanobot.utils.helpers import (
    build_assistant_message,
    current_time_str,
    detect_image_mime,
    truncate_text,
)
from nanobot.utils.prompt_templates import render_template


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    _MAX_RECENT_HISTORY = 50
    _MAX_HISTORY_CHARS = 32_000  # hard cap on recent history section size
    _RUNTIME_CONTEXT_END = "[/Runtime Context]"

    def __init__(self, workspace: Path, timezone: str | None = None, disabled_skills: list[str] | None = None):
        self.workspace = workspace
        self.timezone = timezone
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace, disabled_skills=set(disabled_skills) if disabled_skills else None)

    def resolve_active_skills(self, session_metadata: dict[str, Any] | None = None) -> list[str]:
        """Return the skills configured to be always active."""
        return self.skills.get_always_skills()

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        channel: str | None = None,
        session_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        parts = [self._get_identity(channel=channel)]
        PromptStacker.log("identity", parts[-1])
        EventStacker.log("identity", parts[-1])

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
            PromptStacker.log("bootstrap", bootstrap)
            EventStacker.log("bootstrap", bootstrap)

        memory = self.memory.get_memory_context()
        if memory and not self._is_template_content(self.memory.read_memory(), "memory/MEMORY.md"):
            mem_part = f"# Memory\n\n{memory}"
            parts.append(mem_part)
            PromptStacker.log("memory", mem_part)
            EventStacker.log("memory", mem_part)

        active_skills = (
            skill_names
            if skill_names is not None
            else self.resolve_active_skills(session_metadata)
        )
        if active_skills:
            always_content = self.skills.load_skills_for_context(active_skills)
            if always_content:
                skill_part = f"# Active Skills\n\n{always_content}"
                parts.append(skill_part)
                PromptStacker.log("active_skills", skill_part)
                EventStacker.log("active_skills", skill_part)

        meta = session_metadata if isinstance(session_metadata, dict) else {}

        pe_guidance = self._load_session_pe_guidance(meta)
        if pe_guidance:
            parts.append(pe_guidance)
            PromptStacker.log("pe_shot_prompt_writer", pe_guidance)
            EventStacker.log("pe_shot_prompt_writer", pe_guidance)

        entries = self.memory.read_unprocessed_history(
            since_cursor=self.memory.get_last_dream_cursor()
        )
        if entries:
            capped = entries[-self._MAX_RECENT_HISTORY :]
            history_text = "\n".join(
                f"- [{entry['timestamp']}] {entry['content']}" for entry in capped
            )
            history_text = truncate_text(history_text, self._MAX_HISTORY_CHARS)
            history_part = "# Recent History\n\n" + history_text
            parts.append(history_part)
            PromptStacker.log("recent_history", history_part)
            EventStacker.log("recent_history", history_part)
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _load_session_pe_guidance(session_metadata: dict[str, Any]) -> str | None:
        """Inject the session-selected PE set's shot-prompt-writer contract.

        The caption contract is the only artifact that differs between PE sets.
        Injecting it directly into the system prompt makes the selected set take
        effect deterministically, without relying on the model calling
        ``get_guidance``. Only fires when the session explicitly picked a set
        (``metadata['pe_set']``); unselected sessions keep the prior behavior.

        A/B integrity guarantee: the injected content MUST come from the exact
        set the session selected. If the set is unknown, or resolution would
        silently fall back to ``default``'s copy, we log an error and inject
        NOTHING rather than contaminate the experiment with the wrong prompt.
        """
        pe_set = session_metadata.get("pe_set")
        if not isinstance(pe_set, str) or not pe_set:
            return None
        try:
            from nanobot.prompts import PEManager

            manager = PEManager.instance()
            known = {entry["name"] for entry in manager.list_sets()}
            if pe_set not in known:
                logger.error(
                    "PE injection aborted: session pe_set={} is not a known set {}",
                    pe_set,
                    sorted(known),
                )
                return None
            ref = manager.resolve_reference("shot-prompt-writer", name=pe_set)
        except Exception as exc:
            logger.error("PE injection failed resolving pe_set={}: {}", pe_set, exc)
            return None
        if ref is None or not ref.is_file():
            logger.error(
                "PE injection aborted: pe_set={} has no shot-prompt-writer reference", pe_set
            )
            return None
        # Guard against a silent overlay fallback to ``default``: the resolved
        # file must live under ``<root>/<pe_set>/references/``.
        owning_set = ref.parent.parent.name
        if owning_set != pe_set:
            logger.error(
                "PE injection aborted: pe_set={} resolved to set {!r} (fallback); "
                "refusing to inject the wrong contract",
                pe_set,
                owning_set,
            )
            return None
        content = ref.read_text(encoding="utf-8").strip()
        if not content:
            logger.error("PE injection aborted: pe_set={} shot-prompt-writer is empty", pe_set)
            return None
        logger.info("PE injection: session using set={} ({} chars)", pe_set, len(content))
        return f"# Shot Caption Contract (PE set: {pe_set})\n\n{content}"

    def _get_identity(self, channel: str | None = None) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "",
        )

    @staticmethod
    def _build_runtime_context(
        channel: str | None, chat_id: str | None, timezone: str | None = None,
        session_summary: str | None = None,
        *,
        session_key: str | None = None,
        session_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        lines = [f"Current Time: {current_time_str(timezone)}"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        if session_summary:
            lines += ["", "[Resumed Session]", session_summary]
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines) + "\n" + ContextBuilder._RUNTIME_CONTEXT_END

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    def _pe_bootstrap_dirs(self) -> list[Path]:
        """Active-then-default PE ``bootstrap/`` dirs, read live so hot-switch applies."""
        try:
            from nanobot.prompts import PEManager

            return [Path(p) for p in PEManager.instance().bootstrap_dir()]
        except Exception:
            return []

    def _load_bootstrap_files(self) -> str:
        """Load bootstrap files: an active PE set may override the workspace copy."""
        parts = []
        pe_dirs = self._pe_bootstrap_dirs()

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            for pe_dir in pe_dirs:
                candidate = pe_dir / filename
                if candidate.is_file():
                    file_path = candidate
                    break
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _is_template_content(content: str, template_path: str) -> bool:
        """Check if *content* is identical to the bundled template (user hasn't customized it)."""
        try:
            tpl = pkg_files("nanobot") / "templates" / template_path
            if tpl.is_file():
                return content.strip() == tpl.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        return False

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        current_role: str = "user",
        session_summary: str | None = None,
        session_metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    skill_names,
                    channel=channel,
                    session_metadata=session_metadata,
                ),
            },
            *history,
        ]

        if history:
            PromptStacker.log("history", f"[{len(history)} history messages]")
            EventStacker.log("history", f"[{len(history)} history messages]")

        if current_role == "system":
            instruction = current_message.strip()
            if instruction:
                first = dict(messages[0])
                first["content"] = f"{first.get('content', '')}\n\n# Runtime System Instruction\n{instruction}"
                messages[0] = first
                PromptStacker.log("system_instruction", instruction)
                EventStacker.log("system_instruction", instruction)
            return messages

        runtime_ctx = self._build_runtime_context(
            channel,
            chat_id,
            self.timezone,
            session_summary=session_summary,
            session_key=session_key,
            session_metadata=session_metadata,
        )
        PromptStacker.log("runtime_context", runtime_ctx)
        EventStacker.log("runtime_context", runtime_ctx)

        if current_message:
            PromptStacker.log("user_message", current_message)
            EventStacker.log("user_message", current_message)

        user_content = self._build_user_content(current_message, media)
        user_content = self._maybe_inject_story_reference_image(
            user_content,
            session_metadata=session_metadata,
            session_key=session_key,
        )

        # Merge runtime context and user content into a single user message
        # to avoid consecutive same-role messages that some providers reject.
        if isinstance(user_content, str):
            merged = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content
        if messages[-1].get("role") == current_role:
            last = dict(messages[-1])
            last["content"] = self._merge_message_content(last.get("content"), merged)
            messages[-1] = last
            return messages
        messages.append({"role": current_role, "content": merged})
        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": {"path": str(p)},
            })

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def _maybe_inject_story_reference_image(
        self,
        user_content: str | list[dict[str, Any]],
        *,
        session_metadata: dict[str, Any] | None,
        session_key: str | None,
    ) -> str | list[dict[str, Any]]:
        """Attach the persisted first-frame image while the story is still writable."""
        from nanobot.session.reference_image import (
            download_reference_image_data_uri,
            is_reference_image_locked,
            normalize_reference_image,
            reference_image_needs_story_rewrite,
            story_reference_image_inject_note,
        )
        metadata = session_metadata if isinstance(session_metadata, dict) else {}
        state = self._director_state_for_session(session_key)
        locked = is_reference_image_locked(state) or is_reference_image_locked(metadata)
        if locked:
            return user_content
        ref = normalize_reference_image(
            (state or {}).get("reference_image") or metadata.get("reference_image")
        )
        if not ref:
            return user_content
        url = str(ref.get("url") or "").strip()
        if not url:
            return user_content
        try:
            data_uri = download_reference_image_data_uri(url)
        except Exception:
            logger.exception(
                "reference image inject failed session_key={} url={}",
                session_key or "-",
                url,
            )
            metadata["reference_image_inject_failed"] = True
            return user_content
        metadata.pop("reference_image_inject_failed", None)
        image_block = {
            "type": "image_url",
            "image_url": {"url": data_uri},
            "_meta": {"source": "reference_image", "url": url},
        }
        note = story_reference_image_inject_note(
            replaced=reference_image_needs_story_rewrite(metadata),
        )
        if isinstance(user_content, str):
            return [image_block, {"type": "text", "text": f"{note}\n\n{user_content}"}]
        if isinstance(user_content, list):
            return [image_block, {"type": "text", "text": note}, *user_content]
        return user_content

    def _director_state_for_session(self, session_key: str | None) -> dict[str, Any] | None:
        if not session_key:
            return None
        root = self.workspace / "director"
        map_path = root / "session_map.json"
        if not map_path.is_file():
            return None
        try:
            payload = json.loads(map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        work_id = payload.get(session_key)
        if isinstance(work_id, dict):
            work_id = work_id.get("work_id")
        if not isinstance(work_id, str) or not work_id.strip():
            return None
        state_path = root / "works" / work_id.strip() / "state.json"
        if not state_path.is_file():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return state if isinstance(state, dict) else None

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: Any,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        messages.append(build_assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        ))
        return messages
