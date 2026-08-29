"""Ask-user tool: send structured question cards to the WebUI."""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import (
    ArraySchema,
    BooleanSchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.bus.events import OutboundMessage

REFERENCE_IMAGE_EDIT_OPTION = "我想修改/增删参考图"
STORY_CONFIRM_OPTION = "可以，按这个来"
STORY_REVISE_OPTION = "需要修改"
_REFERENCE_IMAGE_EDIT_ALIASES = frozenset(
    {
        REFERENCE_IMAGE_EDIT_OPTION,
        "我要修改/增删参考图",
    }
)
_STORY_DIRECTION_CARD_IDS = frozenset({"confirm_story", "story_direction"})
_STORY_DIRECTION_OPTION_MARKERS = frozenset({STORY_CONFIRM_OPTION, STORY_REVISE_OPTION})
_STORY_DIRECTION_QUESTION_MARKERS = (
    "故事方向",
    "什么样的故事",
    "什么故事",
    "想拍什么",
    "故事题材",
)
_NOT_STORY_DIRECTION_QUESTION_MARKERS = (
    "上传首帧",
    "上传参考图",
    "还需上传",
    "确认使用此参考图",
    "完成参考图修改",
    "几个镜头",
    "多少镜头",
)
_UPLOAD_GATE_OPTIONS = frozenset({"需要上传,已上传完毕", "不上传"})

_QUESTION_SCHEMA = ObjectSchema(
    {
        "id": StringSchema("Stable id for this card (auto-generated if omitted)"),
        "question": StringSchema("Question prompt shown above the option chips"),
        "options": ArraySchema(
            StringSchema("Option label"),
            description="Tap choices for the user",
            min_items=1,
        ),
        "allow_custom": BooleanSchema(
            description="Whether the user may type a custom answer",
            default=False,
        ),
    },
    required=["question", "options"],
)

PersistCallback = Callable[
    [str, str, str, str, list[dict[str, Any]], str, str],
    Awaitable[None],
]


def is_story_confirm_option(label: str) -> bool:
    return (label or "").strip() == STORY_CONFIRM_OPTION


def is_reference_image_edit_option(label: str) -> bool:
    return (label or "").strip() in _REFERENCE_IMAGE_EDIT_ALIASES


def _option_labels(options: list[Any]) -> list[str]:
    labels: list[str] = []
    for option in options:
        if isinstance(option, dict):
            labels.append(str(option.get("label") or "").strip())
        else:
            labels.append(str(option).strip())
    return [label for label in labels if label]


def is_story_direction_card(
    *,
    card_id: str,
    question: str,
    option_labels: list[str],
) -> bool:
    """True for confirm-story / story-premise cards that must offer reference-image edit."""
    if any(marker in question for marker in _NOT_STORY_DIRECTION_QUESTION_MARKERS):
        return False
    if _UPLOAD_GATE_OPTIONS.intersection(option_labels):
        return False
    if card_id in _STORY_DIRECTION_CARD_IDS:
        return True
    if any(marker in question for marker in _STORY_DIRECTION_QUESTION_MARKERS):
        return True
    return bool(_STORY_DIRECTION_OPTION_MARKERS.intersection(option_labels))


def ensure_reference_image_edit_option(card: dict[str, Any]) -> dict[str, Any]:
    """Append the canonical edit-reference option when a story-direction card omitted it."""
    options = card.get("options")
    if not isinstance(options, list):
        return card
    labels = _option_labels(options)
    if any(label in _REFERENCE_IMAGE_EDIT_ALIASES for label in labels):
        return card
    if not is_story_direction_card(
        card_id=str(card.get("id") or ""),
        question=str(card.get("question") or ""),
        option_labels=labels,
    ):
        return card
    card["options"] = [*options, {"label": REFERENCE_IMAGE_EDIT_OPTION}]
    return card


def normalize_question_cards(raw_questions: list[Any]) -> list[dict[str, Any]] | str:
    """Normalize and validate question payloads for WebUI + session storage."""
    if not isinstance(raw_questions, list) or not raw_questions:
        return "Error: questions must be a non-empty list"

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_questions):
        if not isinstance(item, dict):
            return f"Error: questions[{index}] must be an object"

        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            return f"Error: questions[{index}].question must be a non-empty string"

        options_raw = item.get("options")
        if not isinstance(options_raw, list) or not options_raw:
            return f"Error: questions[{index}].options must be a non-empty list"

        options: list[dict[str, str]] = []
        for opt_index, opt in enumerate(options_raw):
            if isinstance(opt, str) and opt.strip():
                options.append({"label": opt.strip()})
                continue
            if isinstance(opt, dict) and isinstance(opt.get("label"), str) and opt["label"].strip():
                options.append({"label": opt["label"].strip()})
                continue
            return f"Error: questions[{index}].options[{opt_index}] must be a string or {{label}}"

        card_id = item.get("id")
        if not isinstance(card_id, str) or not card_id.strip():
            card_id = f"q-{index}"

        allow_custom = item.get("allow_custom", item.get("allowCustom", False)) is True

        normalized.append(
            ensure_reference_image_edit_option(
                {
                    "id": card_id.strip(),
                    "question": question.strip(),
                    "options": options,
                    "allow_custom": allow_custom,
                    "status": "pending",
                    "answered": None,
                }
            )
        )

    return normalized


@tool_parameters(
    tool_parameters_schema(
        content=StringSchema(
            "Intro text shown above the cards. Do not repeat option labels here."
        ),
        questions=ArraySchema(
            _QUESTION_SCHEMA,
            description="Structured question cards for the user to tap",
            min_items=1,
        ),
        required=["content", "questions"],
    )
)
class AskUserTool(Tool):
    """Present structured multiple-choice cards in the WebUI."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        persist_callback: PersistCallback | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_session_key: str = "",
    ):
        self._send_callback = send_callback
        self._persist_callback = persist_callback
        self._default_channel: ContextVar[str] = ContextVar(
            "ask_user_default_channel", default=default_channel
        )
        self._default_chat_id: ContextVar[str] = ContextVar(
            "ask_user_default_chat_id", default=default_chat_id
        )
        self._default_session_key: ContextVar[str] = ContextVar(
            "ask_user_default_session_key", default=default_session_key
        )
        self._tool_call_id: ContextVar[str] = ContextVar("ask_user_tool_call_id", default="")
        self._sent_in_turn_var: ContextVar[bool] = ContextVar(
            "ask_user_sent_in_turn", default=False
        )

    def set_context(
        self,
        channel: str,
        chat_id: str,
        *,
        session_key: str | None = None,
    ) -> None:
        self._default_channel.set(channel)
        self._default_chat_id.set(chat_id)
        if session_key:
            self._default_session_key.set(session_key)

    def set_tool_call_id(self, tool_call_id: str) -> None:
        self._tool_call_id.set(tool_call_id.strip())

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        self._send_callback = callback

    def set_persist_callback(self, callback: PersistCallback) -> None:
        self._persist_callback = callback

    def start_turn(self) -> None:
        self._sent_in_turn = False

    @property
    def _sent_in_turn(self) -> bool:
        return self._sent_in_turn_var.get()

    @_sent_in_turn.setter
    def _sent_in_turn(self, value: bool) -> None:
        self._sent_in_turn_var.set(value)

    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return (
            "Ask the user one or more multiple-choice questions using interactive "
            "cards in the WebUI. Use this instead of listing options as plain text. "
            "Put the intro in `content` and each question in `questions` with "
            "`options` labels. The user's tap is persisted and survives page refresh."
        )

    async def execute(
        self,
        content: str,
        questions: list[Any] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        from nanobot.utils.helpers import strip_think

        content = strip_think(content or "")
        if not content.strip():
            return "Error: content must be a non-empty string"

        normalized = normalize_question_cards(questions or [])
        if isinstance(normalized, str):
            return normalized

        channel = channel or self._default_channel.get()
        chat_id = chat_id or self._default_chat_id.get()
        session_key = self._default_session_key.get()

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"
        if not self._send_callback:
            return "Error: ask_user delivery not configured"

        batch_id = str(uuid.uuid4())
        metadata: dict[str, Any] = {
            "questions": normalized,
            "question_batch_id": batch_id,
        }

        try:
            if self._persist_callback and session_key:
                tool_call_id = self._tool_call_id.get().strip()
                if not tool_call_id:
                    tool_call_id = f"call_{batch_id}"
                await self._persist_callback(
                    session_key,
                    tool_call_id,
                    batch_id,
                    content.strip(),
                    normalized,
                    channel,
                    chat_id,
                )
            await self._send_callback(
                OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    content="",
                    metadata=metadata,
                )
            )
            if channel == self._default_channel.get() and chat_id == self._default_chat_id.get():
                self._sent_in_turn = True
            return (
                f"Question cards sent to {channel}:{chat_id} "
                f"({len(normalized)} card(s), batch={batch_id})"
            )
        except Exception as e:
            return f"Error sending question cards: {e}"

