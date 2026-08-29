"""Session persistence helpers for ask_user question cards."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

_BATCH_ID_RE = re.compile(
    r"batch=([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)


def extract_batch_id_from_tool_result(content: str) -> str | None:
    """Return the question batch uuid embedded in an ask_user tool result."""
    if not content:
        return None
    match = _BATCH_ID_RE.search(content)
    return match.group(1) if match else None


def wire_questions_snapshot(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a WebUI-friendly questions array with stable option labels."""
    snapshot: list[dict[str, Any]] = []
    for card in questions:
        if not isinstance(card, dict):
            continue
        question = card.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        options_raw = card.get("options")
        if not isinstance(options_raw, list) or not options_raw:
            continue
        options: list[dict[str, str]] = []
        for opt in options_raw:
            if isinstance(opt, str) and opt.strip():
                options.append({"label": opt.strip()})
            elif isinstance(opt, dict) and isinstance(opt.get("label"), str) and opt["label"].strip():
                options.append({"label": opt["label"].strip()})
        if not options:
            continue
        entry: dict[str, Any] = {
            "id": card.get("id") if isinstance(card.get("id"), str) else f"q-{len(snapshot)}",
            "question": question.strip(),
            "options": options,
            "status": card.get("status") if isinstance(card.get("status"), str) else "pending",
        }
        if card.get("allow_custom") is True:
            entry["allow_custom"] = True
        if card.get("answered") is not None:
            entry["answered"] = card["answered"]
        snapshot.append(entry)
    return snapshot


def _questions_for_arguments(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build ask_user arguments payload from normalized question cards."""
    out: list[dict[str, Any]] = []
    for card in questions:
        if not isinstance(card, dict):
            continue
        question = card.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        options_raw = card.get("options")
        if not isinstance(options_raw, list) or not options_raw:
            continue
        options: list[Any] = []
        for opt in options_raw:
            if isinstance(opt, str) and opt.strip():
                options.append(opt.strip())
            elif isinstance(opt, dict) and isinstance(opt.get("label"), str) and opt["label"].strip():
                options.append(opt["label"].strip())
        if not options:
            continue
        entry: dict[str, Any] = {
            "id": card.get("id") if isinstance(card.get("id"), str) else f"q-{len(out)}",
            "question": question.strip(),
            "options": options,
        }
        if card.get("allow_custom") is True:
            entry["allow_custom"] = True
        if card.get("answered") is not None:
            entry["answered"] = card["answered"]
        if card.get("status") is not None:
            entry["status"] = card["status"]
        out.append(entry)
    return out


def questions_display_text(
    questions: list[dict[str, Any]],
    *,
    intro: str = "",
) -> str:
    """Flatten ask_user cards into plain assistant text for WebUI replay."""
    parts: list[str] = []
    intro = intro.strip()
    if intro:
        parts.append(intro)
    for card in questions:
        if not isinstance(card, dict):
            continue
        question = card.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        text = question.strip()
        if text not in parts:
            parts.append(text)
    return "\n".join(parts)


def build_ask_user_tool_call(
    tool_call_id: str,
    content: str,
    questions: list[dict[str, Any]],
    *,
    question_batch_id: str | None = None,
) -> dict[str, Any]:
    """Build one OpenAI-style ask_user tool_call entry."""
    # Intro text is ephemeral for the WebUI — only question cards are replayed.
    payload: dict[str, Any] = {
        "content": "",
        "questions": _questions_for_arguments(questions),
    }
    if question_batch_id:
        payload["question_batch_id"] = question_batch_id
    arguments = json.dumps(payload, ensure_ascii=False)
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": "ask_user",
            "arguments": arguments,
        },
    }


def build_ask_user_session_messages(
    *,
    tool_call_id: str,
    content: str,
    questions: list[dict[str, Any]],
    batch_id: str,
    channel: str,
    chat_id: str,
) -> list[dict[str, Any]]:
    """Return assistant + tool rows for one ask_user turn in fixed wire format."""
    now = datetime.now().isoformat()
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            build_ask_user_tool_call(
                tool_call_id,
                content,
                questions,
                question_batch_id=batch_id,
            )
        ],
        "question_batch_id": batch_id,
        "questions": wire_questions_snapshot(questions),
        "timestamp": now,
    }
    tool_result = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": "ask_user",
        "content": (
            f"Question cards sent to {channel}:{chat_id} "
            f"({len(questions)} card(s), batch={batch_id})"
        ),
        "timestamp": now,
    }
    return [assistant, tool_result]


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]] | None:
    """Normalize persisted tool_calls to OpenAI wire shape."""
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    normalized: list[dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        tool_call_id = tc.get("id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = fn.get("arguments")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        elif not isinstance(arguments, str):
            arguments = "{}"
        normalized.append(
            {
                "id": tool_call_id.strip(),
                "type": "function",
                "function": {
                    "name": name.strip(),
                    "arguments": arguments,
                },
            }
        )
    return normalized or None


def questions_snapshot_from_ask_user_tool_calls(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rebuild WebUI question cards from persisted ask_user tool_calls."""
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict) or fn.get("name") != "ask_user":
            continue
        parsed = _parse_tool_arguments(fn.get("arguments"))
        if not parsed:
            continue
        raw_questions = parsed.get("questions")
        if not isinstance(raw_questions, list) or not raw_questions:
            continue
        snapshot = wire_questions_snapshot(raw_questions)
        if snapshot:
            return snapshot
    return []


def sync_tool_calls_from_question_snapshot(message: dict[str, Any]) -> None:
    """Mirror assistant ``questions`` back into ask_user tool_call arguments."""
    if message.get("role") != "assistant":
        return
    snapshot = message.get("questions")
    if not isinstance(snapshot, list) or not snapshot:
        return
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    args_questions = _questions_for_arguments(snapshot)
    if not args_questions:
        return
    batch_id = message.get("question_batch_id")
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict) or fn.get("name") != "ask_user":
            continue
        parsed = _parse_tool_arguments(fn.get("arguments")) or {}
        payload: dict[str, Any] = {
            "content": "",
            "questions": args_questions,
        }
        if isinstance(batch_id, str) and batch_id.strip():
            payload["question_batch_id"] = batch_id.strip()
        elif isinstance(parsed.get("question_batch_id"), str):
            payload["question_batch_id"] = parsed["question_batch_id"]
        fn["arguments"] = json.dumps(payload, ensure_ascii=False)


def sync_message_question_snapshot(message: dict[str, Any]) -> None:
    """Keep assistant ``questions`` in sync with ask_user tool_call arguments."""
    if message.get("role") != "assistant":
        return
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    snapshot = questions_snapshot_from_ask_user_tool_calls(tool_calls)
    if snapshot:
        message["questions"] = snapshot


def normalize_persisted_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize one session row for stable WebUI replay."""
    entry = dict(message)
    role = entry.get("role")
    content = entry.get("content")

    if role == "assistant":
        tool_calls = normalize_tool_calls(entry.get("tool_calls"))
        if tool_calls:
            entry["tool_calls"] = tool_calls
            # ask_user tool_call arguments are authoritative; a stale native
            # ``questions`` snapshot must not override a later card batch.
            snapshot = questions_snapshot_from_ask_user_tool_calls(tool_calls)
            if snapshot:
                entry["questions"] = snapshot
                entry["content"] = ""
            else:
                existing = entry.get("questions")
                if isinstance(existing, list) and existing:
                    entry["questions"] = wire_questions_snapshot(existing)
                    entry["content"] = ""
                elif content is None:
                    entry["content"] = ""
                elif isinstance(content, str):
                    entry["content"] = content
                else:
                    entry["content"] = ""
        elif isinstance(entry.get("questions"), list) and entry["questions"]:
            entry["questions"] = wire_questions_snapshot(entry["questions"])
            entry["content"] = ""
        for key in ("reasoning_content", "thinking_blocks", "extra_content", "provider_specific_fields"):
            entry.pop(key, None)
    elif role == "tool":
        if isinstance(content, str):
            entry["content"] = content
        tool_call_id = entry.get("tool_call_id")
        if isinstance(tool_call_id, str):
            entry["tool_call_id"] = tool_call_id.strip()
        name = entry.get("name")
        if isinstance(name, str):
            entry["name"] = name.strip()
    elif role == "user" and isinstance(content, str):
        from nanobot.session.agent_inject import visible_user_content

        entry["content"] = visible_user_content(content)

    return entry


def session_has_tool_turn(messages: list[dict[str, Any]], tool_call_id: str) -> bool:
    """True when assistant+tool rows for *tool_call_id* are already persisted."""
    for message in messages:
        if message.get("role") != "tool":
            continue
        if message.get("tool_call_id") == tool_call_id:
            return True
    return False


def session_has_ask_user_turn(messages: list[dict[str, Any]], tool_call_id: str) -> bool:
    """True when an ask_user turn for *tool_call_id* is already in session history."""
    needle = tool_call_id.strip()
    if not needle:
        return False
    if session_has_tool_turn(messages, needle):
        return True
    for message in messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            fn = tc.get("function")
            if (
                isinstance(tc_id, str)
                and tc_id.strip() == needle
                and isinstance(fn, dict)
                and fn.get("name") == "ask_user"
            ):
                return True
    return False


def session_has_recent_ask_user_cards(messages: list[dict[str, Any]]) -> bool:
    """True when trailing assistant rows already delivered ask_user question cards."""
    for message in reversed(messages):
        role = message.get("role")
        if role == "user":
            return False
        if role != "assistant":
            continue
        questions = message.get("questions")
        if isinstance(questions, list) and questions:
            return True
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and questions_snapshot_from_ask_user_tool_calls(
            tool_calls
        ):
            return True
    return False


def _parse_tool_arguments(arguments: Any) -> dict[str, Any] | None:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _batch_id_from_assistant(message: dict[str, Any]) -> str | None:
    batch_id = message.get("question_batch_id")
    if isinstance(batch_id, str) and batch_id.strip():
        return batch_id.strip()
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return None
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict) or fn.get("name") != "ask_user":
            continue
        parsed = _parse_tool_arguments(fn.get("arguments"))
        if not parsed:
            continue
        args_batch = parsed.get("question_batch_id")
        if isinstance(args_batch, str) and args_batch.strip():
            return args_batch.strip()
    return None


def find_tool_call_id_for_batch(
    messages: list[dict[str, Any]],
    question_batch_id: str,
) -> str | None:
    """Resolve tool_call_id from a question batch id via the tool result row."""
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if extract_batch_id_from_tool_result(content) != question_batch_id:
            continue
        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id.strip():
            return tool_call_id.strip()
    return None


def _card_matches(card: dict[str, Any], card_id: str, index: int) -> bool:
    stored_id = card.get("id")
    if isinstance(stored_id, str) and stored_id == card_id:
        return True
    if card_id == f"q-{index}":
        return True
    return False


def _mark_card_answered(card: dict[str, Any], value: str) -> None:
    card["answered"] = value
    card["status"] = "answered"


def _update_cards_in_questions(
    questions: list[Any],
    card_id: str,
    value: str,
) -> bool:
    for index, card in enumerate(questions):
        if not isinstance(card, dict):
            continue
        if not _card_matches(card, card_id, index):
            continue
        _mark_card_answered(card, value)
        return True
    return False


def _update_assistant_tool_call_cards(
    message: dict[str, Any],
    *,
    question_batch_id: str,
    card_id: str,
    value: str,
    tool_call_id: str | None = None,
) -> bool:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return False
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        if tool_call_id and tc.get("id") != tool_call_id:
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict) or fn.get("name") != "ask_user":
            continue
        parsed = _parse_tool_arguments(fn.get("arguments"))
        if not parsed:
            continue
        args_batch = parsed.get("question_batch_id")
        if (
            isinstance(args_batch, str)
            and args_batch.strip()
            and args_batch.strip() != question_batch_id
        ):
            continue
        questions = parsed.get("questions")
        if not isinstance(questions, list):
            continue
        if not _update_cards_in_questions(questions, card_id, value):
            continue
        parsed["question_batch_id"] = question_batch_id
        fn["arguments"] = json.dumps(parsed, ensure_ascii=False)
        message["question_batch_id"] = question_batch_id
        sync_message_question_snapshot(message)
        return True
    return False


def update_ask_user_card_answer(
    messages: list[dict[str, Any]],
    question_batch_id: str,
    card_id: str,
    value: str,
) -> bool:
    """Update answered state on the ask_user assistant row for *question_batch_id*."""
    tool_call_id = find_tool_call_id_for_batch(messages, question_batch_id)

    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        batch_id = _batch_id_from_assistant(message)
        if batch_id and batch_id != question_batch_id:
            continue
        if not batch_id:
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            ask_ids = [
                tc.get("id")
                for tc in tool_calls
                if isinstance(tc, dict)
                and isinstance(tc.get("function"), dict)
                and tc["function"].get("name") == "ask_user"
                and isinstance(tc.get("id"), str)
            ]
            if not ask_ids:
                continue
            if tool_call_id:
                if tool_call_id not in ask_ids:
                    continue
            else:
                continue
        if _update_assistant_tool_call_cards(
            message,
            question_batch_id=question_batch_id,
            card_id=card_id,
            value=value,
            tool_call_id=tool_call_id,
        ):
            return True

    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        if message.get("question_batch_id") != question_batch_id:
            continue
        questions = message.get("questions")
        if not isinstance(questions, list):
            continue
        if _update_cards_in_questions(questions, card_id, value):
            sync_tool_calls_from_question_snapshot(message)
            return True

    return False


def _card_is_pending(card: dict[str, Any]) -> bool:
    if card.get("status") == "answered":
        return False
    return card.get("answered") is None


def _ensure_assistant_questions(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    questions = message.get("questions")
    if isinstance(questions, list) and questions:
        return questions
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return None
    snapshot = questions_snapshot_from_ask_user_tool_calls(tool_calls)
    if not snapshot:
        return None
    message["questions"] = snapshot
    return snapshot


def _assistant_has_questions(message: dict[str, Any]) -> bool:
    return _ensure_assistant_questions(message) is not None


def _card_accepts_reply(card: dict[str, Any], reply: str) -> bool:
    options_raw = card.get("options")
    if isinstance(options_raw, list):
        for opt in options_raw:
            if isinstance(opt, str) and opt.strip() == reply:
                return True
            if isinstance(opt, dict) and opt.get("label") == reply:
                return True
    if card.get("allow_custom") is True:
        return True
    return False


def _batch_accepts_reply(questions: list[Any], reply: str) -> bool:
    pending = [
        card
        for card in questions
        if isinstance(card, dict) and _card_is_pending(card)
    ]
    if not pending:
        return False
    return any(_card_accepts_reply(card, reply) for card in pending)


def session_has_user_reply(messages: list[dict[str, Any]], reply: str) -> bool:
    """True when *reply* already exists as any user row in *messages*."""
    needle = reply.strip()
    if not needle:
        return False
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip() == needle:
            return True
    return False


def session_recently_contains_user_reply(
    messages: list[dict[str, Any]],
    reply: str,
    *,
    tail: int = 6,
) -> bool:
    """True when an identical user reply already appears near the end of *messages*."""
    if session_has_user_reply(messages, reply):
        return True
    needle = reply.strip()
    if not needle:
        return False
    for message in messages[-tail:]:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip() == needle:
            return True
    return False


def session_has_user_reply_after_recent_ask_user_cards(
    messages: list[dict[str, Any]],
) -> bool:
    """True when the user already replied after the latest ask_user card batch."""
    card_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        if _assistant_has_questions(message):
            card_index = index
            break
    if card_index is None:
        return False
    for message in messages[card_index + 1 :]:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return True
    return False


def should_drop_assistant_blurb_after_cards(messages: list[dict[str, Any]]) -> bool:
    """Drop assistant text only while ask_user cards still await a user reply."""
    if not session_has_recent_ask_user_cards(messages):
        return False
    return not session_has_user_reply_after_recent_ask_user_cards(messages)


def ensure_user_reply_for_batch(
    messages: list[dict[str, Any]],
    question_batch_id: str,
    reply: str,
) -> bool:
    """Insert a user row for *reply* right after the ask_user batch when missing."""
    reply = reply.strip()
    if not reply:
        return False

    batch_index: int | None = None
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if _batch_id_from_assistant(message) == question_batch_id:
            batch_index = index
            break
    if batch_index is None:
        return False

    if session_has_user_reply(messages, reply):
        return False

    insert_at = batch_index + 1
    while insert_at < len(messages) and messages[insert_at].get("role") == "tool":
        insert_at += 1

    for index in range(insert_at, len(messages)):
        role = messages[index].get("role")
        if role == "assistant" and _assistant_has_questions(messages[index]):
            break
        if role != "user":
            continue
        content = messages[index].get("content")
        if isinstance(content, str) and content.strip() == reply:
            return False
        break

    messages.insert(
        insert_at,
        {
            "role": "user",
            "content": reply,
            "timestamp": datetime.now().isoformat(),
        },
    )
    return True


def apply_following_user_replies_to_question_cards(
    messages: list[dict[str, Any]],
) -> bool:
    """Mark ask_user cards answered when a user reply matches a preceding batch."""
    changed = False
    for user_index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        reply = content.strip()

        for assistant_index in range(user_index - 1, -1, -1):
            assistant = messages[assistant_index]
            if assistant.get("role") != "assistant":
                continue
            questions = _ensure_assistant_questions(assistant)
            if not questions or not _batch_accepts_reply(questions, reply):
                continue
            marked = False
            for card in questions:
                if (
                    isinstance(card, dict)
                    and _card_is_pending(card)
                    and _card_accepts_reply(card, reply)
                ):
                    _mark_card_answered(card, reply)
                    marked = True
            if marked:
                sync_tool_calls_from_question_snapshot(assistant)
                changed = True
            break
    return changed


def attach_batch_id_to_assistant_tool_call(
    messages: list[dict[str, Any]],
    *,
    tool_call_id: str,
    question_batch_id: str,
) -> None:
    """Backfill question_batch_id onto the assistant ask_user row for a tool result."""
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict) or tc.get("id") != tool_call_id:
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict) or fn.get("name") != "ask_user":
                continue
            message["question_batch_id"] = question_batch_id
            parsed = _parse_tool_arguments(fn.get("arguments"))
            if not isinstance(parsed, dict):
                sync_message_question_snapshot(message)
                return
            parsed["question_batch_id"] = question_batch_id
            raw_questions = parsed.get("questions")
            if (not isinstance(raw_questions, list) or not raw_questions) and isinstance(
                message.get("questions"), list
            ):
                parsed["questions"] = _questions_for_arguments(message["questions"])
            fn["arguments"] = json.dumps(parsed, ensure_ascii=False)
            sync_message_question_snapshot(message)
            return

