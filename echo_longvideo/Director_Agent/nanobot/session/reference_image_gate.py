"""Deterministic first-frame upload-gate matching (PDF truth table)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SESSION_GATE_DECLINE_STREAK_KEY = "reference_image_gate_decline_streak"
SESSION_GATE_SKIP_NEXT_KEY = "reference_image_gate_skip_next_message"
SESSION_GATE_PENDING_QUESTION_KEY = "reference_image_gate_pending_question"

CONFIRM_LABEL = "需要上传,已上传完毕"
DECLINE_LABEL = "不上传"
GATE_OPTIONS = (CONFIRM_LABEL, DECLINE_LABEL)

CONFIRM_ALIASES = frozenset({CONFIRM_LABEL, "已上传完毕", "已上传", "确认上传"})
DECLINE_ALIASES = frozenset({DECLINE_LABEL, "无需上传"})

MISMATCH_MISSING_QUESTION = "识别到未上传参考图，是否还需上传参考图？"
MISMATCH_PRESENT_QUESTION = "识别到已上传参考图，是否确认使用此参考图？"
INITIAL_GATE_QUESTION = "在开始构思之前，是否要上传首帧参考图？"

MATCH_INJECT_PREFIX = (
    "REFERENCE_IMAGE_GATE match=true present={present} intent={intent}. "
    "The user's upload-gate choice matches persisted first-frame state. "
    "Continue conceiving the story. Reply 「接下来我们开始构思吧，你想要什么样的故事呢？」 "
    "when they have not given a story idea yet. Do not re-ask whether to upload "
    "unless they pick 我想修改/增删参考图."
)

Intent = Literal["confirm", "decline", "other"]


@dataclass(frozen=True)
class UploadGateResult:
    intent: Intent
    match: bool
    skip_agent: bool
    delete_image: bool
    reset_streak: bool
    next_streak: int
    mismatch_question: str | None
    inject_note: str | None


def classify_upload_gate_intent(answer: str) -> Intent:
    label = (answer or "").strip()
    if label in CONFIRM_ALIASES:
        return "confirm"
    if label in DECLINE_ALIASES:
        return "decline"
    return "other"


def is_upload_gate_answer(answer: str) -> bool:
    return classify_upload_gate_intent(answer) != "other"


def decline_streak(metadata: dict[str, Any] | None) -> int:
    if not isinstance(metadata, dict):
        return 0
    try:
        return max(0, int(metadata.get(SESSION_GATE_DECLINE_STREAK_KEY) or 0))
    except (TypeError, ValueError):
        return 0


def evaluate_upload_gate(
    *,
    present: bool,
    answer: str,
    streak: int = 0,
) -> UploadGateResult:
    """Match option intent against persisted image presence.

    Second consecutive decline while present=true is a match and deletes the image.
    """
    intent = classify_upload_gate_intent(answer)
    if intent == "other":
        return UploadGateResult(
            intent=intent,
            match=True,
            skip_agent=False,
            delete_image=False,
            reset_streak=False,
            next_streak=streak,
            mismatch_question=None,
            inject_note=None,
        )
    if intent == "confirm":
        if present:
            return UploadGateResult(
                intent=intent,
                match=True,
                skip_agent=False,
                delete_image=False,
                reset_streak=True,
                next_streak=0,
                mismatch_question=None,
                inject_note=MATCH_INJECT_PREFIX.format(present="true", intent="confirm"),
            )
        return UploadGateResult(
            intent=intent,
            match=False,
            skip_agent=True,
            delete_image=False,
            reset_streak=True,
            next_streak=0,
            mismatch_question=MISMATCH_MISSING_QUESTION,
            inject_note=None,
        )
    # decline
    if not present:
        return UploadGateResult(
            intent=intent,
            match=True,
            skip_agent=False,
            delete_image=False,
            reset_streak=True,
            next_streak=0,
            mismatch_question=None,
            inject_note=MATCH_INJECT_PREFIX.format(present="false", intent="decline"),
        )
    if streak >= 1:
        return UploadGateResult(
            intent=intent,
            match=True,
            skip_agent=False,
            delete_image=True,
            reset_streak=True,
            next_streak=0,
            mismatch_question=None,
            inject_note=MATCH_INJECT_PREFIX.format(present="false", intent="decline"),
        )
    return UploadGateResult(
        intent=intent,
        match=False,
        skip_agent=True,
        delete_image=False,
        reset_streak=False,
        next_streak=streak + 1,
        mismatch_question=MISMATCH_PRESENT_QUESTION,
        inject_note=None,
    )


def commit_upload_gate(metadata: dict[str, Any], result: UploadGateResult) -> None:
    """Write streak / skip-next flags onto session metadata."""
    if result.reset_streak or result.next_streak == 0:
        metadata.pop(SESSION_GATE_DECLINE_STREAK_KEY, None)
    if result.next_streak > 0:
        metadata[SESSION_GATE_DECLINE_STREAK_KEY] = result.next_streak
    if result.skip_agent:
        metadata[SESSION_GATE_SKIP_NEXT_KEY] = True
        if result.mismatch_question:
            metadata[SESSION_GATE_PENDING_QUESTION_KEY] = result.mismatch_question
    else:
        metadata.pop(SESSION_GATE_SKIP_NEXT_KEY, None)
        metadata.pop(SESSION_GATE_PENDING_QUESTION_KEY, None)


def consume_skip_next_message(metadata: dict[str, Any] | None) -> str | None:
    """Pop skip-next and return the pending mismatch question, if any."""
    if not isinstance(metadata, dict):
        return None
    skip = bool(metadata.pop(SESSION_GATE_SKIP_NEXT_KEY, False))
    question = metadata.pop(SESSION_GATE_PENDING_QUESTION_KEY, None)
    if skip and isinstance(question, str) and question.strip():
        return question.strip()
    return None


def mismatch_card(question: str) -> dict[str, Any]:
    return {
        "id": "upload-gate",
        "question": question,
        "options": [{"label": CONFIRM_LABEL}, {"label": DECLINE_LABEL}],
        "allow_custom": False,
        "status": "pending",
        "answered": None,
    }
