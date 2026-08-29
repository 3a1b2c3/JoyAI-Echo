"""Pure state transitions for human review of selected shot memories."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class MemoryReviewConflict(ValueError):
    """The browser acted on a stale or incompatible review state."""


def _assert_current(
    review: dict[str, Any],
    *,
    review_id: str,
    attempt: int,
) -> None:
    if (
        str(review.get("review_id") or "") != review_id
        or int(review.get("attempt") or 0) != int(attempt)
    ):
        raise MemoryReviewConflict("stale memory review attempt")


def approve_memory_review(
    review: dict[str, Any],
    *,
    review_id: str,
    attempt: int,
    updated_at: str,
    retained_memory_ids: list[str] | None = None,
) -> bool:
    """Approve the exact proposal, returning whether state changed."""
    _assert_current(review, review_id=review_id, attempt=attempt)
    status = str(review.get("status") or "")
    if status == "approved":
        return False
    if status != "awaiting_review":
        raise MemoryReviewConflict(f"memory review cannot be approved from {status}")
    available_ids = {
        str(item.get("memory_id") or "")
        for item in review.get("selections", [])
        if isinstance(item, dict) and item.get("memory_id")
    }
    if retained_memory_ids is None:
        source_ids = review.get("retained_memory_ids")
        retained_memory_ids = (
            list(source_ids)
            if isinstance(source_ids, list)
            else [
                str(item.get("memory_id") or "")
                for item in review.get("selections", [])
                if isinstance(item, dict) and item.get("memory_id")
            ]
        )
    retained = list(dict.fromkeys(str(value) for value in retained_memory_ids if value))
    unknown = sorted(set(retained) - available_ids)
    if unknown:
        raise MemoryReviewConflict(
            "retained memory is not in this review: " + ", ".join(unknown)
        )
    review["retained_memory_ids"] = retained
    review["status"] = "approved"
    review["error"] = None
    review["updated_at"] = updated_at
    return True


def reselect_memory_review(
    review: dict[str, Any],
    *,
    review_id: str,
    attempt: int,
    updated_at: str,
    memory_id: str | None = None,
) -> bool:
    """Reject one proposal item, or every item for legacy callers."""
    _assert_current(review, review_id=review_id, attempt=attempt)
    status = str(review.get("status") or "")
    if status == "reselecting":
        return False
    if status not in {"awaiting_review", "error"}:
        raise MemoryReviewConflict(f"memory review cannot be reselected from {status}")

    selections = [
        item for item in review.get("selections", []) if isinstance(item, dict)
    ]
    if memory_id:
        selections = [
            item for item in selections
            if str(item.get("memory_id") or "") == memory_id
        ]
        if not selections:
            raise MemoryReviewConflict(f"memory {memory_id} is not in this review")
    rejected = {
        int(value)
        for value in review.get("rejected_candidate_indices", [])
        if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
    }
    for selection in selections:
        try:
            rejected.add(int(selection["candidate_index"]))
        except (KeyError, TypeError, ValueError):
            continue

    history = review.setdefault("history", [])
    if not isinstance(history, list):
        history = []
        review["history"] = history
    history.append(
        {
            "attempt": int(attempt),
            "selections": deepcopy(selections),
            "rejected_at": updated_at,
        }
    )
    review["rejected_candidate_indices"] = sorted(rejected)
    review["reselect_memory_id"] = memory_id
    review["status"] = "reselecting"
    review["error"] = None
    review["updated_at"] = updated_at
    return True
