"""VLM-based vision selection for director memory reviews."""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from loguru import logger

Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]
Sleeper = Callable[[float], None]


class MemorySelectorError(ValueError):
    """The selector request or model response is invalid."""


class SelectorTransportError(RuntimeError):
    """The configured VLM provider request failed."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.retryable = self.status_code == 429
        self.retry_after_s = retry_after_s


def _default_transport(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        try:
            retry_after_s = float(retry_after) if retry_after is not None else None
        except ValueError:
            retry_after_s = None
        raise SelectorTransportError(
            exc.code,
            detail or str(exc),
            retry_after_s=retry_after_s,
        ) from exc
    except urllib.error.URLError as exc:
        raise SelectorTransportError(0, str(exc.reason)) from exc
    if not isinstance(parsed, dict):
        raise MemorySelectorError("VLM provider response must be a JSON object")
    return parsed


def _extract_json(text: str) -> dict[str, Any]:
    value = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", value, re.DOTALL)
    if fenced:
        value = fenced.group(1)
    start, end = value.find("{"), value.rfind("}")
    if start >= 0 and end > start:
        value = value[start : end + 1]
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        logger.warning("VLM returned non-JSON response: {}", (text or "")[:500])
        raise MemorySelectorError("VLM response is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise MemorySelectorError("VLM response must be a JSON object")
    return parsed


class MemoryVlmSelector:
    """Select reviewable memories using the configured VLM model."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        transport: Transport = _default_transport,
        sleeper: Sleeper = time.sleep,
        timeout_s: float = 180,
        max_attempts: int = 3,
    ) -> None:
        if not api_base.strip() or not api_key.strip() or not model.strip():
            raise MemorySelectorError(
                "VLM provider apiBase, apiKey, and model are required for memory review"
            )
        if max_attempts < 1:
            raise MemorySelectorError("max_attempts must be positive")
        self.model = model.strip()
        self.endpoint = f"{api_base.rstrip('/')}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.transport = transport
        self.sleeper = sleeper
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts

    def _complete(self, *, content: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
        }
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.transport(
                    self.endpoint, self.headers, body, self.timeout_s)
                logger.debug(
                    "VLM response model={} finish_reason={} content_len={}",
                    self.model,
                    response.get("choices", [{}])[0].get("finish_reason", ""),
                    len(response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""),
                )
                text = response["choices"][0]["message"]["content"]
                if not isinstance(text, str):
                    raise TypeError
                if not text.strip():
                    logger.warning(
                        "VLM returned empty content model={} full_response={}",
                        self.model,
                        json.dumps(response, ensure_ascii=False)[:1000],
                    )
                return _extract_json(text)
            except SelectorTransportError as exc:
                if not exc.retryable or attempt >= self.max_attempts:
                    raise
                self.sleeper(
                    exc.retry_after_s
                    if exc.retry_after_s is not None and exc.retry_after_s >= 0
                    else 60.0
                )
            except (KeyError, IndexError, TypeError) as exc:
                logger.warning(
                    "VLM response missing content model={} full_response={}",
                    self.model,
                    json.dumps(response, ensure_ascii=False)[:1000],
                )
                raise MemorySelectorError("VLM response has no message content") from exc
        raise AssertionError("unreachable")

    @staticmethod
    def _available_candidates(
        candidates: list[dict[str, Any]], rejected: set[int] | None
    ) -> list[dict[str, Any]]:
        rejected = rejected or set()
        available = [item for item in candidates if int(item["candidate_index"]) not in rejected]
        if not available:
            raise MemorySelectorError("no memory candidates remain after exclusions")
        return available

    @staticmethod
    def _content(prompt: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for item in candidates:
            raw = Path(str(item["path"])).read_bytes()
            encoded = base64.b64encode(raw).decode("ascii")
            content.extend([
                {"type": "text", "text": (
                    f"candidate_index={item['candidate_index']} "
                    f"frame_index={item['frame_index']} "
                    f"timestamp_sec={item['timestamp_sec']}")},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}"}},
            ])
        return content

    def select_characters(
        self,
        *,
        shot_id: int,
        caption: str,
        character_ids: list[str],
        candidates: list[dict[str, Any]],
        rejected_candidate_indices: set[int] | None = None,
    ) -> dict[str, Any]:
        available = self._available_candidates(candidates, rejected_candidate_indices)
        if len(available) > 8:
            results = [
                self.select_characters(
                    shot_id=shot_id,
                    caption=caption,
                    character_ids=character_ids,
                    candidates=available[index:index + 8],
                )
                for index in range(0, len(available), 8)
            ]
            best_by_id: dict[str, dict[str, Any]] = {}
            for result in results:
                for item in result["selections"]:
                    memory_id = str(item["character_id"])
                    previous = best_by_id.get(memory_id)
                    if previous is None or float(item["confidence"]) > float(
                        previous["confidence"]
                    ):
                        best_by_id[memory_id] = item
            selected: list[dict[str, Any]] = []
            used_candidates: set[int] = set()
            for memory_id in character_ids:
                item = best_by_id.get(memory_id)
                if item is None or int(item["candidate_index"]) in used_candidates:
                    continue
                used_candidates.add(int(item["candidate_index"]))
                selected.append(item)
            return {
                "reasoning": " | ".join(str(result["reasoning"]) for result in results),
                "selections": selected,
            }
        manifest = [{key: item[key] for key in (
            "candidate_index", "frame_index", "timestamp_sec")} for item in available]
        prompt = (
            f"SHOT_ID: {shot_id}\nTARGET_CHARACTER_IDS: {character_ids}\n"
            f"SHOT_CAPTION: {caption}\nCANDIDATES: {json.dumps(manifest)}\n\n"
            "For each target ID, select one frame only when the exact person is clearly "
            "identifiable. Never map one visible person to two IDs. Return only JSON: "
            "{\"reasoning\":\"...\",\"selections\":[{\"character_id\":\"ID_A\","
            "\"candidate_index\":0,\"confidence\":0.0,\"target_only\":true,"
            "\"visible_character_ids\":[\"ID_A\"],\"reasoning\":\"...\"}]}.")
        available_indices = {int(item["candidate_index"]) for item in available}
        for attempt in range(2):
            current_prompt = prompt
            if attempt:
                current_prompt += (
                    "\nThe previous pass returned no character selections. Re-check every "
                    "candidate against the target descriptions and choose the clearest "
                    "identifiable frame for each visible target. A different person may be "
                    "partially visible at the edge; in that case set target_only=false and "
                    "list every visible ID. Return an empty selections list only when none "
                    "of the candidates visibly contains any target ID."
                )
            result = self._complete(
                content=self._content(current_prompt, available),
                max_tokens=4000,
            )
            self._validate_characters(
                result,
                set(character_ids),
                available_indices,
            )
            if result["selections"] or attempt == 1:
                return result
        raise AssertionError("unreachable")

    def decide_scene_transition(
        self,
        *,
        previous_shot_id: int,
        previous_caption: str,
        next_shot_id: int,
        next_caption: str,
    ) -> dict[str, Any]:
        """Decide whether the next outer shot changes the story scene."""
        prompt = (
            "Compare two adjacent OUTER shots from one screenplay. Decide whether "
            "the next shot changes scene. A scene transition means a material change "
            "in location, time, environment, or story situation. A camera cut, angle "
            "change, framing change, or continued action in the same setting is NOT "
            "a scene transition. Return only JSON: "
            '{"scene_transition":true,"reasoning":"..."}.\n\n'
            f"PREVIOUS_SHOT_ID: {previous_shot_id}\n"
            f"PREVIOUS_CAPTION:\n{previous_caption.strip()}\n\n"
            f"NEXT_SHOT_ID: {next_shot_id}\n"
            f"NEXT_CAPTION:\n{next_caption.strip()}"
        )
        result = self._complete(
            content=[{"type": "text", "text": prompt}],
            max_tokens=2000,
        )
        if (
            not isinstance(result.get("scene_transition"), bool)
            or not isinstance(result.get("reasoning"), str)
            or not result["reasoning"].strip()
        ):
            raise MemorySelectorError("invalid VLM scene transition decision")
        return {
            "scene_transition": result["scene_transition"],
            "reasoning": result["reasoning"].strip(),
        }

    @staticmethod
    def _validate_characters(
        result: dict[str, Any], requested: set[str], available: set[int]
    ) -> None:
        selections = result.get("selections")
        if not isinstance(result.get("reasoning"), str) or not isinstance(selections, list):
            raise MemorySelectorError("invalid VLM character selection schema")
        seen_ids: set[str] = set()
        seen_candidates: set[int] = set()
        for item in selections:
            if not isinstance(item, dict):
                raise MemorySelectorError("character selection must be an object")
            character_id = item.get("character_id")
            try:
                candidate_index = int(item["candidate_index"])
                confidence = float(item["confidence"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MemorySelectorError("invalid character candidate fields") from exc
            if (character_id not in requested or character_id in seen_ids
                    or candidate_index not in available or candidate_index in seen_candidates
                    or not 0 <= confidence <= 1
                    or not isinstance(item.get("target_only"), bool)
                    or not isinstance(item.get("visible_character_ids"), list)
                    or not all(isinstance(value, str) for value in item["visible_character_ids"])
                    or not isinstance(item.get("reasoning"), str)
                    or not item["reasoning"].strip()):
                raise MemorySelectorError("invalid VLM character selection")
            seen_ids.add(character_id)
            seen_candidates.add(candidate_index)

    def select_representative(
        self,
        *,
        shot_id: int,
        caption: str,
        candidates: list[dict[str, Any]],
        rejected_candidate_indices: set[int] | None = None,
    ) -> dict[str, Any]:
        available = self._available_candidates(candidates, rejected_candidate_indices)
        manifest = [{key: item[key] for key in (
            "candidate_index", "frame_index", "timestamp_sec")} for item in available]
        prompt = (
            f"SHOT_ID: {shot_id}\nSHOT_CAPTION: {caption}\n"
            f"CANDIDATES: {json.dumps(manifest)}\n\n"
            "Select exactly one clear, information-rich continuity frame. Return only JSON: "
            "{\"candidate_index\":0,\"confidence\":0.0,\"reasoning\":\"...\"}.")
        result = self._complete(content=self._content(prompt, available), max_tokens=4000)
        try:
            candidate_index = int(result["candidate_index"])
            confidence = float(result["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MemorySelectorError("invalid VLM representative fields") from exc
        if (candidate_index not in {int(item["candidate_index"]) for item in available}
                or not 0 <= confidence <= 1
                or not isinstance(result.get("reasoning"), str)
                or not result["reasoning"].strip()):
            raise MemorySelectorError("invalid VLM representative selection")
        return result

    def profile_image(
        self,
        *,
        image_path: Path,
        display_name: str = "",
    ) -> dict[str, Any]:
        """Create a compact, editable retrieval profile for one uploaded image."""
        raw = image_path.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        suffix = image_path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
        prompt = (
            "Describe this reusable video-reference asset for a director agent. "
            "Be concise and factual. Include visible people, appearance, clothing, "
            "objects, location, lighting, composition, and continuity cues when present. "
            "Do not invent names or identity IDs. Return only JSON: "
            '{"profile_text":"...","identity_ids":[]}.'
            f"\nFILE_LABEL: {display_name.strip() or image_path.name}"
        )
        result = self._complete(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
            ],
            max_tokens=1200,
        )
        profile_text = result.get("profile_text")
        identity_ids = result.get("identity_ids", [])
        if (
            not isinstance(profile_text, str)
            or not profile_text.strip()
            or not isinstance(identity_ids, list)
            or not all(isinstance(value, str) for value in identity_ids)
        ):
            raise MemorySelectorError("invalid VLM asset profile schema")
        return {
            "profile_text": profile_text.strip(),
            "identity_ids": [value.strip() for value in identity_ids if value.strip()],
        }
