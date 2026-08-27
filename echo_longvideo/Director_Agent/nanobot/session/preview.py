"""LLM-backed session sidebar preview generation."""

from __future__ import annotations

import re

from loguru import logger

from nanobot.providers.base import LLMProvider
from nanobot.utils.prompt_templates import render_template

_PREVIEW_MAX_CHARS = 15
_PREVIEW_SOURCE_KEY = "preview_source"


def clamp_preview(text: str, *, max_chars: int = _PREVIEW_MAX_CHARS) -> str:
    normalized = " ".join((text or "").split()).strip()
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars]


class SessionPreviewGenerator:
    """Summarize the first user turn into a short sidebar title via the configured provider."""

    def __init__(self, provider: LLMProvider, model: str) -> None:
        self._provider = provider
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def summarize(self, first_user_message: str) -> str:
        source = " ".join(first_user_message.split()).strip()
        if not source:
            return ""
        if len(source) <= _PREVIEW_MAX_CHARS:
            return source
        try:
            response = await self._provider.chat_with_retry(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": render_template("agent/session_preview.md", strip=True),
                    },
                    {"role": "user", "content": source},
                ],
                tools=None,
                max_tokens=64,
                temperature=0.2,
            )
        except Exception:
            logger.opt(exception=True).warning("session preview LLM call failed")
            return clamp_preview(source)
        text = (response.content or "").strip()
        text = re.sub(r'^["\'“”‘’]+|["\'“”‘’]+$', "", text)
        text = " ".join(text.split()).strip()
        return clamp_preview(text or source)
