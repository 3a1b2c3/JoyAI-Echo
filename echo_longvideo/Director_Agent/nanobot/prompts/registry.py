"""Convenience accessor for PE-set prompt strings.

Usage::

    from nanobot.prompts import prompts

    text = prompts.text("director.tool.get_story.description")
    msg = prompts.text("director.callback.merge_shot.completed", work_id=w, final_output=o)

Missing keys log a warning and return ``default`` (or ``""``). String templates
use ``str.format`` placeholders; formatting is applied only when kwargs are given.
"""

from __future__ import annotations

import logging

from nanobot.prompts.manager import PEManager

logger = logging.getLogger(__name__)


class _Prompts:
    def text(self, key: str, default: str | None = None, /, **fmt: object) -> str:
        value = PEManager.instance().get_string(key)
        if value is None:
            if default is not None:
                value = default
            else:
                logger.warning("Missing PE prompt key: %s", key)
                return ""
        if fmt:
            try:
                return value.format(**fmt)
            except (KeyError, IndexError, ValueError) as exc:
                logger.warning("PE prompt format failed for key %s: %s", key, exc)
        return value


prompts = _Prompts()
