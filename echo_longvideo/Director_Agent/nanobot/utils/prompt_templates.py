"""Load and render agent system prompt templates (Jinja2) under nanobot/templates/.

Agent prompts live in ``templates/agent/`` (pass names like ``agent/identity.md``).
Shared copy lives under ``agent/_snippets/`` and is included via
``{% include 'agent/_snippets/....md' %}``.
"""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

_TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "templates"


# @lru_cache
def _environment() -> Environment:
    # Plain-text prompts: do not HTML-escape variable values.
    # Search the active PE set's templates first, then the packaged defaults,
    # so an experimental PE set can override individual templates and fall back
    # for the rest. Read the PE dirs dynamically so hot-switching takes effect.
    search_paths: list[str] = []
    try:
        from nanobot.prompts import PEManager

        search_paths.extend(str(p) for p in PEManager.instance().templates_dir())
    except Exception:
        pass
    search_paths.append(str(_TEMPLATES_ROOT))
    return Environment(
        loader=FileSystemLoader(search_paths),
        auto_reload=True,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(name: str, *, strip: bool = False, **kwargs: Any) -> str:
    """Render ``name`` (e.g. ``agent/identity.md``, ``agent/platform_policy.md``) under ``templates/``.

    Use ``strip=True`` for single-line user-facing strings when the file ends
    with a trailing newline you do not want preserved.
    """
    text = _environment().get_template(name).render(**kwargs)
    return text.rstrip() if strip else text
