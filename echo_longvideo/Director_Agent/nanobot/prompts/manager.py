"""Runtime-switchable Prompt Engineering (PE) sets.

A *PE set* is a directory under the configured ``pe/`` root:

    pe/<name>/
        manifest.yaml         # {name, label, description}
        strings.yaml          # extracted prompt strings, dotted keys after flattening
        templates/            # optional Jinja template overrides (e.g. agent/identity.md)
        skills/<skill>/SKILL.md   # optional skill markdown overrides
        bootstrap/SOUL.md     # optional workspace-bootstrap overrides (SOUL/TOOLS/...)

The active set overlays ``default`` for every resource: strings fall back to
``default`` per key; template/skill/bootstrap lookups search ``active`` then
``default`` then the packaged locations.

``PEManager`` is a process-wide singleton so the WebUI can hot-switch the active
set without a restart. Consumers that cache derived artifacts (e.g. the tool
schema cache) subscribe via :meth:`on_change`.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

DEFAULT_SET_NAME = "default"


def default_pe_root() -> Path:
    """Return source-checkout PE resources, or the resources bundled in a wheel."""
    source_root = Path(__file__).resolve().parents[2] / "pe"
    if source_root.is_dir():
        return source_root
    return Path(__file__).resolve().parents[1] / "_bundled" / "pe"


def _flatten(data: Any, prefix: str = "") -> dict[str, str]:
    """Flatten a nested mapping into dotted keys with string values."""
    out: dict[str, str] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                out.update(_flatten(value, child))
            elif value is not None:
                out[child] = value if isinstance(value, str) else str(value)
    return out


class PEManager:
    """Process-wide holder of the active PE set and resource-resolution helpers."""

    _instance: "PEManager | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._root: Path | None = None
        self._active: str = DEFAULT_SET_NAME
        self._enabled: bool = True
        self._listeners: list[Callable[[str], None]] = []
        self._strings_cache: dict[str, dict[str, str]] = {}
        # Per-session active-set overrides (in-memory; lost on restart).
        self._session_sets: dict[str, str] = {}

    @classmethod
    def instance(cls) -> "PEManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    inst = cls()
                    root = default_pe_root()
                    if root.is_dir():
                        inst.configure(root)
                    cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------ config

    def configure(
        self,
        root: str | Path | None,
        active: str = DEFAULT_SET_NAME,
        enabled: bool = True,
    ) -> None:
        """(Re)initialize from config. Called once at CLI startup."""
        self._root = Path(root).expanduser().resolve() if root else None
        self._enabled = enabled
        self._strings_cache.clear()
        names = {entry["name"] for entry in self.list_sets()}
        if active in names:
            self._active = active
        elif DEFAULT_SET_NAME in names:
            self._active = DEFAULT_SET_NAME
        else:
            self._active = active
        logger.info(
            "PEManager configured: root=%s active=%s enabled=%s available=%s",
            self._root,
            self._active,
            self._enabled,
            sorted(names),
        )

    @property
    def root(self) -> Path | None:
        return self._root

    @property
    def active(self) -> str:
        return self._active

    @property
    def enabled(self) -> bool:
        return self._enabled

    # -------------------------------------------------------------------- sets

    def list_sets(self) -> list[dict[str, str]]:
        """Return metadata for every PE set directory under the root."""
        out: list[dict[str, str]] = []
        if not self._root or not self._root.is_dir():
            return out
        for child in sorted(self._root.iterdir()):
            if not child.is_dir():
                continue
            meta = {"name": child.name, "label": child.name, "description": ""}
            manifest = child / "manifest.yaml"
            if manifest.is_file():
                try:
                    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
                    if data.get("name"):
                        meta["name"] = str(data["name"])
                    meta["label"] = str(data.get("label") or meta["name"])
                    meta["description"] = str(data.get("description") or "")
                except Exception as exc:
                    logger.warning("Failed to parse PE manifest %s: %s", manifest, exc)
            out.append(meta)
        return out

    def set_active(self, name: str) -> bool:
        """Switch the active set and notify subscribers. Returns success."""
        if not self._enabled:
            logger.warning("PE switching disabled; ignoring set_active(%r)", name)
            return False
        names = {entry["name"] for entry in self.list_sets()}
        if name not in names:
            logger.warning("PE set %r not found; ignoring", name)
            return False
        if name == self._active:
            return True
        self._active = name
        logger.info("PE active set switched to %r", name)
        self._notify()
        return True

    def active_for_session(self, session_key: str | None) -> str:
        """Resolve the active set for a session: per-session override, else global.

        A stale override (set no longer present on disk) falls back to global.
        """
        if session_key:
            override = self._session_sets.get(session_key)
            if override and override in {e["name"] for e in self.list_sets()}:
                return override
        return self._active

    def set_active_for_session(self, session_key: str, name: str) -> bool:
        """Bind ``name`` as the active set for ``session_key`` only. Returns success."""
        if not self._enabled:
            logger.warning("PE switching disabled; ignoring set_active_for_session(%r)", name)
            return False
        if not session_key:
            return False
        if name not in {e["name"] for e in self.list_sets()}:
            logger.warning("PE set %r not found; ignoring session override", name)
            return False
        self._session_sets[session_key] = name
        logger.info("PE set %r bound to session %s", name, session_key)
        return True

    def on_change(self, callback: Callable[[str], None]) -> None:
        """Register a callback fired (with the new active name) on every switch."""
        self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in list(self._listeners):
            try:
                callback(self._active)
            except Exception:
                logger.exception("PE on_change listener failed")

    # ------------------------------------------------------ resource resolution

    def _resource_dirs(self, subdir: str, name: str | None = None) -> list[str]:
        """Existing ``<set>/<subdir>`` dirs for active then default (deduped).

        ``name`` overrides the active set (used for per-session resolution).
        """
        if not self._root:
            return []
        dirs: list[str] = []
        seen: set[str] = set()
        primary = name or self._active
        candidates = [primary, DEFAULT_SET_NAME] if self._enabled else [primary]
        for set_name in candidates:
            path = self._root / set_name / subdir
            key = str(path)
            if key not in seen and path.is_dir():
                dirs.append(key)
                seen.add(key)
        return dirs

    def templates_dir(self) -> list[str]:
        return self._resource_dirs("templates")

    def skills_dir(self) -> list[str]:
        return self._resource_dirs("skills")

    def bootstrap_dir(self) -> list[str]:
        return self._resource_dirs("bootstrap")

    def references_dir(self, name: str | None = None) -> list[str]:
        return self._resource_dirs("references", name)

    def resolve_reference(self, topic: str, name: str | None = None) -> Path | None:
        """Resolve a guidance topic to a ``<set>/references/<topic>.md`` file.

        Active set overlays default: the first existing match wins. ``name``
        overrides the active set (used for per-session resolution).
        """
        stem = topic.strip().removesuffix(".md")
        if not stem:
            return None
        for base in self.references_dir(name):
            for candidate in (Path(base) / f"{stem}.md", Path(base) / stem):
                if candidate.is_file():
                    return candidate
        return None

    def list_references(self, name: str | None = None) -> list[str]:
        """Available guidance topics (``.md`` basenames), active overlaying default."""
        topics: dict[str, None] = {}
        for base in self.references_dir(name):
            for path in sorted(Path(base).glob("*.md")):
                topics.setdefault(path.stem, None)
        return list(topics)

    # -------------------------------------------------------------- strings

    def _load_strings(self, name: str) -> dict[str, str]:
        if name in self._strings_cache:
            return self._strings_cache[name]
        flat: dict[str, str] = {}
        if self._root:
            path = self._root / name / "strings.yaml"
            if path.is_file():
                try:
                    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                    flat = _flatten(data)
                except Exception as exc:
                    logger.warning("Failed to parse PE strings %s: %s", path, exc)
        self._strings_cache[name] = flat
        return flat

    def get_string(self, key: str) -> str | None:
        """Resolve a flattened string key: active overlays default."""
        if not self._root:
            return None
        if self._enabled:
            active = self._load_strings(self._active)
            if key in active:
                return active[key]
        base = self._load_strings(DEFAULT_SET_NAME)
        return base.get(key)

    def reload(self) -> None:
        """Drop cached strings (e.g. after editing YAML on disk)."""
        self._strings_cache.clear()
