"""Build the embedded WebUI before packaging Echo Director Agent."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Ensure wheels and source distributions never contain a stale or missing WebUI."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        webui = root / "webui"
        dist_index = root / "nanobot" / "web" / "dist" / "index.html"

        # A repository checkout contains the frontend lockfile, so build the
        # embedded UI from its exact dependency graph.  A wheel built from our
        # sdist receives the already-built assets instead of the frontend
        # sources and must not attempt another npm install.
        if (webui / "package-lock.json").is_file():
            npm = shutil.which("npm")
            if npm is None:
                raise RuntimeError("Node.js and npm are required to build Echo Director Agent")
            subprocess.run([npm, "ci"], cwd=webui, check=True)
            subprocess.run([npm, "run", "build"], cwd=webui, check=True)

        if not dist_index.is_file():
            raise RuntimeError("WebUI build did not produce nanobot/web/dist/index.html")
