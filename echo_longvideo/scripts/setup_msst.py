"""Install the pinned MSST-WebUI source and Echo 1.5 Bandit checkpoint."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MSST_REPOSITORY = "https://github.com/SUC-DriverOld/MSST-WebUI.git"
MSST_COMMIT = "43e30b860c611b516ed9b67c75a56792a67ec902"
MSST_DIR = REPO_ROOT / "third_party" / "MSST-WebUI"
MODEL_NAME = "model_bandit_plus_dnr_sdr_11.47.chpt"
MODEL_URL = (
    "https://huggingface.co/Sucial/MSST-WebUI/resolve/main/"
    f"All_Models/multi_stem_models/{MODEL_NAME}"
)
MODEL_SHA256 = "c48284779f7d1258a6527d3aaa18a532d45c1f506e2dcc25d5ab179a8c5e2573"
MODEL_PATH = REPO_ROOT / "checkpoints" / "msst" / MODEL_NAME
CONFIG_PATH = MSST_DIR / "configs_backup" / "multi_stem_models" / f"{MODEL_NAME}.yaml"
CONFIG_SHA256 = "4d3bf5b9fb9d0480bf9cb64eaed23edb37815aaaf905d74ca6393c13a32ce58e"


def _run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(args),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_source() -> None:
    if MSST_DIR.exists():
        if not (MSST_DIR / ".git").is_dir():
            raise RuntimeError(f"existing MSST path is not a Git checkout: {MSST_DIR}")
        current_commit = _run("git", "rev-parse", "HEAD", cwd=MSST_DIR)
        if current_commit != MSST_COMMIT:
            raise RuntimeError(
                f"MSST checkout is at {current_commit}; expected {MSST_COMMIT}. "
                "Move it aside before running setup again."
            )
        if _run("git", "status", "--porcelain", cwd=MSST_DIR):
            raise RuntimeError(f"MSST checkout contains local changes: {MSST_DIR}")
        return

    MSST_DIR.parent.mkdir(parents=True, exist_ok=True)
    _run(
        "git",
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        MSST_REPOSITORY,
        str(MSST_DIR),
    )
    _run("git", "checkout", "--detach", MSST_COMMIT, cwd=MSST_DIR)


def install_model() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.is_file():
        actual = _sha256(MODEL_PATH)
        if actual != MODEL_SHA256:
            raise RuntimeError(
                f"existing model checksum mismatch: expected {MODEL_SHA256}, got {actual}"
            )
        return

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=MODEL_PATH.parent,
            prefix=f".{MODEL_NAME}.",
            suffix=".download",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = urllib.request.Request(
                MODEL_URL,
                headers={"User-Agent": "JoyAI-Echo15-MSST-Setup/1.0"},
            )
            with urllib.request.urlopen(request) as response:
                shutil.copyfileobj(response, temporary)
        actual = _sha256(temporary_path)
        if actual != MODEL_SHA256:
            raise RuntimeError(
                f"downloaded model checksum mismatch: expected {MODEL_SHA256}, got {actual}"
            )
        temporary_path.replace(MODEL_PATH)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    install_source()
    install_model()
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"MSST Bandit config not found: {CONFIG_PATH}")
    config_digest = _sha256(CONFIG_PATH)
    if config_digest != CONFIG_SHA256:
        raise RuntimeError(
            f"MSST config checksum mismatch: expected {CONFIG_SHA256}, got {config_digest}"
        )
    print(f"MSST source: {MSST_DIR}")
    print(f"MSST commit: {MSST_COMMIT}")
    print(f"MSST model:  {MODEL_PATH}")
    print(f"MSST config: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
