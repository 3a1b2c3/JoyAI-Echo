"""In-process MSST speech-stem extraction used by Echo 1.5 memory."""

from __future__ import annotations

import gc
import importlib
import importlib.util
import logging
import os
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MSSTSpeechConfig:
    msst_dir: str
    model_path: str
    config_path: str
    model_type: str = "bandit"
    sample_rate: int = 44100
    device: str = "auto"
    local_rank_env: str = "LOCAL_RANK"


_SEPARATORS: dict[tuple[Any, ...], Any] = {}


def validate_msst_config(config: MSSTSpeechConfig) -> None:
    checks = (
        ("MSST source directory", Path(config.msst_dir), Path.is_dir),
        ("MSST model", Path(config.model_path), Path.is_file),
        ("MSST model config", Path(config.config_path), Path.is_file),
    )
    for label, path, predicate in checks:
        if not predicate(path.expanduser()):
            raise FileNotFoundError(f"{label} not found: {path.expanduser().resolve()}")
    if config.model_type != "bandit":
        raise ValueError("Echo 1.5 production memory requires MSST model_type=bandit")
    if int(config.sample_rate) != 44100:
        raise ValueError("Echo 1.5 production MSST configuration requires 44100 Hz")


@contextmanager
def _temporary_cwd(path: str):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _resolve_device_ids(device: str, local_rank_env: str) -> list[int]:
    if not torch.cuda.is_available():
        return [0]
    normalized = str(device).strip().lower()
    if normalized.startswith("cuda:"):
        try:
            explicit_device = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid CUDA device: {device}") from exc
        if explicit_device < 0 or explicit_device >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {explicit_device} is unavailable; "
                f"visible devices={torch.cuda.device_count()}"
            )
        torch.cuda.set_device(explicit_device)
        return [explicit_device]
    raw_rank = os.environ.get(local_rank_env, os.environ.get("LOCAL_RANK", "0"))
    try:
        local_rank = int(raw_rank)
    except (TypeError, ValueError):
        local_rank = 0
    local_rank = max(0, min(local_rank, torch.cuda.device_count() - 1))
    if normalized in {"auto", "cuda"}:
        torch.cuda.set_device(local_rank)
    return [local_rank]


def _drop_conflicting_modules(msst_dir: str) -> None:
    for name, module in list(sys.modules.items()):
        if not any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in ("utils", "inference", "modules")
        ):
            continue
        module_file = str(getattr(module, "__file__", "") or "")
        if module_file and module_file.startswith(msst_dir):
            continue
        sys.modules.pop(name, None)


def _install_bandit_inference_compat(msst_dir: str) -> None:
    """Avoid importing Bandit's training-only Lightning and spafe paths."""

    if (
        "pytorch_lightning" not in sys.modules
        and importlib.util.find_spec("pytorch_lightning") is None
    ):
        lightning = types.ModuleType("pytorch_lightning")
        lightning.__path__ = []
        lightning.LightningModule = torch.nn.Module
        utilities = types.ModuleType("pytorch_lightning.utilities")
        utilities.__path__ = []
        utility_types = types.ModuleType("pytorch_lightning.utilities.types")
        utility_types.STEP_OUTPUT = Any
        utilities.types = utility_types
        lightning.utilities = utilities
        sys.modules["pytorch_lightning"] = lightning
        sys.modules["pytorch_lightning.utilities"] = utilities
        sys.modules["pytorch_lightning.utilities.types"] = utility_types

    if "spafe" not in sys.modules and importlib.util.find_spec("spafe") is None:

        def unavailable_spafe(*_args, **_kwargs):
            raise RuntimeError("spafe is required for Bark/ERB Bandit configurations")

        spafe = types.ModuleType("spafe")
        spafe.__path__ = []
        fbanks = types.ModuleType("spafe.fbanks")
        fbanks.bark_fbanks = unavailable_spafe
        utils = types.ModuleType("spafe.utils")
        utils.__path__ = []
        converters = types.ModuleType("spafe.utils.converters")
        converters.erb2hz = unavailable_spafe
        converters.hz2bark = unavailable_spafe
        converters.hz2erb = unavailable_spafe
        spafe.fbanks = fbanks
        spafe.utils = utils
        utils.converters = converters
        sys.modules["spafe"] = spafe
        sys.modules["spafe.fbanks"] = fbanks
        sys.modules["spafe.utils"] = utils
        sys.modules["spafe.utils.converters"] = converters

    if msst_dir not in sys.path:
        sys.path.insert(0, msst_dir)
    importlib.import_module("modules.bandit")
    core_name = "modules.bandit.core"
    if core_name not in sys.modules:
        core_path = Path(msst_dir) / "modules" / "bandit" / "core"
        core = types.ModuleType(core_name)
        core.__file__ = str(core_path / "__init__.py")
        core.__path__ = [str(core_path)]
        core.__package__ = core_name
        sys.modules[core_name] = core


def _disable_progress_bars() -> None:
    if os.environ.get("MSST_SPEECH_DISABLE_PROGRESS", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return
    try:
        utils_module = importlib.import_module("utils.utils")
        real_tqdm = getattr(utils_module, "tqdm", None)
        if real_tqdm is None or getattr(real_tqdm, "_echo_msst_quiet", False):
            return

        def quiet_tqdm(*args, **kwargs):
            kwargs["disable"] = True
            return real_tqdm(*args, **kwargs)

        quiet_tqdm._echo_msst_quiet = True
        utils_module.tqdm = quiet_tqdm
    except Exception as exc:  # pragma: no cover - cosmetic compatibility
        logger.warning("Could not disable MSST progress bars: %s", exc)


def get_msst_separator(config: MSSTSpeechConfig):
    validate_msst_config(config)
    msst_dir = str(Path(config.msst_dir).expanduser().resolve())
    model_path = str(Path(config.model_path).expanduser().resolve())
    config_path = str(Path(config.config_path).expanduser().resolve())
    device_ids = _resolve_device_ids(config.device, config.local_rank_env)
    cache_key = (
        msst_dir,
        model_path,
        config_path,
        config.model_type,
        config.device,
        tuple(device_ids),
    )
    separator = _SEPARATORS.get(cache_key)
    if separator is not None:
        return separator

    if msst_dir in sys.path:
        sys.path.remove(msst_dir)
    sys.path.insert(0, msst_dir)
    _drop_conflicting_modules(msst_dir)
    _install_bandit_inference_compat(msst_dir)

    msst_logger = logging.getLogger("logger")
    if not hasattr(msst_logger, "console_handler"):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        msst_logger.addHandler(console_handler)
        msst_logger.console_handler = console_handler
    if not hasattr(msst_logger, "file_handler"):
        msst_logger.file_handler = logging.NullHandler()

    with _temporary_cwd(msst_dir):
        from inference.msst_infer import MSSeparator

        _disable_progress_bars()
        separator = MSSeparator(
            model_type=config.model_type,
            config_path=config_path,
            model_path=model_path,
            device=config.device,
            device_ids=device_ids,
            output_format="wav",
        )
    _SEPARATORS[cache_key] = separator
    logger.info("Loaded MSST speech separator: %s", model_path)
    return separator


def release_msst_separators(device_id: int | None = None) -> int:
    """Release cached separators, optionally only those resident on one GPU."""

    keys = [
        key
        for key in _SEPARATORS
        if device_id is None or key[-1] == (int(device_id),)
    ]
    for key in keys:
        separator = _SEPARATORS.pop(key)
        model = getattr(separator, "model", None)
        if model is not None and hasattr(model, "to"):
            model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return len(keys)


def _ensure_channels_first(waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
    if waveform.ndim == 1:
        return waveform.unsqueeze(0), 1
    if waveform.ndim == 2:
        return waveform, 2
    raise ValueError(f"expected 1D or 2D waveform, got shape={tuple(waveform.shape)}")


def _match_length(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    current_samples = int(waveform.shape[-1])
    if current_samples > target_samples:
        return waveform[..., :target_samples]
    if current_samples < target_samples:
        return torch.nn.functional.pad(waveform, (0, target_samples - current_samples))
    return waveform


def extract_speech_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    config: MSSTSpeechConfig,
) -> torch.Tensor:
    """Extract the speech stem and restore the input shape, rate, device and dtype."""

    if int(sample_rate) <= 0:
        raise ValueError(f"invalid sample_rate={sample_rate}")
    if waveform.numel() == 0 or waveform.shape[-1] <= 1:
        return waveform

    started = time.perf_counter()
    source_device = waveform.device
    source_dtype = waveform.dtype
    source, original_dims = _ensure_channels_first(
        waveform.detach().to(device="cpu", dtype=torch.float32)
    )
    target_samples = int(source.shape[-1])
    source_channels = int(source.shape[0])
    target_rate = int(sample_rate)
    msst_rate = int(config.sample_rate)

    msst_input = source
    if target_rate != msst_rate:
        msst_input = torchaudio.functional.resample(msst_input, target_rate, msst_rate)
    if msst_input.shape[0] == 1:
        msst_input = msst_input.repeat(2, 1)
    elif msst_input.shape[0] > 2:
        msst_input = msst_input.mean(dim=0, keepdim=True).repeat(2, 1)

    separator = get_msst_separator(config)
    with torch.inference_mode(), _temporary_cwd(str(Path(config.msst_dir).expanduser().resolve())):
        results = separator.separate(msst_input.numpy().astype(np.float32, copy=False))
    if "speech" not in results:
        raise RuntimeError(f"MSST result has no speech stem: {list(results)}")

    speech_array = np.asarray(results["speech"], dtype=np.float32)
    if speech_array.ndim == 1:
        speech = torch.from_numpy(speech_array).unsqueeze(0)
    elif speech_array.shape[0] == 2 and speech_array.shape[1] != 2:
        speech = torch.from_numpy(speech_array)
    else:
        speech = torch.from_numpy(speech_array).transpose(0, 1).contiguous()
    if target_rate != msst_rate:
        speech = torchaudio.functional.resample(speech, msst_rate, target_rate)
    speech = _match_length(speech, target_samples)

    if source_channels == 1:
        speech = speech.mean(dim=0, keepdim=True)
    elif source_channels == 2:
        speech = speech.repeat(2, 1) if speech.shape[0] == 1 else speech[:2]
    else:
        speech = speech.mean(dim=0, keepdim=True).repeat(source_channels, 1)
    speech = speech.to(device=source_device, dtype=source_dtype)

    logger.info(
        "MSST speech extraction: audio=%.2fs elapsed=%.2fs",
        target_samples / float(target_rate),
        time.perf_counter() - started,
    )
    return speech.squeeze(0) if original_dims == 1 else speech
