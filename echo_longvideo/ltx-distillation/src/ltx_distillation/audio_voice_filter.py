"""Production-aligned MSST voice filtering for multi-shot audio memory."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import torch

from ltx_distillation.msst_speech_filter import (
    MSSTSpeechConfig,
    extract_speech_waveform,
    release_msst_separators,
    validate_msst_config,
)


_MSST_LOCK = threading.Lock()


@dataclass(frozen=True)
class VoiceFilterConfig:
    enabled: bool = True
    backend: str = "msst_speech"
    min_output_rms: float = 0.004
    msst_dir: str = "third_party/MSST-WebUI"
    msst_model_path: str = "checkpoints/msst/model_bandit_plus_dnr_sdr_11.47.chpt"
    msst_config_path: str = (
        "third_party/MSST-WebUI/configs_backup/multi_stem_models/"
        "model_bandit_plus_dnr_sdr_11.47.chpt.yaml"
    )
    msst_model_type: str = "bandit"
    msst_sample_rate: int = 44100
    msst_device: str = "auto"
    msst_local_rank_env: str = "LOCAL_RANK"

    def as_msst_config(self) -> MSSTSpeechConfig:
        return MSSTSpeechConfig(
            msst_dir=self.msst_dir,
            model_path=self.msst_model_path,
            config_path=self.msst_config_path,
            model_type=self.msst_model_type,
            sample_rate=int(self.msst_sample_rate),
            device=self.msst_device,
            local_rank_env=self.msst_local_rank_env,
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.backend.strip().lower() != "msst_speech":
            raise ValueError("Echo 1.5 production memory requires backend=msst_speech")
        validate_msst_config(self.as_msst_config())


def _as_channels_first(waveform: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(waveform).detach().cpu().float()
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError(f"expected one audio sample, got shape={tuple(value.shape)}")
        value = value[0]
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"expected audio [channels, samples], got {tuple(value.shape)}")
    return value.contiguous()


def filter_voice_only(
    waveform: Optional[torch.Tensor],
    sample_rate: int,
    config: VoiceFilterConfig,
) -> Optional[torch.Tensor]:
    """Extract the MSST speech stem while preserving waveform duration and shape."""

    if waveform is None or not config.enabled:
        return waveform
    waveform = _as_channels_first(waveform)
    if waveform.numel() == 0 or waveform.shape[-1] <= 1:
        return waveform
    config.validate()

    # MSSeparator is cached but does not document concurrent inference safety.
    with _MSST_LOCK:
        speech = extract_speech_waveform(
            waveform,
            int(sample_rate),
            config.as_msst_config(),
        )
    speech = _as_channels_first(speech).to(dtype=waveform.dtype)
    rms = speech.float().square().mean().sqrt()
    if float(rms.item()) < max(0.0, float(config.min_output_rms)):
        return torch.zeros_like(waveform)
    return speech.contiguous()


def release_voice_filter_device(device_id: int) -> int:
    """Release one worker's cached MSST weights without racing extraction."""

    with _MSST_LOCK:
        return release_msst_separators(device_id=device_id)
