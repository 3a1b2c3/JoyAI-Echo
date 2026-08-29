"""Lazy local Echo 1.5 inference runtime used by the API scheduler.

This module intentionally avoids importing Torch or the model stack at import
time. The FastAPI control plane and its scheduler tests can therefore run on a
CPU-only machine; GPU dependencies are loaded only when a worker starts.
"""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from r2v_schema import R2VRequest, load_r2v_request


GIB = 1024**3
REPO_ROOT = Path(__file__).resolve().parent.parent

# Whole-device memory estimates validated with a 241-frame,
# 1280x736 request. ``auto`` uses them only to choose the residency strategy;
# live free memory remains the admission authority. On-disk size is not a good
# proxy here because packed weight size and runtime workspace differ by backend.
RESIDENT_DIT_PEAK_BYTES = {
    "bf16": int(45.2 * GIB),
    "fp8": int(31.6 * GIB),
    "fp4": int(24.4 * GIB),
}
SWAP_DIT_PEAK_BYTES = {
    "bf16": int(15.44 * GIB),
    "fp8": int(15.92 * GIB),
    "fp4": int(14.98 * GIB),
}


def resolve_dit_residency(
    requested: str,
    *,
    precision: str,
    snapshot: "ResourceSnapshot",
    headroom_fraction: float,
) -> dict[str, Any]:
    """Select full GPU residency or layerwise swap from live device capacity."""

    requested = requested.strip().lower()
    if requested not in {"auto", "resident", "swap"}:
        raise ValueError("ECHO_DIT_RESIDENCY must be auto, resident, or swap")
    if precision not in RESIDENT_DIT_PEAK_BYTES:
        raise ValueError(f"unsupported checkpoint precision for residency: {precision}")

    measured_peak = RESIDENT_DIT_PEAK_BYTES[precision]
    reserve = max(int(snapshot.gpu_total_bytes * headroom_fraction), 2 * GIB)
    required_free = measured_peak + reserve
    selected = (
        requested
        if requested != "auto"
        else ("resident" if snapshot.gpu_free_bytes >= required_free else "swap")
    )
    return {
        "requested": requested,
        "selected": selected,
        "precision": precision,
        "measured_resident_peak_bytes": measured_peak,
        "headroom_bytes": reserve,
        "required_free_bytes": required_free,
        "observed_free_bytes": snapshot.gpu_free_bytes,
        "observed_total_bytes": snapshot.gpu_total_bytes,
    }


@dataclass(frozen=True)
class ResourceSnapshot:
    gpu_id: int
    gpu_name: str
    gpu_free_bytes: int
    gpu_total_bytes: int
    ram_available_bytes: int
    ram_total_bytes: int
    observed_at: float

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "gpu_free_gib": round(self.gpu_free_bytes / GIB, 3),
                "gpu_total_gib": round(self.gpu_total_bytes / GIB, 3),
                "ram_available_gib": round(self.ram_available_bytes / GIB, 3),
                "ram_total_gib": round(self.ram_total_bytes / GIB, 3),
            }
        )
        return payload


def resolve_gpu_ids(configured: str) -> list[int]:
    """Resolve logical CUDA device IDs after CUDA_VISIBLE_DEVICES is applied."""

    import torch

    count = torch.cuda.device_count()
    if not torch.cuda.is_available() or count <= 0:
        raise RuntimeError("the local R2V server requires at least one CUDA GPU")
    text = configured.strip()
    gpu_ids = (
        list(range(count))
        if not text
        else [int(value.strip()) for value in text.split(",")]
    )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("ECHO_GPU_IDS contains duplicate logical device IDs")
    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= count]
    if invalid:
        raise ValueError(
            f"ECHO_GPU_IDS contains unavailable devices: {invalid}; visible={count}"
        )
    return gpu_ids


def probe_resources(gpu_id: int) -> ResourceSnapshot:
    import psutil
    import torch

    with torch.cuda.device(gpu_id):
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
        gpu_name = torch.cuda.get_device_name(gpu_id)
    memory = psutil.virtual_memory()
    return ResourceSnapshot(
        gpu_id=gpu_id,
        gpu_name=gpu_name,
        gpu_free_bytes=int(free_bytes),
        gpu_total_bytes=int(total_bytes),
        ram_available_bytes=int(memory.available),
        ram_total_bytes=int(memory.total),
        observed_at=time.time(),
    )


class LocalModelRuntime:
    """Own one GPU's staged condition/generator/VAE model lifecycle."""

    def __init__(
        self,
        *,
        gpu_id: int,
        config_path: Path,
        checkpoint: str | None,
        conditioning_cache_dir: Path,
        requests_root: Path,
        output_root: Path,
        dit_residency: str = "auto",
        gpu_headroom_fraction: float = 0.05,
        ram_headroom_fraction: float = 0.10,
    ) -> None:
        self.gpu_id = int(gpu_id)
        self.config_path = config_path.expanduser().resolve()
        if checkpoint:
            checkpoint_path = Path(checkpoint).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = REPO_ROOT / checkpoint_path
            self.checkpoint = str(checkpoint_path.resolve())
        else:
            self.checkpoint = None
        self.conditioning_cache_dir = conditioning_cache_dir.expanduser().resolve()
        self.requests_root = requests_root.expanduser().resolve()
        self.output_root = output_root.expanduser().resolve()
        self.dit_residency = dit_residency.strip().lower()
        if self.dit_residency not in {"auto", "resident", "swap"}:
            raise ValueError("ECHO_DIT_RESIDENCY must be auto, resident, or swap")
        self.gpu_headroom_fraction = float(gpu_headroom_fraction)
        self.ram_headroom_fraction = float(ram_headroom_fraction)
        self.engine = None
        self.residency_plan: dict[str, Any] | None = None
        self.state = "unloaded"
        self.last_used_at = time.monotonic()

    @property
    def model_loaded(self) -> bool:
        return self.engine is not None and self.engine.generator is not None

    @property
    def weights_loaded(self) -> bool:
        return self.engine is not None and any(
            module is not None
            for module in (
                self.engine.generator,
                self.engine.video_vae,
                self.engine.audio_vae,
            )
        )

    @property
    def model_location(self) -> str:
        if self.engine is None:
            return "unloaded"
        return str(getattr(self.engine, "generator_location", "unloaded"))

    def can_retain_generator_on_cpu(
        self, ram_available_bytes: int, ram_reserve_bytes: int
    ) -> bool:
        """Choose CPU retention only when it will not consume the RAM reserve."""

        if not self.model_loaded:
            return False
        if self.model_location == "cpu":
            return ram_available_bytes >= ram_reserve_bytes
        retained_bytes = int(self.engine.generation_storage_bytes())
        return ram_available_bytes - retained_bytes >= ram_reserve_bytes

    def conditioning_generator_policy(self, snapshot: ResourceSnapshot) -> str:
        """Prefer GPU residency across a cache miss, then CPU, then release."""

        if not self.model_loaded:
            return "release"
        engine = self._ensure_engine()
        if self.model_location.startswith("cuda"):
            gemma_bytes = sum(
                path.stat().st_size
                for path in engine.gemma_path.rglob("*.safetensors")
                if path.is_file()
            )
            # Gemma language weights, Echo connectors and conditioning
            # activations are all transient. Derive their budget from the two
            # checkpoint sources instead of a device-specific GiB constant.
            condition_gpu_bytes = (
                int(gemma_bytes * 1.10)
                + int(engine.model_checkpoint.stat().st_size * 0.15)
                + int(snapshot.gpu_total_bytes * self.gpu_headroom_fraction)
            )
            if gemma_bytes > 0 and snapshot.gpu_free_bytes >= condition_gpu_bytes:
                return "gpu"
        ram_reserve = int(snapshot.ram_total_bytes * self.ram_headroom_fraction)
        if self.can_retain_generator_on_cpu(snapshot.ram_available_bytes, ram_reserve):
            return "cpu"
        return "release"

    def admission_requirements(
        self, snapshot: ResourceSnapshot
    ) -> tuple[int, int, dict[str, Any]]:
        """Estimate the next load/compute working set from the selected release."""

        engine = self._ensure_engine()
        component_file_bytes = int(engine.model_checkpoint.stat().st_size)
        modelopt_path = engine.release_checkpoint.modelopt_path
        generator_file_bytes = (
            int(modelopt_path.stat().st_size)
            if modelopt_path is not None
            else component_file_bytes
        )
        release_file_bytes = component_file_bytes + (
            generator_file_bytes if modelopt_path is not None else 0
        )
        precision = str(engine.release_checkpoint.precision)
        gpu_reserve = int(snapshot.gpu_total_bytes * self.gpu_headroom_fraction)
        ram_reserve = int(snapshot.ram_total_bytes * self.ram_headroom_fraction)

        if engine.config.dit_layerwise_offload:
            # Offload profiles are dominated by activations plus prefetched
            # blocks rather than the complete checkpoint size. Use the measured
            # whole-device peak plus the same safety reserve as auto selection.
            gpu_required = SWAP_DIT_PEAK_BYTES[precision] + max(gpu_reserve, 2 * GIB)
            mode = "layerwise_offload"
        elif self.model_loaded and self.model_location.startswith("cuda"):
            gpu_required = int(snapshot.gpu_total_bytes * 0.20)
            mode = "warm_gpu"
        else:
            if self.model_loaded:
                model_bytes = int(engine.generation_storage_bytes())
            else:
                model_bytes = generator_file_bytes
            gpu_required = model_bytes + gpu_reserve
            if self.residency_plan is not None:
                gpu_required = max(
                    gpu_required,
                    int(self.residency_plan["required_free_bytes"]),
                )
            mode = "load_from_cpu" if self.model_loaded else "cold_load"

        # Cold loading maps the independent release artifacts and may create an
        # offload pool. Warm CPU/GPU weights are already reflected in
        # psutil.available and need only the proportional system reserve.
        ram_required = ram_reserve
        if not self.weights_loaded:
            ram_required += int(release_file_bytes * 1.25)
        residency = dict(self.residency_plan or {})
        for key, value in tuple(residency.items()):
            if key.endswith("_bytes"):
                residency[f"{key.removesuffix('_bytes')}_gib"] = round(value / GIB, 3)
                del residency[key]
        return (
            gpu_required,
            ram_required,
            {
                "mode": mode,
                "precision": precision,
                "release_files_gib": round(release_file_bytes / GIB, 3),
                "model_location": self.model_location,
                "dit_residency": residency,
            },
        )

    def _ensure_engine(self):
        if self.engine is not None:
            return self.engine
        from inference import InferenceConfig, InferenceEngine
        from ltx_distillation.release_checkpoint import resolve_release_checkpoint

        overrides: dict[str, Any] = {
            "device": f"cuda:{self.gpu_id}",
            "requests_dir": str(self.requests_root),
            "conditioning_cache_dir": str(self.conditioning_cache_dir),
            "output_root": str(self.output_root),
        }
        if self.checkpoint:
            overrides["checkpoint"] = self.checkpoint
        config = InferenceConfig(self.config_path, **overrides)
        release = resolve_release_checkpoint(config.checkpoint)
        snapshot = probe_resources(self.gpu_id)
        self.residency_plan = resolve_dit_residency(
            self.dit_residency,
            precision=release.precision,
            snapshot=snapshot,
            headroom_fraction=self.gpu_headroom_fraction,
        )
        config.dit_layerwise_offload = self.residency_plan["selected"] == "swap"
        print(
            "[Server] DiT residency "
            f"requested={self.residency_plan['requested']} "
            f"selected={self.residency_plan['selected']} "
            f"precision={release.precision} "
            f"free={snapshot.gpu_free_bytes / GIB:.2f}GiB "
            f"required_for_resident={self.residency_plan['required_free_bytes'] / GIB:.2f}GiB",
            flush=True,
        )
        if config.voice_filter.msst_device in {"auto", "cuda"}:
            config.voice_filter = replace(
                config.voice_filter,
                msst_device=f"cuda:{self.gpu_id}",
            )
        self.engine = InferenceEngine(config)
        return self.engine

    def load_request(self, request_file: Path) -> R2VRequest:
        engine = self._ensure_engine()
        config = engine.config
        return load_r2v_request(
            request_file,
            default_num_frames=config.num_frames,
            default_width=config.video_width,
            default_height=config.video_height,
            default_seed=config.seed,
            prompt_max_chars=config.prompt_max_chars,
        )

    def prepare_conditions(
        self,
        request_files: list[Path],
        requests: list[R2VRequest],
        stage_callback,
        *,
        generator_policy: str = "cpu",
    ) -> dict[Path, Any]:
        """Load valid caches, or make GPU room and batch cache misses."""

        engine = self._ensure_engine()
        self.state = "conditioning_cache_lookup"
        stage_callback(self.state)
        try:
            return engine.prepare_conditions(request_files, requests)
        except (FileNotFoundError, ValueError):
            self.state = (
                "staging_generator_on_cpu"
                if generator_policy == "cpu" and self.model_loaded
                else (
                    "keeping_generator_on_gpu"
                    if generator_policy == "gpu" and self.model_loaded
                    else "unloading_for_conditioning"
                )
            )
            stage_callback(self.state)
            engine.stage_generator_for_conditioning(
                generator_policy if self.model_loaded else "release"
            )
            self.state = "conditioning"
            stage_callback(self.state)
            try:
                return engine.prepare_conditions(
                    request_files, requests, encode_cache_misses=True
                )
            finally:
                from ltx_distillation.audio_voice_filter import (
                    release_voice_filter_device,
                )

                release_voice_filter_device(self.gpu_id)

    def load_cached_conditions(
        self,
        request_files: list[Path],
        requests: list[R2VRequest],
        stage_callback,
    ) -> dict[Path, Any] | None:
        """Fast path that never loads encoders and never mutates model residency."""

        engine = self._ensure_engine()
        self.state = "conditioning_cache_lookup"
        stage_callback(self.state)
        try:
            return engine.prepare_conditions(request_files, requests)
        except (FileNotFoundError, ValueError):
            return None

    def run(
        self,
        request_file: Path,
        request: R2VRequest,
        output_dir: Path,
        bundle,
        stage_callback,
    ):
        engine = self._ensure_engine()
        self.state = "loading_generator"
        engine.load_generator()
        result = engine.run_request(
            request_file,
            request,
            output_dir,
            bundle,
            stage_callback=self._stage_callback(stage_callback),
            decode_generator_policy=self._decode_generator_policy,
        )
        self.state = "ready"
        self.last_used_at = time.monotonic()
        return result

    def _decode_generator_policy(self) -> str:
        """Keep weights on the fastest safe tier for the VAE decode stage."""

        engine = self._ensure_engine()
        snapshot = probe_resources(self.gpu_id)
        residency_plan = getattr(self, "residency_plan", None)
        if (
            residency_plan is not None
            and residency_plan["selected"] == "resident"
            and engine.dit_offload is None
            and engine.generator_location.startswith("cuda")
        ):
            # A resident profile keeps DiT in device memory during tiled decode
            # and between requests. The engine catches a real decoder-placement
            # OOM and falls back to releasing DiT, so contention remains safe.
            return "gpu"
        # Layerwise offload already owns a CPU pool; deactivation merely returns
        # the active blocks to that pool and does not allocate another copy.
        if engine.dit_offload is not None:
            return "cpu"
        gpu_decode_headroom = int(
            snapshot.gpu_total_bytes * max(self.gpu_headroom_fraction, 0.20)
        )
        if snapshot.gpu_free_bytes >= gpu_decode_headroom:
            return "gpu"
        if engine.generator_location == "cpu":
            return "cpu"
        retained_bytes = int(engine.generation_storage_bytes())
        ram_reserve = int(snapshot.ram_total_bytes * self.ram_headroom_fraction)
        if snapshot.ram_available_bytes - retained_bytes >= ram_reserve:
            return "cpu"
        return "release"

    def _stage_callback(self, callback):
        def update(stage: str) -> None:
            self.state = stage
            callback(stage)

        return update

    def unload(self) -> None:
        self.state = "unloading"
        if self.engine is not None:
            self.engine.unload_generator()
        self.engine = None
        gc.collect()
        try:
            import torch

            with torch.cuda.device(self.gpu_id):
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
        self.state = "unloaded"
        self.last_used_at = time.monotonic()
