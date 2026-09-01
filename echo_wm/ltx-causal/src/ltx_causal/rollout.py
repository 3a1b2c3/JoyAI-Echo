"""Autoregressive causal rollout for joint video and audio generation."""

from __future__ import annotations

import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ltx_core.types import VIDEO_SCALE_FACTORS

# Opt-in (ECHO_WM_PROFILE_CACHE=1): profile exactly one cache-update
# forward() call (see the bottom of _generate_av_blocks below) with
# torch.profiler, to find out what's actually expensive in it -- this call
# costs the same ~0.5-0.6s regardless of attention-window size or
# resolution changes (see TROUBLESHOOTING.md item -14/-16 area), meaning
# whatever's expensive isn't the windowed self-attention those levers
# touch. Rather than guess again, get a real operator-level breakdown.
# Profiling block 3 (not block 0, which pays extra one-time warmup-ish
# cost) once, not every block, to keep this cheap when disabled and
# low-overhead for the one profiled call.
_PROFILE_CACHE_UPDATE = os.environ.get("ECHO_WM_PROFILE_CACHE", "0") == "1"
_PROFILE_BLOCK_INDEX = 3
_cache_profiled = False

# Opt-in (ECHO_WM_GRAPH_CAPTURE_TEST=1): feasibility-only check -- does
# torch.cuda.graph() capture even succeed for one real cache-update
# forward() call, at a later (steady-state windowing) block? Diagnostic
# only, does not change real generation behavior (runs an extra,
# discarded capture attempt alongside the normal call). Known likely
# blocker going in: update_kv_cache's searchsorted(...).item() calls
# (added to remove nonzero(), see TROUBLESHOOTING.md) are themselves a
# CPU-GPU sync point, which graph capture forbids entirely -- this test's
# real value is confirming that (or being pleasantly surprised) with a
# real error message instead of more speculation.
_GRAPH_CAPTURE_TEST = os.environ.get("ECHO_WM_GRAPH_CAPTURE_TEST", "0") == "1"
_GRAPH_CAPTURE_BLOCK_INDEX = 5
_graph_capture_tested = False

from ltx_core.model.transformer.modality import Modality

from .cache import configure_bounded_caches
from .causal_wrapper import CausalModelWrapper
from .scheduling import (
    DEFAULT_CAUSAL_TIMESTEPS,
    causal_audio_blocks,
    causal_audio_frames,
    causal_video_blocks,
    resolve_causal_sigmas,
)

Block = tuple[int, int]
# Called after each post-prefix block is written into the output buffers.
# Args: (block_index, total_blocks, video_block, video_output_so_far,
# audio_output_so_far) — block_index/total_blocks are 0-indexed among the
# streamed (non-prefix) blocks; video_block is this block's (start, end)
# latent-frame range (the newly-completed part); video_output_so_far/
# audio_output_so_far are the FULL-sequence output buffers with real content
# through this block and zeros beyond (same shape patchify/unpatchify
# expects — there is no per-slice unpatchify).
OnBlock = Callable[[int, int, Block, torch.Tensor, torch.Tensor], None]


def _modality(
    latent: torch.Tensor,
    positions: torch.Tensor,
    context: torch.Tensor,
    sigma: float,
    *,
    context_mask: torch.Tensor | None,
) -> Modality:
    return Modality(
        latent=latent,
        sigma=torch.ones(latent.shape[0], device=latent.device, dtype=latent.dtype),
        timesteps=torch.full(latent.shape[:2], sigma, device=latent.device, dtype=latent.dtype),
        positions=positions,
        context=context,
        context_mask=context_mask,
    )


def _random_like(reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _advance_sample(
    denoised: torch.Tensor,
    next_sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    return (1 - next_sigma) * denoised + next_sigma * _random_like(denoised, generator)


@dataclass(frozen=True)
class _BlockForward:
    """Bind rollout-wide context and expose a compact per-block model call."""

    wrapper: CausalModelWrapper
    caches: list[dict]
    video_positions: torch.Tensor
    audio_positions: torch.Tensor
    video_context: torch.Tensor
    audio_context: torch.Tensor
    context_mask: torch.Tensor | None
    action_cond: dict[str, torch.Tensor]

    @classmethod
    def create(
        cls,
        *,
        wrapper: CausalModelWrapper,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        video_positions: torch.Tensor,
        audio_positions: torch.Tensor,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        context_mask: torch.Tensor | None,
        action_cond: dict[str, torch.Tensor],
    ) -> _BlockForward:
        caches = wrapper.init_caches(
            batch_size=clean_video.shape[0],
            video_frames=video_positions.shape[2] // wrapper.patches_per_frame,
            audio_frames=clean_audio.shape[1],
            text_seq_len=video_context.shape[1],
            device=clean_video.device,
            dtype=clean_video.dtype,
        )
        configure_bounded_caches(
            wrapper,
            caches,
            video_positions,
            audio_positions,
            action_cond,
            clean_video.dtype,
        )
        return cls(
            wrapper=wrapper,
            caches=caches,
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_context=video_context,
            audio_context=audio_context,
            context_mask=context_mask,
            action_cond=action_cond,
        )

    def __call__(
        self,
        video_latent: torch.Tensor,
        video_block: Block,
        video_sigma: float,
        audio_latent: torch.Tensor | None = None,
        audio_block: Block = (0, 0),
        audio_sigma: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        video_start, video_end = video_block
        audio_start, audio_end = audio_block
        patches_per_frame = self.wrapper.patches_per_frame
        sliced_action = {
            key: value[:, video_start:video_end]
            if key in {"ucpe_viewmats", "ucpe_Ks"}
            else value
            for key, value in self.action_cond.items()
        }
        video = _modality(
            video_latent,
            self.video_positions[
                :, :, video_start * patches_per_frame : video_end * patches_per_frame
            ],
            self.video_context,
            video_sigma,
            context_mask=self.context_mask,
        )
        audio = None
        if audio_latent is not None:
            audio = _modality(
                audio_latent,
                self.audio_positions[:, :, audio_start:audio_end],
                self.audio_context,
                video_sigma if audio_sigma is None else audio_sigma,
                context_mask=self.context_mask,
            )
        return self.wrapper(
            video,
            audio,
            sliced_action,
            self.caches,
            video_start,
            audio_start,
        )


@dataclass
class _RolloutBuffers:
    """Noise sources and output tensors shared by all rollout blocks."""

    initial_video: torch.Tensor
    initial_audio: torch.Tensor
    video_output: torch.Tensor
    audio_output: torch.Tensor
    clean_image: torch.Tensor

    @classmethod
    def create(
        cls,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        patches_per_frame: int,
        generator: torch.Generator,
    ) -> _RolloutBuffers:
        clean_image = clean_video[:, :patches_per_frame]
        video_output = torch.zeros_like(clean_video)
        video_output[:, :patches_per_frame] = clean_image
        return cls(
            initial_video=_random_like(clean_video, generator),
            initial_audio=_random_like(clean_audio, generator),
            video_output=video_output,
            audio_output=torch.zeros_like(clean_audio),
            clean_image=clean_image,
        )


def _denoise_audio_prefix(
    forward: _BlockForward,
    clean_image: torch.Tensor,
    initial_audio: torch.Tensor,
    video_block: Block,
    audio_block: Block,
    sigmas: list[float],
    generator: torch.Generator,
) -> torch.Tensor:
    audio_start, audio_end = audio_block
    audio_sample = initial_audio[:, audio_start:audio_end]
    for step, sigma in enumerate(sigmas):
        _, denoised_audio = forward(
            clean_image,
            video_block,
            0.0,
            audio_sample,
            audio_block,
            sigma,
        )
        if denoised_audio is None:
            raise RuntimeError("causal AV model returned no audio")
        audio_sample = (
            denoised_audio
            if step == len(sigmas) - 1
            else _advance_sample(denoised_audio, sigmas[step + 1], generator)
        )
    return audio_sample


def _denoise_av_block(
    forward: _BlockForward,
    initial_video: torch.Tensor,
    initial_audio: torch.Tensor,
    video_block: Block,
    audio_block: Block,
    sigmas: list[float],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    video_start, video_end = video_block
    audio_start, audio_end = audio_block
    patches_per_frame = forward.wrapper.patches_per_frame
    video_sample = initial_video[
        :, video_start * patches_per_frame : video_end * patches_per_frame
    ]
    audio_sample = initial_audio[:, audio_start:audio_end]

    for step, sigma in enumerate(sigmas):
        denoised_video, denoised_audio = forward(
            video_sample,
            video_block,
            sigma,
            audio_sample,
            audio_block,
            sigma,
        )
        if denoised_audio is None:
            raise RuntimeError("causal AV model returned no audio")
        if step == len(sigmas) - 1:
            video_sample, audio_sample = denoised_video, denoised_audio
        else:
            next_sigma = sigmas[step + 1]
            video_sample = _advance_sample(denoised_video, next_sigma, generator)
            audio_sample = _advance_sample(denoised_audio, next_sigma, generator)
    return video_sample, audio_sample


def _generate_audio_prefix(
    forward: _BlockForward,
    buffers: _RolloutBuffers,
    video_block: Block,
    audio_block: Block,
    sigmas: list[float],
    generator: torch.Generator,
) -> None:
    """Commit the image sink, generate its audio prefix, then refresh caches."""
    forward(buffers.clean_image, video_block, 0.0)
    audio_prefix = _denoise_audio_prefix(
        forward,
        buffers.clean_image,
        buffers.initial_audio,
        video_block,
        audio_block,
        sigmas,
        generator,
    )
    audio_start, audio_end = audio_block
    buffers.audio_output[:, audio_start:audio_end] = audio_prefix
    forward(buffers.clean_image, video_block, 0.0, audio_prefix, audio_block, 0.0)


def _generate_av_blocks(
    forward: _BlockForward,
    buffers: _RolloutBuffers,
    video_blocks: list[Block],
    audio_blocks: list[Block],
    sigmas: list[float],
    generator: torch.Generator,
    on_block: OnBlock | None = None,
) -> None:
    """Generate, store, and cache all blocks after the image sink."""
    patches_per_frame = forward.wrapper.patches_per_frame
    total_blocks = len(video_blocks) - 1
    for block_index, (video_block, audio_block) in enumerate(
        zip(video_blocks[1:], audio_blocks[1:], strict=True)
    ):
        t_block_start = time.time()
        video_sample, audio_sample = _denoise_av_block(
            forward,
            buffers.initial_video,
            buffers.initial_audio,
            video_block,
            audio_block,
            sigmas,
            generator,
        )
        t_denoise = time.time() - t_block_start
        video_start, video_end = video_block
        audio_start, audio_end = audio_block
        buffers.video_output[
            :, video_start * patches_per_frame : video_end * patches_per_frame
        ] = video_sample
        buffers.audio_output[:, audio_start:audio_end] = audio_sample
        t_callback = 0.0
        if on_block is not None:
            t_callback_start = time.time()
            on_block(block_index, total_blocks, video_block, buffers.video_output, buffers.audio_output)
            t_callback = time.time() - t_callback_start
        t_cache_start = time.time()
        global _cache_profiled
        if _PROFILE_CACHE_UPDATE and block_index == _PROFILE_BLOCK_INDEX and not _cache_profiled:
            _cache_profiled = True
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                forward(video_sample, video_block, 0.0, audio_sample, audio_block, 0.0)
            print(f"\n[profile] cache-update forward() at block {block_index}, "
                  f"top ops by CUDA time:\n", flush=True)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25), flush=True)
            trace_path = "cache_update_profile.json"
            prof.export_chrome_trace(trace_path)
            print(f"[profile] full trace exported to {trace_path} "
                  f"(open in chrome://tracing or https://ui.perfetto.dev for details)\n", flush=True)
        else:
            forward(video_sample, video_block, 0.0, audio_sample, audio_block, 0.0)
        t_cache = time.time() - t_cache_start

        global _graph_capture_tested
        if _GRAPH_CAPTURE_TEST and block_index == _GRAPH_CAPTURE_BLOCK_INDEX and not _graph_capture_tested:
            _graph_capture_tested = True
            print(f"[graph-test] attempting torch.cuda.graph() capture of cache-update "
                  f"forward() at block {block_index}...", flush=True)
            # Safe to call forward() extra times here with the *same*
            # (video_sample, video_block): update_kv_cache is deliberately
            # idempotent for repeated same-range writes (it's designed to
            # handle repeated denoising-step overwrites of the same block --
            # see its docstring), so these extra warmup/capture/replay calls
            # converge to the same final cache state as the one real call
            # above, not corrupt it.
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):  # warmup, per torch.cuda.graph's own recommendation
                        forward(video_sample, video_block, 0.0, audio_sample, audio_block, 0.0)
                torch.cuda.current_stream().wait_stream(s)

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    forward(video_sample, video_block, 0.0, audio_sample, audio_block, 0.0)
                print("[graph-test] CAPTURE SUCCEEDED. Attempting one replay...", flush=True)
                g.replay()
                torch.cuda.synchronize()
                print("[graph-test] REPLAY SUCCEEDED. This forward() call is "
                      "graph-capturable -- real integration is worth pursuing.", flush=True)
            except Exception as exc:  # noqa: BLE001 - diagnostic, report whatever breaks
                print(f"[graph-test] CAPTURE/REPLAY FAILED: {type(exc).__name__}: {exc}", flush=True)
                print("[graph-test] Full traceback (to find the exact operation "
                      "responsible, not just the generic capture-time error):", flush=True)
                traceback.print_exc()
                print("[graph-test] Diagnostic only -- real generation is unaffected by "
                      "this failure (extra calls above were discarded/idempotent).", flush=True)
        # Real-time target: video_chunk_size is in *latent* frames, not
        # decoded output frames -- the VAE has an 8x temporal compression
        # (VIDEO_SCALE_FACTORS.time, ltx_core.types.py), so a
        # video_chunk_size=3 block actually decodes to 24 output frames,
        # not 3. Confirmed on real hardware (user-reported: "each block is
        # 24 frames") -- an earlier version of this line divided by
        # video_chunk_size directly, understating the real-time target by
        # 8x and overstating how far off real-time every block was by the
        # same factor. fps isn't threaded into this function (only
        # configs/inference_wm_causal.yaml has it) -- 16.0 is that config's
        # current value; update this if fps changes.
        fps = 16.0
        output_frames_per_block = forward.wrapper.cache.video_chunk_size * VIDEO_SCALE_FACTORS.time
        target_s = output_frames_per_block / fps
        block_total = time.time() - t_block_start
        print(f"[rollout] block {block_index}/{total_blocks}: denoise={t_denoise:.3f}s "
              f"callback={t_callback:.3f}s cache={t_cache:.3f}s total={block_total:.3f}s "
              f"(real-time target={target_s:.3f}s, {block_total / target_s:.1f}x too slow)",
              flush=True)


@torch.no_grad()
def causal_rollout(  # noqa: PLR0913
    *,
    wrapper: CausalModelWrapper,
    clean_video: torch.Tensor,
    clean_audio: torch.Tensor,
    video_positions: torch.Tensor,
    audio_positions: torch.Tensor,
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    context_mask: torch.Tensor | None,
    action_cond: dict[str, torch.Tensor],
    seed: int,
    timesteps: tuple[int, ...] | list[int] = DEFAULT_CAUSAL_TIMESTEPS,
    on_block: OnBlock | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate all audio-video blocks and refresh caches with clean outputs."""
    patches_per_frame = wrapper.patches_per_frame
    video_frames = video_positions.shape[2] // patches_per_frame
    chunk_size = wrapper.cache.video_chunk_size
    video_blocks = causal_video_blocks(video_frames, chunk_size)
    audio_blocks = causal_audio_blocks(video_frames, chunk_size)
    if clean_audio.shape[1] != causal_audio_frames(video_frames, chunk_size):
        raise ValueError("audio latent length does not match the causal AV block layout")

    sigmas = resolve_causal_sigmas(timesteps)
    generator = torch.Generator(device=clean_video.device).manual_seed(seed)
    buffers = _RolloutBuffers.create(
        clean_video,
        clean_audio,
        patches_per_frame,
        generator,
    )
    forward = _BlockForward.create(
        wrapper=wrapper,
        clean_video=clean_video,
        clean_audio=clean_audio,
        video_positions=video_positions,
        audio_positions=audio_positions,
        video_context=video_context,
        audio_context=audio_context,
        context_mask=context_mask,
        action_cond=action_cond,
    )

    _generate_audio_prefix(
        forward,
        buffers,
        video_blocks[0],
        audio_blocks[0],
        sigmas,
        generator,
    )
    _generate_av_blocks(forward, buffers, video_blocks, audio_blocks, sigmas, generator, on_block)
    return buffers.video_output, buffers.audio_output
