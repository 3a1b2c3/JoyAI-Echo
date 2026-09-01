#!/usr/bin/env python3
"""Gradio app for Echo-WM world model inference.

Features:
  - Image upload or selection from examples
  - Six-field cinematic prompt from PROMPT_SKILL.md
  - Action string input with presets (WASD + camera controls)
  - Genie-style HUD overlay (on by default)
  - Video + audio generation

Run:
    CUDA_VISIBLE_DEVICES=0 python gradio_echo_wm.py
    # then open http://0.0.0.0:7860
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

print("[startup] script launched, importing gradio/torch/yaml...", flush=True)

import gradio as gr
import torch
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

print("[startup] gradio/torch/yaml imported, CUDA available:", torch.cuda.is_available(), flush=True)

# Gradio's own allowed_paths prefix-matching has repeatedly 403'd files that
# are genuinely under OUTPUT_ROOT/EXAMPLES_DIR (symlink/realpath
# normalization mismatch between what we pass to allowed_paths and how
# Gradio compares the incoming request path -- never fully pinned down, and
# not worth continuing to chase since we already own this override). Trust
# our own known-safe directories directly instead of delegating.
import gradio.utils as _gr_utils  # noqa: E402

_orig_is_allowed_file = _gr_utils.is_allowed_file


def _trusted_is_allowed_file(path, blocked_paths, allowed_paths, created_paths):
    try:
        resolved = Path(path).resolve()
        for trusted_root in (OUTPUT_ROOT, EXAMPLES_DIR):
            resolved_root = trusted_root.resolve()
            if resolved == resolved_root or resolved_root in resolved.parents:
                return True, "trusted output/example root"
    except OSError:
        pass
    return _orig_is_allowed_file(path, blocked_paths, allowed_paths, created_paths)


_gr_utils.is_allowed_file = _trusted_is_allowed_file
try:
    import gradio.route_utils as _gr_route_utils  # noqa: E402
    _gr_route_utils.utils.is_allowed_file = _trusted_is_allowed_file
except Exception as _e:  # noqa: BLE001
    print(f"[DEBUG] could not patch route_utils.utils.is_allowed_file: {_e}", flush=True)

# Setup paths
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
for package in ("ltx-core/src", "ltx-causal/src", "ltx-pipelines/src"):
    sys.path.insert(0, str(ROOT / package))

print("[startup] importing ltx_core/ltx_causal/ltx_pipelines...", flush=True)

from ltx_core.components.guiders import MultiModalGuiderParams  # noqa: E402
from ltx_core.types import Audio  # noqa: E402
from ltx_causal import CausalCacheConfig, DEFAULT_CAUSAL_TIMESTEPS  # noqa: E402
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline  # noqa: E402
from ltx_pipelines.causal_ti2vid import CausalTI2VidPipeline  # noqa: E402
from ltx_core.model.video_vae.tiling import TilingConfig  # noqa: E402
from ltx_core.model.video_vae.video_vae import get_video_chunks_number  # noqa: E402
from ltx_pipelines.utils.args import ImageConditioningInput  # noqa: E402
from ltx_pipelines.utils.media_io import encode_video, StreamingEncoder  # noqa: E402

print("[startup] ltx_* imported, importing helpers...", flush=True)

from helpers.action_condition import (  # noqa: E402
    action_config,
    build_action_condition,
    build_action_trajectory,
    build_causal_action_condition,
)
from helpers.action_camera import (  # noqa: E402
    DEFAULT_PITCH_LIMIT_DEG,
    DEFAULT_ROTATION_SPEED_DEG,
    DEFAULT_TRANSLATION_SPEED,
    parse_action_string,
)
from helpers.action_overlay import overlay_genie_on_video  # noqa: E402

print("[startup] all imports done", flush=True)

# Default paths (Base model — full multi-step diffusion, no live preview)
DEFAULT_CONFIG = ROOT / "configs" / "inference_wm.yaml"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "echo-wm-base.safetensors"
DEFAULT_GEMMA = ROOT / "checkpoints" / "gemma-3"

# Default paths (Flash Preview / causal — 4-step autoregressive, supports
# per-block streaming preview via EchoWMCausalEngine below)
DEFAULT_CAUSAL_CONFIG = ROOT / "configs" / "inference_wm_causal.yaml"
DEFAULT_CAUSAL_CHECKPOINT = ROOT / "checkpoints" / "echo-wm-flash.safetensors"
EXAMPLES_DIR = ROOT / "examples"
OUTPUT_ROOT = ROOT / "outputs" / "gradio_app"
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

# Make OUTPUT_ROOT Gradio's own upload/temp folder (gradio.utils.get_upload_folder()
# reads this), so every file written under it lands in Gradio's unconditionally
# trusted `created_paths` set — sidesteps allowed_paths string-matching, which was
# 403ing on files served during a streaming generation.
os.environ.setdefault("GRADIO_TEMP_DIR", str(OUTPUT_ROOT))

# NOTE: gr.Video/gr.File values that look like a real http(s):// URL get
# server-side downloaded-and-cached by Gradio (via safehttpx's SSRF guard),
# which hard-rejects ANY private/loopback address (LAN IPs, 127.0.0.1,
# localhost alike, no exceptions, no bypass short of a domain_whitelist
# Gradio doesn't expose) — so a URL pointing at this box can never work,
# tunnel or not. Plain local filesystem paths (routed through
# /gradio_api/file= + is_allowed_file()) are therefore the only viable
# option; that path relies on GRADIO_TEMP_DIR being set (above) so files
# under OUTPUT_ROOT land in Gradio's own trusted upload folder.
def output_url(path: Path) -> str:
    return str(path)

NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "game UI, video game interface, HUD, heads-up display, menu, status bar, "
    "health bar, score, minimap, crosshair, reticle, buttons, icons, subtitles, "
    "captions, watermark, logo, text overlay, user interface"
)

# Action presets (~240 frames for 10s at 24fps)
ACTION_PRESETS = {
    "forward (w-240)": "w-240",
    "S-curve weave": "wj-60,wl-60,wj-60,wl-60",
    "orbit left": "wj-40,dj-80,wj-40,dj-80",
    "look around": "w-40,l-60,i-30,k-30,j-40,w-40",
    "approach + tilt up": "w-70,wi-40,i-30,w-40,none-60",
    "strafe scan": "w-40,d-50,a-100,d-50",
}

# Denoising-step presets for the causal (Flash Preview) engine -- lets speed/
# quality be A/B'd live in the UI instead of editing configs/*.yaml and
# restarting the server for every step-count test. "(config default)" defers
# to whatever configs/inference_wm_causal*.yaml actually has set (passes
# timesteps=None, so EchoWMCausalEngine.generate() falls back to
# self.timesteps), rather than silently overriding it.
STEP_PRESETS: dict[str, tuple[int, ...] | None] = {
    "(config default)": None,
    "4 (native)": (1000, 750, 500, 250),
    "3": (1000, 625, 250),
    "2 (fastest, most quality risk)": (1000, 500),
}

ACTION_HELP = (
    "**Action DSL** — segments `<keys>-<frames>` joined by commas; keys held simultaneously.\n"
    "`w`/`s` forward/back · `a`/`d` strafe left/right · `i`/`k` pitch up/down · `j`/`l` yaw (pan) left/right · "
    "`none` holds still. Combine, e.g. `wj` = forward + pan-left. Frames total ≈ num_frames − 1 (240 for a 10s clip)."
)

# Case directories under examples/ that hold `<id>/input.png` + `<id>/case.json`.
# The causal cases target the chained multi-shot script, not this single-shot demo.
CASE_COLLECTIONS = ("wm_cases",)


def discover_cases() -> dict[str, dict]:
    """Map "<collection>/<id>" -> case fields merged with its first-frame path.

    Cases live in examples/<collection>/<id>/ with a case.json carrying the public
    semantic controls (prompt, action, optional fov_deg/seed) next to input.png.
    """
    cases: dict[str, dict] = {}
    for collection in CASE_COLLECTIONS:
        root = EXAMPLES_DIR / collection
        if not root.is_dir():
            continue
        for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            meta_path = case_dir / "case.json"
            image_path = case_dir / "input.png"
            if not meta_path.is_file() or not image_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError as exc:
                print(f"[cases] skipping {case_dir.name}: bad case.json ({exc})", flush=True)
                continue
            meta["image"] = str(image_path)
            label = case_dir.name if len(CASE_COLLECTIONS) == 1 else f"{collection}/{case_dir.name}"
            cases[label] = meta
    return cases


CASES = discover_cases()

# Live-preview streaming bridge: the pipeline's on_block callback runs on a
# plain (sync) worker thread (see EchoWMCausalEngine.generate below), but
# delivering bytes to the browser happens over a WebSocket handled by the
# asyncio event loop uvicorn runs. asyncio.Queue isn't thread-safe to push
# into directly from another thread, so the sync side schedules the push via
# loop.call_soon_threadsafe() instead of calling queue.put_nowait() itself.
# Keyed by a per-generation run_id (created in on_generate, sent to the
# browser via the hidden stream_trigger textbox, and used by the frontend JS
# to open /ws/stream/<run_id>). A queue value of None is the end-of-stream
# sentinel.
_main_event_loop: asyncio.AbstractEventLoop | None = None
_stream_queues: dict[str, "asyncio.Queue[bytes | None]"] = {}
_stream_lock = threading.Lock()


def _register_stream(run_id: str) -> None:
    with _stream_lock:
        _stream_queues[run_id] = asyncio.Queue()


def _push_stream_chunk(run_id: str, data: bytes | None) -> None:
    if _main_event_loop is None:
        return
    with _stream_lock:
        q = _stream_queues.get(run_id)
    if q is None:
        return
    _main_event_loop.call_soon_threadsafe(q.put_nowait, data)


class EchoWMEngine:
    """Loads the Echo-WM model once; generates on demand."""

    def __init__(
        self,
        checkpoint: Path,
        gemma_path: Path,
        config_path: Path,
        device: torch.device,
    ):
        self.device = device
        self.checkpoint = checkpoint
        self.gemma_path = gemma_path
        self.config_path = config_path

        print(f"[engine] Loading config from {config_path}", flush=True)
        self.cfg = yaml.safe_load(config_path.read_text()) or {}
        self.video_cfg = self.cfg.get("video", {})
        self.action_cfg = self.cfg.get("action", {})

        print(f"[engine] Loading Echo-WM model...", flush=True)
        print(f"  checkpoint: {checkpoint}", flush=True)
        print(f"  gemma: {gemma_path}", flush=True)

        # The pipeline only records paths here; the 47GB of weights are read lazily on
        # the first generation. Probe the checkpoint header now so a wrong --checkpoint
        # fails at startup instead of surfacing minutes later as a failed generation.
        self._probe_checkpoint(checkpoint)

        self.pipeline = TI2VidOneStagePipeline(
            checkpoint_path=str(checkpoint),
            gemma_root=str(gemma_path),
            loras=(),
            device=device,
            action_config=None,  # Will be set per generation
        )
        print("[engine] Ready (weights load on first generation).", flush=True)

    @staticmethod
    def _probe_checkpoint(checkpoint: Path) -> None:
        """Validate the checkpoint is a readable safetensors file before serving."""
        if not checkpoint.is_file():
            raise SystemExit(f"[engine] Checkpoint not found: {checkpoint}")
        from safetensors import safe_open

        try:
            with safe_open(str(checkpoint), framework="pt") as f:
                n_tensors = len(f.keys())
        except Exception as exc:  # noqa: BLE001 - surface any unreadable file the same way
            raise SystemExit(f"[engine] Cannot read checkpoint {checkpoint}: {exc}") from exc
        size_gb = checkpoint.stat().st_size / 1024**3
        print(f"  verified: {n_tensors} tensors, {size_gb:.1f} GiB", flush=True)

    @torch.inference_mode()
    def generate(
        self,
        image_path: str,
        prompt: str,
        action_str: str,
        seed: int,
        num_frames: int,
        fps: float,
        steps: int,
        video_cfg: float,
        audio_cfg: float,
        width: int,
        height: int,
        fov_deg: float,
        translation_speed: float,
        rotation_speed_deg: float,
        pitch_limit_deg: float,
        generate_audio: bool,
        overlay: bool,
        out_dir: Path,
    ) -> tuple[Path, Path | None, dict]:
        """Returns (video_path, overlaid_path_or_None, timing)."""
        timing: dict[str, float] = {}

        # Validate action string early
        parse_action_string(action_str)

        # Build action condition
        t0 = time.time()
        action_cond = build_action_condition(
            action_str,
            num_frames=num_frames,
            width=width,
            height=height,
            translation_speed=translation_speed,
            rotation_speed_deg=rotation_speed_deg,
            pitch_limit_deg=pitch_limit_deg,
            fov_deg=fov_deg,
            device=self.device,
            fps=fps,
        )
        timing["action_prep"] = time.time() - t0

        # Generate
        t0 = time.time()
        # Update action config for this generation
        self.pipeline.action_config = action_config(width, height)

        video, audio = self.pipeline(
            prompt=prompt,
            negative_prompt=self.cfg.get("negative_prompt", NEGATIVE_PROMPT),
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            num_inference_steps=steps,
            video_guider_params=MultiModalGuiderParams(
                cfg_scale=video_cfg,
                stg_scale=self.video_cfg.get("stg_scale", 1.0),
                stg_blocks=self.video_cfg.get("stg_blocks", [29]),
            ),
            audio_guider_params=MultiModalGuiderParams(
                cfg_scale=audio_cfg,
                stg_scale=self.video_cfg.get("stg_scale", 1.0),
                stg_blocks=self.video_cfg.get("stg_blocks", [29]),
            ),
            images=[ImageConditioningInput(str(image_path), 0, 1.0)],
            action_cond=action_cond,
            video_tiling_config=TilingConfig.default(),
        )
        timing["generate"] = time.time() - t0

        # Save video
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / "output.mp4"

        t0 = time.time()
        encode_video(
            video=video,
            fps=int(fps),
            audio=audio if generate_audio else None,
            output_path=str(video_path),
            video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
        )
        timing["encode"] = time.time() - t0

        # Overlay if requested
        overlaid_path = None
        if overlay:
            t0 = time.time()
            trajectory = build_action_trajectory(
                action_str,
                num_frames=num_frames,
                translation_speed=translation_speed,
                rotation_speed_deg=rotation_speed_deg,
                pitch_limit_deg=pitch_limit_deg,
                fps=fps,
            )
            overlaid_path = out_dir / "output_action.mp4"
            overlay_genie_on_video(video_path, trajectory, output_path=overlaid_path)
            timing["overlay"] = time.time() - t0

        return video_path, overlaid_path, timing


class EchoWMCausalEngine:
    """Loads Echo-WM Flash Preview (causal, 4-step autoregressive) once; streams
    each denoised block as its own short mp4 as soon as it's ready, instead of
    only returning the full video at the end like EchoWMEngine.
    """

    def __init__(
        self,
        checkpoint: Path,
        gemma_path: Path,
        config_path: Path,
        device: torch.device,
    ):
        self.device = device
        self.checkpoint = checkpoint
        self.gemma_path = gemma_path
        self.config_path = config_path

        print(f"[engine] Loading config from {config_path}", flush=True)
        self.cfg = yaml.safe_load(config_path.read_text()) or {}
        self.causal_cfg = self.cfg.get("causal", {})

        print("[engine] Loading Echo-WM Flash (causal) model...", flush=True)
        print(f"  checkpoint: {checkpoint}", flush=True)
        print(f"  gemma: {gemma_path}", flush=True)
        EchoWMEngine._probe_checkpoint(checkpoint)

        self.cache_config = CausalCacheConfig(
            video_local_attn_size=self.causal_cfg.get("video_local_attn_size", 19),
            video_sink_size=self.causal_cfg.get("video_sink_size", 7),
            video_chunk_size=self.causal_cfg.get("video_chunk_size", 3),
        )
        self.cache_config.validate()
        self.timesteps = tuple(self.causal_cfg.get("timesteps", DEFAULT_CAUSAL_TIMESTEPS))

        self.pipeline = CausalTI2VidPipeline(
            checkpoint_path=str(checkpoint),
            gemma_root=str(gemma_path),
            device=device,
            action_config=None,  # Set per generation
            cache_config=self.cache_config,
        )
        print("[engine] Ready (weights load on first generation, streaming enabled).", flush=True)

    def generate(
        self,
        image_path: str,
        prompt: str,
        action_str: str,
        seed: int,
        num_frames: int,
        fps: float,
        width: int,
        height: int,
        fov_deg: float,
        translation_speed: float,
        rotation_speed_deg: float,
        pitch_limit_deg: float,
        generate_audio: bool,
        overlay: bool,
        out_dir: Path,
        timesteps: tuple[int, ...] | None = None,
        on_stream_chunk=None,
    ):
        """Generator. Yields ("block", index, total, block_video_path) as each
        block finishes, then a final ("done", video_path, overlaid_path_or_None,
        timing).

        Runs the (blocking) pipeline call on a background thread and relays its
        on_block callbacks through a queue, since a callback fired from inside
        a blocking call cannot itself yield from this generator's frame.

        `timesteps` overrides self.timesteps for this call only -- kernel
        compilation/backend dispatch happens per tensor *shape*, not per
        scalar timestep value, so a warmup call can safely use fewer steps
        (cutting real per-step compute) while still exercising every kernel
        a real generation would use.

        `on_stream_chunk(bytes | None)`, if given, is called (from the same
        worker thread as on_block below -- caller is responsible for making
        it thread-safe) with each newly-available fragmented-MP4 byte range
        as blocks are decoded, for live MSE streaming to the browser. A
        final call with None marks end-of-stream.
        """
        timing: dict[str, float] = {}
        parse_action_string(action_str)
        timesteps = timesteps if timesteps is not None else self.timesteps

        t0 = time.time()
        action_cond = build_causal_action_condition(
            device=self.device,
            action=action_str,
            num_frames=num_frames,
            width=width,
            height=height,
            translation_speed=translation_speed,
            rotation_speed_deg=rotation_speed_deg,
            pitch_limit_deg=pitch_limit_deg,
            fov_deg=fov_deg,
            fps=fps,
        )
        timing["action_prep"] = time.time() - t0

        self.pipeline.action_config = action_config(width, height)

        out_dir.mkdir(parents=True, exist_ok=True)
        blocks_dir = out_dir / "blocks"
        blocks_dir.mkdir(parents=True, exist_ok=True)

        result_queue: queue.Queue = queue.Queue()
        frame_count = {"n": 0}

        # A typical GOP (~0.5s at the configured fps) -- doesn't need to
        # line up with the pipeline's own block boundaries. write_chunk()
        # just returns whatever fragments happened to close since the last
        # call (possibly none, possibly several); the encoder forces a
        # keyframe/fragment boundary every gop_size frames regardless of how
        # write_chunk() is called.
        stream_encoder = (
            StreamingEncoder(
                width=width, height=height, fps=int(fps), frames_per_chunk=max(int(fps) // 2, 1),
                include_audio=generate_audio,
            )
            if on_stream_chunk is not None
            else None
        )

        # Encoding (StreamingEncoder.write_chunk, CPU-bound: numpy convert +
        # libx264/AAC encode) was previously called directly from on_block,
        # which runs on the same thread that then immediately does the
        # cache-update forward() and the next block's denoise -- serializing
        # CPU encode work with GPU work for no real reason (encode only
        # needs this block's already-denoised output, nothing downstream of
        # it). Measured on real hardware: ~0.4s/block of the ~1.9s total was
        # this encode step. Moving it to a dedicated background thread (FIFO
        # queue, so muxed byte order stays correct) lets it overlap with the
        # next block's GPU work instead of blocking it.
        encode_queue: queue.Queue = queue.Queue()

        def _encode_worker() -> None:
            while True:
                item = encode_queue.get()
                if item is None:  # sentinel: no more chunks, drain complete
                    break
                video_chunk_item, audio_chunk_item = item
                try:
                    new_bytes = stream_encoder.write_chunk(video_chunk_item, audio_chunk_item)
                    if new_bytes:
                        on_stream_chunk(new_bytes)
                except Exception as exc:  # noqa: BLE001 - live preview is best-effort
                    print(f"[stream] write_chunk failed (continuing without live stream): {exc}", flush=True)

        encode_thread = None
        if stream_encoder is not None:
            encode_thread = threading.Thread(target=_encode_worker, daemon=True)
            encode_thread.start()

        def on_block(block_index: int, total_blocks: int, video_chunk, audio_chunk) -> None:
            # Some callbacks in this pipeline carry zero video frames (e.g.
            # audio-only/bookkeeping chunks) -- expected, not an error. There's
            # nothing to encode or show for these; encoding them anyway
            # produces no output file (PyAV's muxer writes nothing for zero
            # frames) and pointing the UI at that file 403s at the Gradio
            # file-serving layer since it never exists.
            if video_chunk.shape[0] == 0:
                frame_count["n"] += 0
                result_queue.put(("progress", block_index, total_blocks))
                return
            # block_path is kept as a *name* for status-text/log purposes
            # only -- the file itself is no longer written when the
            # WebSocket live stream is active (stream_encoder is not None),
            # since nothing displays it anymore (the old block-swap
            # gr.Video preview this used to feed is gone). Encoding the
            # same frames twice (once here, once into stream_encoder below)
            # was pure wasted per-block latency.
            block_path = blocks_dir / f"block_{block_index:03d}.mp4"
            if stream_encoder is None:
                encode_video(
                    video=video_chunk,
                    fps=int(fps),
                    audio=audio_chunk if generate_audio else None,
                    output_path=str(block_path),
                    video_chunks_number=1,
                )
            frame_count["n"] += video_chunk.shape[0]
            if stream_encoder is not None:
                # Non-blocking hand-off -- the actual encode+push happens on
                # _encode_worker's thread, concurrently with whatever this
                # (rollout) thread does next (cache-update forward(), then
                # the next block's denoise). video_chunk/audio_chunk are
                # already .clone()'d by the caller (raw_on_block in
                # causal_ti2vid.py), so they're safe to hand to another
                # thread.
                encode_queue.put((video_chunk, audio_chunk if generate_audio else None))
            result_queue.put(("block", block_index, total_blocks, block_path, frame_count["n"]))

        def worker() -> None:
            try:
                video, audio = self.pipeline(
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=fps,
                    images=[ImageConditioningInput(str(image_path), 0, 1.0)],
                    action_cond=action_cond,
                    timesteps=timesteps,
                    video_tiling_config=TilingConfig.default(),
                    on_block=on_block,
                )
                result_queue.put(("final", video, audio))
            except Exception as exc:  # noqa: BLE001 - relay to the consumer thread
                result_queue.put(("error", exc))

        t0 = time.time()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        video = audio = None
        while True:
            try:
                item = result_queue.get(timeout=2.0)
            except queue.Empty:
                # No new block/done from the pipeline in 2s -- still running
                # (e.g. denoising/audio rollout steps between causal video
                # decode chunks), just nothing new to show yet. Let the
                # caller know it's still alive instead of going silent.
                yield ("heartbeat", time.time() - t0)
                continue
            if item[0] == "block":
                yield item
            elif item[0] == "progress":
                yield item
            elif item[0] == "error":
                thread.join()
                if stream_encoder is not None:
                    # Drain whatever's still queued (best-effort -- the
                    # generation itself failed, but don't leave the encode
                    # thread running past this call) before telling the
                    # browser the stream is over.
                    encode_queue.put(None)
                    encode_thread.join()
                    on_stream_chunk(None)
                raise item[1]
            else:
                _, video, audio = item
                break
        thread.join()
        timing["generate"] = time.time() - t0

        if stream_encoder is not None:
            # Sentinel + join: wait for every already-queued block's
            # write_chunk() to actually finish before calling close() --
            # otherwise close()'s flush could race with (and interleave
            # incorrectly ahead of) a still-pending block's encode.
            encode_queue.put(None)
            encode_thread.join()
            try:
                tail_bytes = stream_encoder.close()
                if tail_bytes:
                    on_stream_chunk(tail_bytes)
            except Exception as exc:  # noqa: BLE001 - live preview is best-effort
                print(f"[stream] StreamingEncoder.close failed: {exc}", flush=True)
            on_stream_chunk(None)

        video_path = out_dir / "output.mp4"
        t0 = time.time()
        # `video` is a lazy generator (nothing decoded yet) returned by
        # self.pipeline(), whose own @torch.inference_mode() scope has
        # already closed by this point. Consuming it (inside encode_video,
        # via next()) must happen under its own inference_mode, or the
        # decoder's weights — created as inference tensors — crash trying to
        # build an autograd graph in plain (grad-enabled) mode.
        with torch.inference_mode():
            encode_video(
                video=video,
                fps=int(fps),
                audio=audio if generate_audio else None,
                output_path=str(video_path),
                video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
            )
        timing["encode"] = time.time() - t0

        overlaid_path = None
        if overlay:
            t0 = time.time()
            trajectory = build_action_trajectory(
                action_str,
                num_frames=num_frames,
                translation_speed=translation_speed,
                rotation_speed_deg=rotation_speed_deg,
                pitch_limit_deg=pitch_limit_deg,
                fps=fps,
            )
            overlaid_path = out_dir / "output_action.mp4"
            overlay_genie_on_video(video_path, trajectory, output_path=overlaid_path)
            timing["overlay"] = time.time() - t0

        yield ("done", video_path, overlaid_path, timing)


def build_ui(engine: EchoWMEngine) -> gr.Blocks:
    """Build Gradio interface."""
    run_counter = {"n": 0}

    def on_preset(name: str):
        return gr.update(value=ACTION_PRESETS.get(name, "w-240"))

    def on_case(name: str):
        """Fill the first frame and the case's authored controls."""
        case = CASES.get(name)
        if case is None:
            return (gr.update(),) * 5
        return (
            gr.update(value=case["image"]),
            gr.update(value=case.get("prompt", "")),
            gr.update(value=case.get("action", "w-240")),
            gr.update(value=case.get("fov_deg", 70.0)),
            gr.update(value=case.get("seed", 42)),
        )

    def on_generate(
        image_path, prompt, action_str, seed, num_frames, fps, steps,
        video_cfg, audio_cfg, width, height, fov_deg,
        translation_speed, rotation_speed, pitch_limit,
        gen_audio, overlay,
    ):
        if not image_path:
            yield "❌ Pick or upload an image first.", None, None
            return
        if not (prompt or "").strip():
            yield "❌ Prompt is empty.", None, None
            return
        try:
            parse_action_string(action_str)
        except Exception as e:
            yield f"❌ Invalid action string: {e}", None, None
            return

        run_counter["n"] += 1
        out_dir = OUTPUT_ROOT / f"run_{run_counter['n']:04d}"

        est_time = int(num_frames) * steps // 100  # rough estimate
        yield (
            f"⏳ Generating {int(num_frames)}f @ {int(steps)} steps (~{est_time}s)…\n"
            f"action=[{action_str}] seed={int(seed)}"
        ), None, None

        t0 = time.time()
        try:
            video_path, overlaid_path, timing = engine.generate(
                image_path=image_path,
                prompt=prompt,
                action_str=action_str,
                seed=int(seed),
                num_frames=int(num_frames),
                fps=float(fps),
                steps=int(steps),
                video_cfg=float(video_cfg),
                audio_cfg=float(audio_cfg),
                width=int(width),
                height=int(height),
                fov_deg=float(fov_deg),
                translation_speed=float(translation_speed),
                rotation_speed_deg=float(rotation_speed),
                pitch_limit_deg=float(pitch_limit),
                generate_audio=bool(gen_audio),
                overlay=bool(overlay),
                out_dir=out_dir,
            )
        except Exception as e:
            yield f"❌ Generation failed: {e}\n{traceback.format_exc()[-800:]}", None, None
            return

        shown = overlaid_path or video_path
        parts = "  ".join(f"{k}={v:.1f}s" for k, v in timing.items())
        msg = f"✅ Done in {time.time() - t0:.1f}s ({parts}).\n  video: {video_path.name}"
        if overlaid_path:
            msg += f"\n  overlay: {overlaid_path.name}"
        yield msg, output_url(shown), str(video_path)

    with gr.Blocks(title="Echo-WM World Model") as demo:
        gr.Markdown(
            f"# Echo-WM: Action-Conditioned World Model\n"
            f"Checkpoint: `{engine.checkpoint.name}` · Gemma: `{engine.gemma_path.name}`"
        )

        with gr.Row():
            with gr.Column(scale=1):
                case_picker = gr.Dropdown(
                    list(CASES),
                    label="Example case (fills image, prompt, action, FOV, seed)",
                    value=None,
                )
                image = gr.Image(label="First-frame image", type="filepath", height=300)

                with gr.Row():
                    prompt = gr.Textbox(
                        label="Prompt",
                        lines=4,
                        placeholder="Describe the scene, style, perspective...",
                    )
                gr.Markdown(ACTION_HELP)
                action = gr.Textbox(label="Action string", value="w-240")
                preset = gr.Dropdown(
                    list(ACTION_PRESETS),
                    label="Action preset",
                    value="forward (w-240)",
                )

                with gr.Accordion("Video Settings", open=False):
                    with gr.Row():
                        # Fixed: the model is trained at this resolution.
                        width = gr.Number(label="Width", value=1280, precision=0, interactive=False)
                        height = gr.Number(label="Height", value=704, precision=0, interactive=False)
                    with gr.Row():
                        num_frames = gr.Number(label="Frames (241=10s)", value=241, precision=0)
                        fps = gr.Number(label="FPS", value=24, precision=1)
                    with gr.Row():
                        steps = gr.Slider(10, 50, value=30, step=1, label="Inference steps")
                        seed = gr.Number(label="Seed", value=42, precision=0)
                    with gr.Row():
                        video_cfg = gr.Slider(1.0, 8.0, value=4.0, step=0.5, label="Video CFG")
                        audio_cfg = gr.Slider(1.0, 8.0, value=2.0, step=0.5, label="Audio CFG")

                with gr.Accordion("Action Settings", open=False):
                    fov_deg = gr.Slider(30, 120, value=70, step=5, label="FOV (degrees)")
                    translation_speed = gr.Slider(
                        0.005, 0.1, value=DEFAULT_TRANSLATION_SPEED, step=0.005,
                        label="Translation speed (w/s/a/d per frame)",
                    )
                    rotation_speed = gr.Slider(
                        0.1, 3.0, value=DEFAULT_ROTATION_SPEED_DEG, step=0.1,
                        label="Rotation speed (°/frame, i/k/j/l)",
                    )
                    pitch_limit = gr.Slider(
                        0, 90, value=DEFAULT_PITCH_LIMIT_DEG, step=5,
                        label="Pitch limit (degrees)",
                    )

                with gr.Row():
                    overlay = gr.Checkbox(label="Action HUD overlay", value=True)
                    gen_audio = gr.Checkbox(label="Generate audio", value=True)

                generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

            with gr.Column(scale=1):
                out_video = gr.Video(label="Result", height=400)
                status = gr.Textbox(label="Status", lines=6, interactive=False)
                raw_file = gr.File(label="Raw video (no overlay)", interactive=False)

        # Event handlers
        case_picker.change(
            on_case, inputs=case_picker,
            outputs=[image, prompt, action, fov_deg, seed],
        )
        preset.change(on_preset, inputs=preset, outputs=action)
        generate_btn.click(
            on_generate,
            inputs=[
                image, prompt, action, seed, num_frames, fps, steps,
                video_cfg, audio_cfg, width, height, fov_deg,
                translation_speed, rotation_speed, pitch_limit,
                gen_audio, overlay,
            ],
            outputs=[status, out_video, raw_file],
            concurrency_limit=1,
        )

    return demo


def build_causal_ui(engine: EchoWMCausalEngine) -> gr.Blocks:
    """Build the Gradio interface for the streaming Flash Preview (causal) engine."""
    run_counter = {"n": 0}
    video_cfg = engine.cfg.get("video", {})
    # UI defaults: config file's width/height if set, else 896x512 --
    # deliberately lower than the historical 1280x704 default (same aspect
    # ratio, ~51% the pixel count) for faster iteration. Still a multiple of
    # 32 (required by assert_resolution) and adjustable in the UI below, not
    # a model-imposed fixed size.
    default_width = int(video_cfg.get("width", 896))
    default_height = int(video_cfg.get("height", 512))
    default_num_frames = int(video_cfg.get("num_frames", 241))
    default_fps = float(video_cfg.get("fps", 24))
    default_seed = int(video_cfg.get("seed", 42))
    action_cfg = engine.cfg.get("action", {})

    def on_preset(name: str):
        return gr.update(value=ACTION_PRESETS.get(name, "w-240"))

    def on_case(name: str):
        case = CASES.get(name)
        if case is None:
            return (gr.update(),) * 5
        return (
            gr.update(value=case["image"]),
            gr.update(value=case.get("prompt", "")),
            gr.update(value=case.get("action", "w-240")),
            gr.update(value=case.get("fov_deg", 70.0)),
            gr.update(value=case.get("seed", 42)),
        )

    def on_generate(
        image_path, prompt, action_str, seed, num_frames, fps, width, height,
        fov_deg, translation_speed, rotation_speed, pitch_limit,
        gen_audio, overlay, denoise_steps_choice,
    ):
        if not image_path:
            yield "❌ Pick or upload an image first.", None, None, None
            return
        if not (prompt or "").strip():
            yield "❌ Prompt is empty.", None, None, None
            return
        try:
            parse_action_string(action_str)
        except Exception as e:
            yield f"❌ Invalid action string: {e}", None, None, None
            return

        run_counter["n"] += 1
        out_dir = OUTPUT_ROOT / f"run_causal_{run_counter['n']:04d}"

        # run_id addresses this generation's WebSocket stream
        # (/ws/stream/<run_id>). Setting stream_trigger's value below sends
        # it to the browser; the head JS polls that hidden textbox and opens
        # the socket + MediaSource as soon as it changes.
        run_id = uuid.uuid4().hex
        _register_stream(run_id)

        def on_stream_chunk(data: bytes | None) -> None:
            _push_stream_chunk(run_id, data)

        # "<run_id>|<1-or-0>" -- the frontend needs to know upfront whether
        # this stream carries an audio track at all, since MediaSource's
        # addSourceBuffer() MIME type must declare the exact codec set
        # present in the byte stream (see _mse_stream_js's connect()).
        stream_trigger_value = f"{run_id}|{'1' if gen_audio else '0'}"
        yield (
            f"⏳ Streaming generation started…\naction=[{action_str}] seed={int(seed)}"
        ), stream_trigger_value, None, None

        t0 = time.time()
        try:
            for item in engine.generate(
                image_path=image_path,
                prompt=prompt,
                action_str=action_str,
                seed=int(seed),
                num_frames=int(num_frames),
                fps=float(fps),
                width=int(width),
                height=int(height),
                fov_deg=float(fov_deg),
                translation_speed=float(translation_speed),
                rotation_speed_deg=float(rotation_speed),
                pitch_limit_deg=float(pitch_limit),
                generate_audio=bool(gen_audio),
                overlay=bool(overlay),
                out_dir=out_dir,
                on_stream_chunk=on_stream_chunk,
                timesteps=STEP_PRESETS.get(denoise_steps_choice),
            ):
                if item[0] == "block":
                    _, block_index, total_blocks, block_path, frames_so_far = item
                    elapsed = time.time() - t0
                    fps_estimate = frames_so_far / elapsed if elapsed > 0 else 0.0
                    yield (
                        f"⏳ Block {block_index + 1}/{total_blocks} ({elapsed:.1f}s elapsed, "
                        f"{frames_so_far} frames so far, ~{fps_estimate:.2f} fps generated)…\n"
                        f"  last update: {time.strftime('%H:%M:%S')} -- {block_path.name}"
                    ), gr.update(), None, None
                elif item[0] == "progress":
                    _, block_index, total_blocks = item
                    elapsed = time.time() - t0
                    yield (
                        f"⏳ Block {block_index + 1}/{total_blocks} ({elapsed:.1f}s elapsed) -- "
                        f"no new video content this block (audio-only/bookkeeping step)…"
                    ), gr.update(), gr.update(), gr.update()
                elif item[0] == "heartbeat":
                    elapsed = item[1]
                    yield (
                        f"⏳ Still generating... ({elapsed:.1f}s elapsed, no new preview "
                        f"chunk yet -- denoising/audio steps between video decodes)…"
                    ), gr.update(), gr.update(), gr.update()
                else:
                    _, video_path, overlaid_path, timing = item
                    shown = overlaid_path or video_path
                    parts = "  ".join(f"{k}={v:.1f}s" for k, v in timing.items())
                    generate_secs = timing.get("generate", 0.0)
                    avg_fps = int(num_frames) / generate_secs if generate_secs > 0 else 0.0
                    msg = (
                        f"✅ Done in {time.time() - t0:.1f}s ({parts}).\n"
                        f"  avg generation speed: ~{avg_fps:.2f} fps ({int(num_frames)} frames / {generate_secs:.1f}s)\n"
                        f"  video: {video_path.name}"
                    )
                    if overlaid_path:
                        msg += f"\n  overlay: {overlaid_path.name}"
                    yield msg, gr.update(), output_url(shown), str(video_path)
        except Exception as e:
            on_stream_chunk(None)  # unstick the browser's WebSocket if we errored mid-stream
            yield f"❌ Generation failed: {e}\n{traceback.format_exc()}", gr.update(), None, None

    # Watches every <video> element's `src` attribute for changes and logs
    # each one with a timestamp -- shows directly in the browser console
    # whether Gradio's frontend ever actually applies a block update to the
    # DOM, independent of whatever's happening server-side (queue/SSE
    # delivery vs. the video element just not re-rendering).
    _video_src_watch_js = """
    <script>
    (function() {
      function logChange(video, reason) {
        console.log("[video-watch]", new Date().toISOString(), reason, "currentSrc:", video.currentSrc, "src attr:", video.getAttribute("src"));
      }
      function watch(video) {
        if (video.__srcWatched) return;
        video.__srcWatched = true;
        // Covers: src attribute changes on <video> itself, <source> children
        // being added/removed/changed (some players swap sources instead of
        // setting video.src directly), and any other child mutation.
        const obs = new MutationObserver((muts) => {
          for (const m of muts) {
            if (m.type === "attributes" && m.attributeName === "src") {
              logChange(video, "attribute mutation");
            } else if (m.type === "childList") {
              logChange(video, "child nodes changed (e.g. <source> swap)");
            }
          }
        });
        obs.observe(video, {attributes: true, attributeFilter: ["src"], childList: true, subtree: true});
        // loadstart/loadeddata fire whenever the browser actually starts
        // loading new media, regardless of *how* src was set (attribute,
        // property, or a <source> child) -- this is the most reliable signal.
        video.addEventListener("loadstart", () => logChange(video, "loadstart event"));
        video.addEventListener("loadeddata", () => logChange(video, "loadeddata event"));
        video.addEventListener("error", () => console.log("[video-watch]", new Date().toISOString(), "ERROR event", video.error));
        console.log("[video-watch] now watching", video, "current src:", video.currentSrc);
      }
      const scan = () => document.querySelectorAll("video").forEach(watch);
      scan();
      new MutationObserver(scan).observe(document.body, {childList: true, subtree: true});

      // Also log Gradio's underlying queue traffic (SSE + fetch), so we can
      // tell whether an update message ever reaches the browser at all,
      // separate from whether the video element re-renders once it does.
      const origFetch = window.fetch;
      window.fetch = function(...args) {
        const url = String(args[0]);
        if (url.includes("queue") || url.includes("gradio_api")) {
          console.log("[net-watch]", new Date().toISOString(), "fetch", url);
        }
        return origFetch.apply(this, args);
      };
      const OrigES = window.EventSource;
      if (OrigES) {
        window.EventSource = function(url, opts) {
          console.log("[net-watch]", new Date().toISOString(), "EventSource opened", url);
          const es = new OrigES(url, opts);
          es.addEventListener("message", (ev) => {
            console.log("[net-watch]", new Date().toISOString(), "SSE message", ev.data);
          });
          return es;
        };
        window.EventSource.prototype = OrigES.prototype;
      }
    })();
    </script>
    """

    # Replaces gr.Video(streaming=True) (tried earlier, reverted -- see
    # TROUBLESHOOTING.md item -1) with a raw <video> fed via MediaSource
    # Extensions over a dedicated WebSocket (/ws/stream/<run_id>, mounted in
    # main()). Avoids the block-swap reload stutter entirely: instead of the
    # browser re-fetching+re-decoding a whole new file per block, bytes are
    # appended incrementally to one continuous SourceBuffer.
    #
    # stream-trigger is a hidden textbox on_generate sets to a fresh run_id
    # at the start of each generation; this script polls its DOM value
    # (Gradio doesn't expose a clean JS hook for "value changed", and a
    # MutationObserver on a controlled-input's value attribute is
    # unreliable across frameworks -- polling is simple and cheap here)
    # and opens a new WebSocket + MediaSource whenever it changes.
    _mse_stream_js = """
    <style>
      /* stream_trigger must be visible=True in Gradio (invisible components
         aren't mounted in the DOM at all, so JS can't poll them) -- hide it
         purely cosmetically here instead. */
      #stream-trigger { display: none !important; }
    </style>
    <script>
    (function() {
      function log(...args) { console.log("[mse-stream]", new Date().toISOString(), ...args); }
      log("script loaded, starting poll for #stream-trigger");

      let lastRunId = null;
      let ws = null;
      let mediaSource = null;
      let sourceBuffer = null;
      let pendingChunks = [];
      let streamDone = false;
      let pollCount = 0;
      let sawElement = false;

      function teardown() {
        if (ws) { try { ws.close(); } catch (e) {} }
        ws = null;
        mediaSource = null;
        sourceBuffer = null;
        pendingChunks = [];
        streamDone = false;
      }

      function maybeEndStream() {
        if (streamDone && sourceBuffer && !sourceBuffer.updating && pendingChunks.length === 0
            && mediaSource && mediaSource.readyState === "open") {
          try { mediaSource.endOfStream(); log("endOfStream()"); } catch (e) { log("endOfStream error", e); }
        }
      }

      function appendNext() {
        if (!sourceBuffer || sourceBuffer.updating || pendingChunks.length === 0) return;
        const chunk = pendingChunks.shift();
        try {
          sourceBuffer.appendBuffer(chunk);
        } catch (e) {
          log("appendBuffer error (dropping chunk)", e);
        }
      }

      function connect(runId, hasAudio) {
        teardown();
        const video = document.getElementById("live-preview-video");
        if (!video) { log("no #live-preview-video element found yet"); return; }

        mediaSource = new MediaSource();
        video.src = URL.createObjectURL(mediaSource);

        mediaSource.addEventListener("sourceopen", () => {
          log("MediaSource opened for run", runId, "hasAudio:", hasAudio);
          // avc1.640028 = H.264 High Profile Level 4.0 (matches libx264's
          // default here); mp4a.40.2 = AAC-LC (matches the "aac" encoder in
          // StreamingEncoder). This must match the actual track count in
          // the byte stream exactly -- declaring an audio codec MSE doesn't
          // find in the init segment (or vice versa) fails every
          // appendBuffer call, not just this one, so it's driven by
          // hasAudio (parsed from the trigger value) rather than guessed.
          const mimeType = hasAudio
            ? 'video/mp4; codecs="avc1.640028,mp4a.40.2"'
            : 'video/mp4; codecs="avc1.640028"';
          try {
            sourceBuffer = mediaSource.addSourceBuffer(mimeType);
          } catch (e) {
            log("addSourceBuffer failed", e);
            return;
          }
          sourceBuffer.addEventListener("updateend", () => {
            appendNext();
            maybeEndStream();
          });
          appendNext();
        });

        const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = proto + "//" + window.location.host + "/ws/stream/" + runId;
        ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";
        ws.onopen = () => log("WebSocket connected", wsUrl);
        ws.onmessage = (ev) => {
          if (typeof ev.data === "string") {
            if (ev.data === "__DONE__") {
              log("stream done");
              streamDone = true;
              maybeEndStream();
            }
            return;
          }
          pendingChunks.push(ev.data);
          appendNext();
        };
        ws.onerror = (e) => log("WebSocket error", e);
        ws.onclose = () => log("WebSocket closed");

        // A single play() call at connect time is unreliable: Chrome's power
        // saver can interrupt muted video-only autoplay (observed: "video-only
        // background media was paused to save power") even while the tab is
        // in the foreground if e.g. devtools has focus instead of the page.
        // Retry on every signal that more data/visibility changed, and stop
        // once genuinely playing.
        //
        // Separately: browsers block autoplay WITH sound unless the site has
        // "high media engagement" or the play() call follows a user gesture
        // -- there's no way around that from a live-updating background
        // preview. So try unmuted first (works for a returning user Chrome
        // already trusts); if blocked, fall back to muted (so the picture at
        // least plays) and let a click on the video unmute + resume audio.
        let playing = false;
        let mutedFallback = false;
        function tryPlay(reason) {
          if (playing || !video.paused) { playing = true; return; }
          video.play().then(() => {
            playing = true;
            log("play() succeeded (" + reason + ")" + (video.muted ? " [muted]" : " [with audio]"));
          }).catch((e) => {
            log("play() blocked (" + reason + "):", e.message);
            if (hasAudio && !video.muted) {
              log("falling back to muted playback -- click the video to enable sound");
              video.muted = true;
              mutedFallback = true;
              tryPlay(reason + "+muted-fallback");
            }
          });
        }
        if (hasAudio) {
          video.addEventListener("click", () => {
            if (mutedFallback) {
              video.muted = false;
              mutedFallback = false;
              video.play().then(() => log("unmuted after click")).catch((e) => log("unmute-play failed", e));
            }
          });
        } else {
          video.muted = true;  // no audio track exists at all -- avoid a pointless unmuted play() attempt
        }
        tryPlay("initial");
        video.addEventListener("loadeddata", () => tryPlay("loadeddata"));
        video.addEventListener("canplay", () => tryPlay("canplay"));
        document.addEventListener("visibilitychange", () => {
          if (!document.hidden) tryPlay("visibilitychange");
        });
        video.addEventListener("playing", () => { playing = true; log("playing event fired"); });
      }

      function poll() {
        pollCount++;
        const el = document.querySelector("#stream-trigger textarea, #stream-trigger input");
        if (el && !sawElement) {
          sawElement = true;
          log("#stream-trigger element found in DOM (took", pollCount, "polls)");
        }
        // Every ~10s, log a heartbeat so "nothing happened" and "script never
        // started" are distinguishable from the console alone -- this was
        // genuinely ambiguous before (see TROUBLESHOOTING.md) and cost a
        // long back-and-forth to pin down.
        if (pollCount % 50 === 0) {
          log("poll heartbeat, elementFound:", sawElement, "currentValue:", el ? el.value : "(no element)");
        }
        if (el && el.value && el.value !== lastRunId) {
          lastRunId = el.value;
          // Trigger value is "<run_id>|<1-or-0>" -- the hasAudio flag has to
          // travel with the run id since MSE's addSourceBuffer() MIME type
          // must declare the exact track set the byte stream actually has
          // (see connect() above).
          const [runId, hasAudioFlag] = lastRunId.split("|");
          log("new run id detected:", runId, "hasAudio:", hasAudioFlag === "1");
          connect(runId, hasAudioFlag === "1");
        }
        setTimeout(poll, 200);
      }
      poll();
    })();
    </script>
    """

    with gr.Blocks(
        title="Echo-WM Flash Preview (Streaming)",
        head=_video_src_watch_js + _mse_stream_js,
    ) as demo:
        gr.Markdown(
            f"# Echo-WM Flash Preview: 4-Step Autoregressive, Streaming\n"
            f"Checkpoint: `{engine.checkpoint.name}` · Gemma: `{engine.gemma_path.name}`\n\n"
            f"Blocks stream into the live preview below as they're denoised+decoded; "
            f"the Result panel gets the full, correctly-assembled final video once "
            f"generation completes."
        )

        with gr.Row():
            with gr.Column(scale=1):
                case_picker = gr.Dropdown(
                    list(CASES),
                    label="Example case (fills image, prompt, action, FOV, seed)",
                    value=None,
                )
                image = gr.Image(label="First-frame image", type="filepath", height=300)

                with gr.Row():
                    prompt = gr.Textbox(
                        label="Prompt",
                        lines=4,
                        placeholder="Describe the scene, style, perspective...",
                    )
                gr.Markdown(ACTION_HELP)
                action = gr.Textbox(label="Action string", value="w-240")
                preset = gr.Dropdown(
                    list(ACTION_PRESETS),
                    label="Action preset",
                    value="forward (w-240)",
                )

                with gr.Accordion("Video Settings", open=False):
                    with gr.Row():
                        width = gr.Number(
                            label="Width (multiple of 32)", value=default_width, precision=0
                        )
                        height = gr.Number(
                            label="Height (multiple of 32)", value=default_height, precision=0
                        )
                    with gr.Row():
                        num_frames = gr.Number(
                            label="Frames (must be 1 + 8n)", value=default_num_frames, precision=0
                        )
                        fps = gr.Number(label="FPS", value=default_fps, precision=1)
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=default_seed, precision=0)
                        denoise_steps = gr.Dropdown(
                            list(STEP_PRESETS), label="Denoising steps",
                            value="(config default)",
                        )

                with gr.Accordion("Action Settings", open=False):
                    fov_deg = gr.Slider(
                        30, 120, value=action_cfg.get("fov_deg", 70.0), step=5, label="FOV (degrees)"
                    )
                    translation_speed = gr.Slider(
                        0.005, 0.1, value=action_cfg.get("translation_speed", DEFAULT_TRANSLATION_SPEED),
                        step=0.005, label="Translation speed (w/s/a/d per frame)",
                    )
                    rotation_speed = gr.Slider(
                        0.1, 3.0, value=action_cfg.get("rotation_speed_deg", DEFAULT_ROTATION_SPEED_DEG),
                        step=0.1, label="Rotation speed (°/frame, i/k/j/l)",
                    )
                    pitch_limit = gr.Slider(
                        0, 90, value=action_cfg.get("pitch_limit_deg", DEFAULT_PITCH_LIMIT_DEG),
                        step=5, label="Pitch limit (degrees)",
                    )

                with gr.Row():
                    overlay = gr.Checkbox(label="Action HUD overlay", value=True)
                    gen_audio = gr.Checkbox(label="Generate audio", value=True)

                generate_btn = gr.Button("🚀 Generate (streaming)", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown(
                    "**Live preview** (streams continuously over WebSocket -- no per-block reload). "
                    "If audio doesn't start automatically, click the video once to enable sound."
                )
                gr.HTML(
                    # No `muted` attribute here -- _mse_stream_js's connect() decides
                    # whether to mute per-generation based on whether it actually has
                    # audio, and manages the unmuted-autoplay-blocked fallback itself.
                    '<video id="live-preview-video" autoplay playsinline '
                    'style="width:100%;max-height:300px;background:#000;"></video>'
                )
                # visible=False components are not just CSS-hidden in current
                # Gradio -- they're not mounted in the DOM at all (conditional
                # {#if visible} render), so a plain JS poll against them never
                # finds anything and silently never fires. Keep this
                # genuinely visible=True (so Gradio actually renders the
                # <textarea>) and hide it purely with CSS instead (see
                # _mse_stream_js's injected <style> block below).
                stream_trigger = gr.Textbox(value="", visible=True, elem_id="stream-trigger")
                out_video = gr.Video(label="Result (final, full quality)", height=300)
                status = gr.Textbox(label="Status", lines=6, interactive=False)
                raw_file = gr.File(label="Raw video (no overlay)", interactive=False)

        case_picker.change(
            on_case, inputs=case_picker,
            outputs=[image, prompt, action, fov_deg, seed],
        )
        preset.change(on_preset, inputs=preset, outputs=action)
        generate_btn.click(
            on_generate,
            inputs=[
                image, prompt, action, seed, num_frames, fps, width, height,
                fov_deg, translation_speed, rotation_speed, pitch_limit,
                gen_audio, overlay, denoise_steps,
            ],
            outputs=[status, stream_trigger, out_video, raw_file],
            concurrency_limit=1,
        )

    return demo


def _warmup(engine, engine_kind: str) -> None:
    """Run one small, throwaway generation at startup so the 47GB of lazily-loaded
    weights and any first-call CUDA kernel compilation happen here instead of
    during the first real user request."""
    if not CASES:
        print("[warmup] No example cases found -- skipping (nothing to warm up with).", flush=True)
        return
    image_path = next(iter(CASES.values()))["image"]
    out_dir = OUTPUT_ROOT / "_warmup"
    t0 = time.time()
    print(f"[warmup] Starting warmup generation (image={image_path})...", flush=True)
    try:
        if engine_kind == "causal":
            # Kernel selection/autotuning (cuDNN/cuBLAS algorithm choice,
            # CUDA JIT compilation, etc.) is shape-dependent -- warming up
            # at a throwaway small resolution doesn't actually warm the
            # kernels a real generation needs. Confirmed on real hardware:
            # despite warmup running to completion first, the first real
            # user generation still paid a ~21s+6s cold-start hit on its
            # first two blocks (vs. ~1s/block steady-state) because warmup
            # was using 128x64 while the real config is 512x288. Use the
            # engine's actual configured width/height here so warmup pays
            # this cost once, before the server starts accepting requests,
            # instead of deferring it onto whoever generates first.
            warmup_video_cfg = engine.cfg.get("video", {})
            warmup_width = int(warmup_video_cfg.get("width", 512))
            warmup_height = int(warmup_video_cfg.get("height", 288))
            warmup_fps = float(warmup_video_cfg.get("fps", 16))
            print(
                f"[warmup] calling engine.generate() (causal, streaming) at "
                f"{warmup_width}x{warmup_height}@{warmup_fps}fps (matches real config)...",
                flush=True,
            )
            for item in engine.generate(
                image_path=image_path,
                prompt="warmup",
                action_str="w-8",
                seed=0,
                # num_frames must satisfy causal --num-frames == 1 + 8*n AND
                # (combined with video_chunk_size=3) the resulting latent
                # length must be 1 + 3*m -- together that means valid
                # non-degenerate values are 1 + 24*k. 25 (k=1) is the
                # smallest value that produces more than zero real blocks;
                # anything smaller either violates the frame-count
                # constraint outright or degenerates to zero real blocks.
                # Deliberately NOT the real config's num_frames=241: block
                # count doesn't affect per-block kernel shape (that's driven
                # by width/height/video_chunk_size, all matched above), so
                # more blocks here would only add warmup time without
                # warming anything new.
                num_frames=25,
                fps=warmup_fps,
                width=warmup_width,
                height=warmup_height,
                fov_deg=70.0,
                translation_speed=DEFAULT_TRANSLATION_SPEED,
                rotation_speed_deg=DEFAULT_ROTATION_SPEED_DEG,
                pitch_limit_deg=DEFAULT_PITCH_LIMIT_DEG,
                generate_audio=False,
                overlay=False,
                out_dir=out_dir,
                # Kernel compilation/backend dispatch happens per tensor
                # shape, not per scalar timestep value, so 1 step still
                # exercises every kernel a real (4-step) generation would --
                # just without paying for 3 extra redundant denoising passes.
                timesteps=DEFAULT_CAUSAL_TIMESTEPS[:1],
            ):
                print(f"[warmup] yielded item kind={item[0]!r} at t={time.time() - t0:.1f}s", flush=True)
            print(f"[warmup] generate() generator exhausted at t={time.time() - t0:.1f}s", flush=True)
        else:
            print("[warmup] calling engine.generate() (base, blocking)...", flush=True)
            engine.generate(
                image_path=image_path,
                prompt="warmup",
                action_str="w-8",
                seed=0,
                # Base engine has no causal frame-count constraint (that
                # rule is specific to CausalTI2VidPipeline), so a small
                # value here is fine as-is.
                num_frames=2,
                fps=24.0,
                steps=2,
                video_cfg=1.0,
                audio_cfg=1.0,
                width=128,
                height=64,
                fov_deg=70.0,
                translation_speed=DEFAULT_TRANSLATION_SPEED,
                rotation_speed_deg=DEFAULT_ROTATION_SPEED_DEG,
                pitch_limit_deg=DEFAULT_PITCH_LIMIT_DEG,
                generate_audio=False,
                overlay=False,
                out_dir=out_dir,
            )
        print(f"[warmup] Done in {time.time() - t0:.1f}s.", flush=True)
    except Exception as exc:  # noqa: BLE001 - warmup is an optimization, not a hard requirement
        print(f"[warmup] Failed after {time.time() - t0:.1f}s (continuing anyway): {exc}", flush=True)
        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=("auto", "base", "causal"), default="auto",
        help="auto picks 'causal' (streaming Flash Preview) if its checkpoint exists, else 'base'.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--gemma-path", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public gradio share link")
    parser.add_argument("--no-warmup", action="store_true",
                         help="Skip the startup warmup generation (weights would then load "
                              "lazily on the first real request instead, making it slow).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required.")

    engine_kind = args.engine
    if engine_kind == "auto":
        default_causal_ckpt = args.checkpoint or DEFAULT_CAUSAL_CHECKPOINT
        engine_kind = "causal" if default_causal_ckpt.is_file() else "base"

    device = torch.device("cuda")
    if engine_kind == "causal":
        engine = EchoWMCausalEngine(
            checkpoint=args.checkpoint or DEFAULT_CAUSAL_CHECKPOINT,
            gemma_path=args.gemma_path or DEFAULT_GEMMA,
            config_path=args.config or DEFAULT_CAUSAL_CONFIG,
            device=device,
        )
        demo = build_causal_ui(engine)
    else:
        engine = EchoWMEngine(
            checkpoint=args.checkpoint or DEFAULT_CHECKPOINT,
            gemma_path=args.gemma_path or DEFAULT_GEMMA,
            config_path=args.config or DEFAULT_CONFIG,
            device=device,
        )
        demo = build_ui(engine)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.no_warmup:
        _warmup(engine, engine_kind)

    if args.share:
        print("[server] --share is not supported in this build (we run our own uvicorn "
              "server for the live-preview WebSocket route instead of demo.launch(), which "
              "is the only thing that can set up a share tunnel) -- ignoring.", flush=True)

    # demo.launch() can't be used anymore: the live-preview player needs a
    # WebSocket route (/ws/stream/<run_id>) that Gradio's own launch() has no
    # hook for. Mount Gradio into a FastAPI app we control instead, and run
    # that app ourselves so both routes share one server/port.
    app = FastAPI()

    @app.websocket("/ws/stream/{run_id}")
    async def stream_ws(websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        with _stream_lock:
            q = _stream_queues.setdefault(run_id, asyncio.Queue())
        try:
            while True:
                data = await q.get()
                if data is None:
                    await websocket.send_text("__DONE__")
                    break
                await websocket.send_bytes(data)
        except WebSocketDisconnect:
            pass
        finally:
            with _stream_lock:
                _stream_queues.pop(run_id, None)

    @app.on_event("startup")
    async def _capture_main_event_loop() -> None:
        global _main_event_loop
        _main_event_loop = asyncio.get_running_loop()

    gr.mount_gradio_app(
        app,
        demo.queue(),
        path="/",
        allowed_paths=[str(OUTPUT_ROOT), str(EXAMPLES_DIR)],
        show_error=True,
    )

    print(f"[server] Engine: {engine_kind}", flush=True)
    print(f"[server] Serving on http://127.0.0.1:{args.port} "
          f"(forward port {args.port} if you are on a remote host)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
