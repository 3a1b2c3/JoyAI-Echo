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
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import gradio as gr
import torch
import yaml

# Setup paths
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
for package in ("ltx-core/src", "ltx-causal/src", "ltx-pipelines/src"):
    sys.path.insert(0, str(ROOT / package))

from ltx_core.components.guiders import MultiModalGuiderParams  # noqa: E402
from ltx_core.types import Audio  # noqa: E402
from ltx_causal import CausalCacheConfig, DEFAULT_CAUSAL_TIMESTEPS  # noqa: E402
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline  # noqa: E402
from ltx_pipelines.causal_ti2vid import CausalTI2VidPipeline  # noqa: E402
from ltx_core.model.video_vae.tiling import TilingConfig  # noqa: E402
from ltx_core.model.video_vae.video_vae import get_video_chunks_number  # noqa: E402
from ltx_pipelines.utils.args import ImageConditioningInput  # noqa: E402
from ltx_pipelines.utils.media_io import encode_video  # noqa: E402

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
    ):
        """Generator. Yields ("block", index, total, block_video_path) as each
        block finishes, then a final ("done", video_path, overlaid_path_or_None,
        timing).

        Runs the (blocking) pipeline call on a background thread and relays its
        on_block callbacks through a queue, since a callback fired from inside
        a blocking call cannot itself yield from this generator's frame.
        """
        timing: dict[str, float] = {}
        parse_action_string(action_str)

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

        def on_block(block_index: int, total_blocks: int, video_chunk, audio_chunk) -> None:
            block_path = blocks_dir / f"block_{block_index:03d}.mp4"
            encode_video(
                video=video_chunk,
                fps=int(fps),
                audio=audio_chunk if generate_audio else None,
                output_path=str(block_path),
                video_chunks_number=1,
            )
            result_queue.put(("block", block_index, total_blocks, block_path))

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
                    timesteps=self.timesteps,
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
            item = result_queue.get()
            if item[0] == "block":
                yield item
            elif item[0] == "error":
                thread.join()
                raise item[1]
            else:
                _, video, audio = item
                break
        thread.join()
        timing["generate"] = time.time() - t0

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
        yield msg, str(shown), str(video_path)

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
    # UI defaults, deliberately lower than the config file's 1280x704 (used by
    # the CLI/other scripts) — same aspect ratio, ~51% the pixel count, for
    # faster iteration. Still a multiple of 32 (required by assert_resolution)
    # and adjustable in the UI below, not a model-imposed fixed size.
    default_width = 896
    default_height = 512
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
        gen_audio, overlay,
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

        yield (
            f"⏳ Streaming generation started…\naction=[{action_str}] seed={int(seed)}"
        ), None, None, None

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
            ):
                if item[0] == "block":
                    _, block_index, total_blocks, block_path = item
                    yield (
                        f"⏳ Block {block_index + 1}/{total_blocks} ({time.time() - t0:.1f}s elapsed)…"
                    ), str(block_path), None, None
                else:
                    _, video_path, overlaid_path, timing = item
                    shown = overlaid_path or video_path
                    parts = "  ".join(f"{k}={v:.1f}s" for k, v in timing.items())
                    msg = f"✅ Done in {time.time() - t0:.1f}s ({parts}).\n  video: {video_path.name}"
                    if overlaid_path:
                        msg += f"\n  overlay: {overlaid_path.name}"
                    yield msg, str(shown), str(shown), str(video_path)
        except Exception as e:
            yield f"❌ Generation failed: {e}\n{traceback.format_exc()[-800:]}", None, None, None

    with gr.Blocks(title="Echo-WM Flash Preview (Streaming)") as demo:
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
                    seed = gr.Number(label="Seed", value=default_seed, precision=0)

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
                stream_video = gr.Video(
                    label="Live preview (updates block-by-block; each block replaces the last)",
                    height=300, autoplay=True,
                )
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
                gen_audio, overlay,
            ],
            outputs=[status, stream_video, out_video, raw_file],
            concurrency_limit=1,
        )

    return demo


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

    print(f"[server] Engine: {engine_kind}", flush=True)
    print(f"[server] Serving on http://127.0.0.1:{args.port} "
          f"(forward port {args.port} if you are on a remote host)", flush=True)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(OUTPUT_ROOT), str(EXAMPLES_DIR)],
        show_error=True,
    )


if __name__ == "__main__":
    main()
