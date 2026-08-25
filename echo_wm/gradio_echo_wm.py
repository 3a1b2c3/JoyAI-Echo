#!/usr/bin/env python3
"""Gradio app for Echo-WM world model inference.

Features:
  - Image upload or selection from examples
  - Six-field cinematic prompt from PROMPT_SKILL.md
  - Action string input with presets (WASD + camera controls)
  - Optional Genie-style HUD overlay
  - Video + audio generation

Run:
    CUDA_VISIBLE_DEVICES=0 python gradio_echo_wm.py
    # then open http://0.0.0.0:7860
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import gradio as gr
import torch
import yaml

# Setup paths
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
for package in ("ltx-core/src", "ltx-pipelines/src"):
    sys.path.insert(0, str(ROOT / package))

from ltx_core.components.guiders import MultiModalGuiderParams  # noqa: E402
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline  # noqa: E402
from ltx_core.model.video_vae.tiling import TilingConfig  # noqa: E402
from ltx_core.model.video_vae.video_vae import get_video_chunks_number  # noqa: E402
from ltx_pipelines.utils.args import ImageConditioningInput  # noqa: E402
from ltx_pipelines.utils.media_io import encode_video  # noqa: E402

from helpers.action_condition import (  # noqa: E402
    action_config,
    build_action_condition,
    build_action_trajectory,
)
from helpers.action_camera import (  # noqa: E402
    DEFAULT_PITCH_LIMIT_DEG,
    DEFAULT_ROTATION_SPEED_DEG,
    DEFAULT_TRANSLATION_SPEED,
    parse_action_string,
)
from helpers.action_overlay import overlay_genie_on_video  # noqa: E402

# Default paths
DEFAULT_CONFIG = ROOT / "configs" / "inference_wm.yaml"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "echo-wm-base.safetensors"
DEFAULT_GEMMA = ROOT / "checkpoints" / "gemma-3"
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

        self.pipeline = TI2VidOneStagePipeline(
            checkpoint_path=str(checkpoint),
            gemma_root=str(gemma_path),
            loras=(),
            device=device,
            action_config=None,  # Will be set per generation
        )
        print("[engine] Ready.", flush=True)

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



def build_ui(engine: EchoWMEngine) -> gr.Blocks:
    """Build Gradio interface."""
    run_counter = {"n": 0}

    def on_preset(name: str):
        return gr.update(value=ACTION_PRESETS.get(name, "w-240"))

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
                        width = gr.Number(label="Width", value=1280, precision=0)
                        height = gr.Number(label="Height", value=704, precision=0)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gemma-path", type=Path, default=DEFAULT_GEMMA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public gradio share link")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required.")

    device = torch.device("cuda")
    engine = EchoWMEngine(
        checkpoint=args.checkpoint,
        gemma_path=args.gemma_path,
        config_path=args.config,
        device=device,
    )

    demo = build_ui(engine)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(OUTPUT_ROOT), str(EXAMPLES_DIR)],
        show_error=True,
    )


if __name__ == "__main__":
    main()
