# Echo-WM: Open and Enterable Omnimodal World Models

<div align="center">

**An open audio-visual world model for controllable, persistent, and interactive world generation.**

[Model Weights](https://huggingface.co/Echo-Team/Echo-WM) ·
[Project Page](https://echo-team-joy-future-academy-jd.github.io/Echo-1.5-Page/wm/) 

</div>

## 🌍 Overview

**Echo-WM** is an omnimodal world model for generative media that responds to continuous navigation while video, environmental sound, music, and speech evolve together.

### Public Base Inference

| Model | Horizon | Input | Control | Output |
|---|---:|---|---|---|
| **Echo-WM Base** | ~10 s | First-frame image + prompt | Action DSL / pure camera control | Video + audio |




## 🚀 Environment Setup

We recommend a dedicated Python environment:

```bash
conda create -n echo-wm python=3.11 -y
conda activate echo-wm
```

Install PyTorch for the CUDA version used by your machine, then install the WM
requirements:

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128

cd /path/to/JoyAI-Echo/echo_wm
pip install -r requirements.txt
```

Verify CUDA before loading the checkpoint:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`requirements.txt` caps `transformers` below 5.0 and `gradio` below 6.0. Both
bounds are required: the bundled text encoder expects the `transformers` 4.x
Siglip layout, and `gradio` 6.x pulls a `huggingface-hub` release that conflicts
with `transformers` 4.x. Do not lift these bounds without updating
`ltx-core/src/ltx_core/text_encoders/`.

`xformers` is listed as an optional acceleration dependency. It is not required
to validate the command-line entrypoint or run the focused tests.

### Verified Versions

This release was validated end to end on:

| Package | Version |
|---|---|
| Python | 3.11 |
| `torch` / `torchaudio` | 2.9.1+cu128 |
| `torchvision` | 0.24.1+cu128 |
| `transformers` | 4.57.6 |
| `gradio` | 5.32.0 |
| `numpy` | 2.4.6 |

Hardware: single NVIDIA H200. A 241-frame clip peaks near 50 GB of device
memory, so a 24 GB card requires a shorter `--num-frames`.

## 📥 Download Checkpoints

The public inference entrypoint expects both files below:

```text
echo_wm/checkpoints/echo-wm-base.safetensors
echo_wm/checkpoints/gemma-3/
```

Download the complete public checkpoint bundle:

```bash
cd /path/to/JoyAI-Echo/echo_wm
python -c "from huggingface_hub import snapshot_download; snapshot_download('Echo-Team/Echo-WM', local_dir='checkpoints')"
```

If your environment provides the Hugging Face CLI, the equivalent command is:

```bash
hf download Echo-Team/Echo-WM --local-dir checkpoints
```

Before inference, check that these paths exist:

```bash
test -f checkpoints/echo-wm-base.safetensors
test -d checkpoints/gemma-3
```

## 🧪 Validate The Installation

Run the focused WM tests:

```bash
cd /path/to/JoyAI-Echo/echo_wm
pip install pytest
python -m pytest -q tests/test_echo_wm.py
```

Check the public CLI without loading a model:

```bash
cd /path/to/JoyAI-Echo/echo_wm
python inference_wm.py --help
```

List the checked-in examples:

```bash
cd /path/to/JoyAI-Echo/echo_wm
python scripts/run_wm_case.py --list
```


## 💻 Inference

### Recommended: run a checked-in case

The checked-in cases include their own `input.png`, six-field prompt, action
string, FOV, and seed. This is the safest way to verify a complete setup:

```bash
cd /path/to/JoyAI-Echo/echo_wm
python scripts/run_wm_case.py \
  --case examples/wm_cases/0010 \
  --checkpoint checkpoints/echo-wm-base.safetensors \
  --gemma-path checkpoints/gemma-3 \
  --output-dir outputs/wm_cases
```

The output is:

```text
outputs/wm_cases/0010/result.mp4
outputs/wm_cases/0010/result.json
```

To run the other cases, replace `0010` with `0004` or `0009`.

### Example: image + prompt + action string

The repository examples are:

```text
examples/wm_cases/0004/input.png
examples/wm_cases/0009/input.png
examples/wm_cases/0010/input.png
```

For example:

```bash
cd /path/to/JoyAI-Echo/echo_wm
python inference_wm.py \
  --image examples/wm_cases/0010/input.png \
  --prompt "Environment: A clear green river runs through a lush fantasy canyon.\n\nCharacter: A solitary adventurer stands on the shore, seen from behind.\n\nStyle: Painterly high-end fantasy environment art.\n\nPerspective: Wide third-person rear view at standing height.\n\nSounds: Water ripples, footsteps, birds, and soft strings.\n\nSpeech: None." \
  --action-str "w-60,a-60,w-60,d-60" \
  --checkpoint checkpoints/echo-wm-base.safetensors \
  --gemma-path checkpoints/gemma-3 \
  --fov-deg 70 \
  --video-cfg 4.0 \
  --audio-cfg 2.0 \
  --output outputs/result.mp4
```

The prompt must use the six fields described in
[`PROMPT_SKILL.md`](PROMPT_SKILL.md): `Environment`, `Character`,
`Style`, `Perspective`, `Sounds`, and `Speech`.

### Optional controls

```text
--negative-prompt TEXT   Override the configured negative prompt
--auto-fov               Estimate FOV with the optional MoGe-2 helper
--fov-deg FLOAT          Use an explicit horizontal FOV (default: 70)
--video-cfg FLOAT        Video guidance scale (default: 4.0)
--audio-cfg FLOAT        Audio guidance scale (default: 2.0)
--no-audio               Write a video-only MP4
--action-overlay         Also write a separate <name>_action.mp4 HUD copy
--steps INT              Override the default 30 inference steps
--seed INT               Set the random seed
```

`--auto-fov` starts `helpers/moge_fov.py` as a subprocess. It is independent
of prompt generation. The public release has no `--auto-prompt` or VLM prompt
helper.

## 🎮 Action Control

Each Action DSL segment is `<keys>-<frames>`, and segments are joined by commas:

```text
w/s       forward / backward
a/d       strafe left / right
i/k       pitch up / down
j/l       yaw left / right
none      hold the camera still
```

Keys can be combined. Example:

```text
w-60,wj-60,w-60,d-60
```

For the default 241-frame clip, use approximately 240 action frames. The action
condition passed to the model contains only `ucpe_viewmats` and `ucpe_Ks`.

## 🖥️ Web Demo

Launch the full Gradio interface from the WM directory:

```bash
cd /path/to/JoyAI-Echo/echo_wm
CHECKPOINT=checkpoints/echo-wm-base.safetensors \
GEMMA_PATH=checkpoints/gemma-3 \
PORT=7860 \
./run_gradio.sh
```

Open `http://localhost:7860`. The interface accepts a first-frame image, a
six-field prompt, an Action DSL string, FOV/action settings, separate video/audio
CFG controls, and optional audio/overlay output.

## 🖼️ Example Gallery

| Case | Scene | Action | Seed |
|---|---|---|---:|
| `0004` | Warm whimsical interior with two figures | `w-60,a-60,w-60,d-60` | 4 |
| `0009` | Red torii gate beside reflective water | `w-60,a-60,w-60,d-60` | 9 |
| `0010` | Green river through a fantasy canyon | `w-60,a-60,w-60,d-60` | 34 |

Run all cases across the GPUs you have:

```bash
cd /path/to/JoyAI-Echo/echo_wm
GPU_LIST=0,1,2 bash scripts/run_wm_cases_multigpu.sh
```

The GPU count does not have to match the case count. Cases are split round-robin
across `GPU_LIST`, and each GPU works through its own share one at a time. With
three cases on two GPUs, the first GPU runs cases 1 and 3 in sequence while the
second runs case 2. A single GPU runs them all serially:

```bash
GPU_LIST=0,1 bash scripts/run_wm_cases_multigpu.sh   # 3 cases, 2 GPUs
GPU_LIST=0 bash scripts/run_wm_cases_multigpu.sh     # 3 cases, serial
```

Supported environment variables:

```text
GPU_LIST        Comma-separated GPU indices (default: 0,1,2)
CASES           Subset of cases to run (default: every case directory found)
PYTHON_BIN      Interpreter to use (default: python3)
ACTION_OVERLAY  Set to any value to also write the HUD copies
```

The script calls `python3` by default. If your environment interpreter is not on
`PATH` under that name, point it at one explicitly:

```bash
GPU_LIST=0,1,2 PYTHON_BIN="$(which python)" bash scripts/run_wm_cases_multigpu.sh
```

To run a subset, or to also create the separate HUD copies:

```bash
CASES="0009 0010" GPU_LIST=0,1 bash scripts/run_wm_cases_multigpu.sh
GPU_LIST=0,1,2 ACTION_OVERLAY=1 bash scripts/run_wm_cases_multigpu.sh
```

Outputs are written under `outputs/wm_cases_multigpu/<case>/`, with a
per-case `run_gpu<N>.log`. The script exits non-zero if any case fails, after
letting the remaining cases finish.

Only one inference process runs per GPU at a time. Keep it that way: loading a
checkpoint is host-memory hungry, and several concurrent loads on one machine can
trip a container memory limit even when device memory is fine.



## ⚙️ Configuration

Public semantic defaults are in `configs/inference_wm.yaml`:

```yaml
video:
  width: 1280
  height: 704
  num_frames: 241
  fps: 24
  steps: 30
  video_cfg: 4.0
  audio_cfg: 2.0

action:
  translation_speed: 0.05
  rotation_speed_deg: 0.5
  pitch_limit_deg: 60.0
  fov_deg: 70.0
```

Private training normalization, scale factors, trainer YAML, and data-filtering
settings are intentionally excluded from the public configuration.




## 📄 Citation

```bibtex
@article{echo_wm,
  title   = {Echo-WM: Open and Enterable Omnimodal World Models},
  author  = {Echo Team},
  journal = {arXiv preprint},
  year    = {2026}
}
```


