<div align="center">

<h1>Echo-WM: Open and Enterable Omnimodal World Models</h1>

**An open audio-visual world model for controllable, persistent, and interactive world generation.**

[Model Weights](https://huggingface.co/Echo-Team/Echo-WM) ·
[Project Page](https://echo-team-joy-future-academy-jd.github.io/Echo-1.5-Page/wm/)

</div>

## 🌍 Overview

**Echo-WM** is an omnimodal world model for generative media that responds to continuous navigation while video, environmental sound, music, and speech evolve together.

### Public Inference

| Model | Horizon | Input | Control | Output |
|---|---:|---|---|---|
| **Echo-WM Base** | ~10 s | First-frame image + prompt | Action DSL / pure camera control | Video + audio |
| **Echo-WM Flash** | autoregressive | First-frame image + prompt | Action DSL / pure camera control | Video + audio |

Base-model instructions continue below. For the 4-step Flash model,
DMD-distilled guidance, and bounded cache options, see
**[README_CAUSAL.md](README_CAUSAL.md)**.

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

## 📥 Download Checkpoints

Download the Echo-WM checkpoints and the Gemma 3 text encoder:

```bash
cd /path/to/JoyAI-Echo/echo_wm

hf download Echo-Team/Echo-WM --local-dir checkpoints
hf download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir checkpoints/gemma-3
```

Gemma 3 is a gated repository: accept the license on its model page and run
`hf auth login` first. The text encoder runs in bfloat16, so use the
`-qat-q4_0-unquantized` weights above — not the quantized Q4_0 files.

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

To run the other cases, replace `0010` with any other case name. Use
`--list` to print them all.

### Example: image + prompt + action string

The repository examples are:

```text
examples/wm_cases/0004/input.png   magic workshop interior
examples/wm_cases/0010/input.png   limestone canyon pool
examples/wm_cases/0011/input.png   alpine wingsuit flight
examples/wm_cases/0013/input.png   meadow cabin, dog from behind
examples/wm_cases/0014/input.png   giant piano ridge above clouds
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
--no-audio               Write a video-only MP4 (audio is on by default)
--no-action-overlay      Skip the HUD copy. A separate <name>_action.mp4 with the
                         WASD/rotation HUD is written by default.
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
six-field prompt, an Action DSL string, FOV/action settings, and separate
video/audio CFG controls. Audio and the action HUD overlay are both enabled by
default; each has a checkbox to turn it off.

## 🚦 Multi-GPU Runs

Run the checked-in cases across the GPUs you have. Cases are split round-robin,
one inference process per GPU at a time, so the GPU count need not match the
case count:

```bash
cd /path/to/JoyAI-Echo/echo_wm
GPU_LIST=0,1,2 bash scripts/run_wm_cases_multigpu.sh
CASES="0010 0014" GPU_LIST=0 bash scripts/run_wm_cases_multigpu.sh   # subset, serial
```

```text
GPU_LIST        Comma-separated GPU indices (default: 0,1,2)
CASES           Subset of cases to run (default: every case directory found)
PYTHON_BIN      Interpreter to use (default: python3)
ACTION_OVERLAY  Write the HUD copies (default: 1; set to 0 to skip them)
```

Outputs land in `outputs/wm_cases_multigpu/<case>/` with a per-case
`run_gpu<N>.log`. Failures are reported at the end, after the remaining cases
finish. Don't raise the per-GPU concurrency: loading a checkpoint is
host-memory hungry, and parallel loads can trip a container memory limit even
when device memory is fine.

## 📄 Citation

```bibtex
@article{zhang2026echowm,
  title         = {EchoWM: Open and Enterable Omnimodal World Models},
  author        = {Zhang, Songchun and Li, Yaowei and Zhuang, Junhao and Jin, Weiyang and Wang, Haoyu and Lu, Xin and Sun, Yilang and Zhang, Shiyi and Li, Haoran and Ma, Xiaoxiao and Li, Yuming and Liu, Yijun and Su, Yaofeng and Ma, Yanwen and Wu, Haoyu and Su, Zihan and Ma, Yue and Zhang, Lvmin and Huang, Haoyang and Xue, Zeyue and Rao, Anyi and Duan, Nan},
  journal       = {arXiv preprint arXiv:2608.23189},
  year          = {2026},
  eprint        = {2608.23189},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.23189}
}
```
