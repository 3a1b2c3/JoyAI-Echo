<p align="center">
  <img src="assets/image.png" alt="JoyAI-Echo generated video gallery" width="100%">
</p>

<div align="center">

<h1>JoyAI-Echo 1.5</h1>

<p><strong>🎬 Long-form audio-video generation with reference-driven multi-shot memory</strong></p>

<p>
  <a href="https://arxiv.org/abs/2608.23383"><b>📄 Paper 1.5</b></a> |
  <a href="https://echo-team-joy-future-academy-jd.github.io/Echo-1.5-Page/"><b>🌐 Project Page</b></a> |
  <a href="#quickstart"><b>🚀 Quickstart</b></a> |
  <a href="https://huggingface.co/jdopensource/JoyAI-Echo"><b>🤗 Model Weights</b></a> |
  <a href="Director_Agent/README.md"><b>🎬 Director Agent</b></a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.8-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.8">
  <img src="https://img.shields.io/badge/CUDA-12.8-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA 12.8">
  <img src="https://img.shields.io/badge/Release-Inference--Only-black?style=flat-square" alt="Inference only">
</p>

</div>

## 📢 Whats New

- 🎉 **JoyAI-Echo 1.5 is now available!** Code, model weights, and R2V inference are released.
- 🎬 **We also introduce Echo Director Agent This Time !**, an agentic workflow for planning and creating multi-shot videos. See [`Director_Agent/`](Director_Agent/README.md).
- 🎮 **JoyAI-Echo 1.5 brings high-quality video generation to consumer GPUs. Fire up your RTX GPU and start creating!**
- 📦 JoyAI-Echo 1.0 remains available on the [`echo1.0` archive branch](https://github.com/jd-opensource/JoyAI-Echo/tree/echo1.0).

## JoyAI-Echo 1.5

JoyAI-Echo 1.5 is an inference-only release for coherent, long-form
audio-visual generation. It combines few-step generation with a paired
cross-modal memory bank, carrying character appearance, voice, and scene
continuity across independently editable shots.



## Reference-to-video generation

Echo 1.5 supports reference-to-video (R2V) generation. Each request may
include a text prompt, an optional first-frame condition, and up to seven
ordered memory slots containing reference images and audio. This makes visual
identity, voice, and story context directly reusable from shot to shot.

A complete request schema and portable example are available in
[`schemas/r2v_request.schema.json`](schemas/r2v_request.schema.json) and
[`examples/the_last_visa/`](examples/the_last_visa/). 🕶️🍿

## Echo Director Agent

We also introduce **Echo Director Agent**, a local-first agent for planning,
generating, reviewing and assembling multi-shot videos. It turns a story idea
into an editable production workflow and submits generation jobs to an Echo 1.5
service.

See [`Director_Agent/`](Director_Agent/README.md) for installation and usage.

## Quickstart

### 1. Clone

```bash
git clone https://github.com/jd-opensource/JoyAI-Echo.git
cd JoyAI-Echo/echo_longvideo
```

All commands below are run from `echo_longvideo/`.

### 2. Install

The reference environment is Python 3.11, PyTorch 2.8 and CUDA 12.8. We
recommend using [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements.txt
python scripts/setup_msst.py
```

[`ffmpeg`](https://ffmpeg.org/download.html) must also be available on `PATH`.
For FP4 inference, install the optional NVIDIA ModelOpt dependency:

```bash
uv pip install -r requirements-fp4.txt
```

Conda users can instead run:

```bash
conda env create -f environment.yml
conda activate joyai-echo15
python scripts/setup_msst.py
```

### 3. Download model weights

Download the release weights from
[Hugging Face](https://huggingface.co/jdopensource/JoyAI-Echo) and arrange them
as follows:

| Checkpoint | Precision | Download |
| --- | --- | --- |
| `echo15_full_dmd` | BF16 | [Hugging Face](https://huggingface.co/jdopensource/JoyAI-Echo/tree/main/echo15_full_dmd) |
| `echo15_fp8` | FP8 | [Hugging Face](https://huggingface.co/jdopensource/JoyAI-Echo/tree/main/echo15_fp8) |
| `echo15_fp4` | FP4 | [Hugging Face](https://huggingface.co/jdopensource/JoyAI-Echo/tree/main/echo15_fp4) |
| `gemma-3-12b` | Text encoder | [Hugging Face](https://huggingface.co/google/gemma-3-12b-it) |

```text
checkpoints/
├── echo15_full_dmd/          # BF16 reference checkpoint
├── echo15_fp8/               # FP8 scaled-matmul checkpoint
├── echo15_fp4/               # packed ModelOpt FP4 checkpoint
├── gemma-3-12b/              # Gemma text encoder
└── msst/                     # installed by scripts/setup_msst.py
```

Each Echo checkpoint directory includes its own `checkpoint.json` manifest.
See [`checkpoints/README.md`](checkpoints/README.md) for the exact layout.

### 4. Run batch inference

The default command loads the model once and processes all R2V JSON requests
under `examples/the_last_visa/requests/`:

```bash
python inference.py --config configs/inference.bf16.yaml
```

Use FP8 or FP4 by selecting the corresponding configuration:

```bash
python inference.py --config configs/inference.fp8.yaml
python inference.py --config configs/inference.fp4.yaml
```

Outputs are written to `inference_result/<work-id>/<shot-id>/`.

## Consumer GPU support

Echo 1.5 includes low-memory profiles for consumer GPUs. They combine
layer-wise DiT weight streaming with tiled Video VAE decoding, trading some
latency for substantially lower peak VRAM usage.

```bash
# BF16 (requires substantial system RAM)
python inference.py --config configs/inference.consumer.bf16.yaml

# FP8
python inference.py --config configs/inference.consumer.fp8.yaml

# FP4 standalone (recommended)
python inference.py --config configs/inference.consumer.fp4.yaml
```

For a 24 GiB VRAM target, precompute conditioning before online generation.
Actual headroom depends on the GPU, driver and request shape.

## Local inference server

The repository includes a small local server for Echo Director Agent and other
R2V clients. The root `server.py` is its command-line entry point; the service
implementation lives in the `server/` package. It provides an in-memory queue
and dynamically keeps model weights on the GPU when memory permits.

```bash
uv pip install -r requirements-server.txt
uv run python server.py --config configs/server.consumer.yaml
```

On Linux or macOS, after setup, start it with one command:

```bash
./scripts/start_server.sh
```

On Windows, start it from Command Prompt with:

```bat
scripts\start_server.cmd
```

Extra server arguments may be appended, for example
`./scripts/start_server.sh --port 8222` or
`scripts\start_server.cmd --port 8222`.

The server YAML owns deployment settings and references a separate inference
YAML, which owns the checkpoint and pipeline settings.

See [`docs/LOCAL_SERVER.md`](docs/LOCAL_SERVER.md) for deployment options.

## Acknowledgements

We gratefully acknowledge the open-source projects that make this release
possible, especially [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3),
[Gemma](https://huggingface.co/google/gemma-3-12b-it) and
[MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.

## Citation

If JoyAI-Echo helps your research, please cite:

```bibtex
@article{duan2026joyaiecho15,
  title         = {Long-Horizon Audio-Visual Generation for Persistent Stories and Interactive Worlds},
  author        = {Duan, Nan and Huang, Haoyang and Jin, Weiyang and Li, Haoran and Li, Yaowei and Li, Yuming and Liu, Yijun and Lu, Xin and Ma, Xiaoxiao and Ma, Yanwen and Su, Yaofeng and Sun, Yilang and Wang, Haoyu and Xue, Zeyue and Zhang, Songchun and Zhuang, Junhao},
  journal       = {arXiv preprint arXiv:2608.23383},
  year          = {2026},
  eprint        = {2608.23383},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.23383}
}
```

**For academic research and non-commercial use only.**

## License

This project is based on LTX-2 by Lightricks Ltd.

Portions of the original LTX-2 codebase have been modified by JD.com for
academic and research purposes only. This project is not intended for
commercial use. For commercial use of LTX-2 or its derivatives, please contact
Lightricks Ltd.

All original copyright, license, patent, trademark, and attribution notices
from LTX-2 are retained. This project remains subject to the LTX-2 Community
License Agreement.
