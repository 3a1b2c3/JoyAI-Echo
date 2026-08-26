<p align="center">
  <img src="echo_longvideo/assets/image.png" alt="JoyAI-Echo generated video gallery" width="100%">
</p>

<div align="center">

<h1>JoyAI-Echo</h1>

<p><strong>🎬  Long-Horizon Audio-Visual Generation for Persistent Stories and Interactive Worlds</strong></p>

<p>
  <a href="https://www.researchgate.net/publication/405770309_JoyAI-Echo_Pushing_the_Frontier_of_Long_Audio-Visual_Generation"><b>📄 Paper 1.0</b></a> |
  <a href="https://arxiv.org/abs/2608.23383"><b>📄 Paper 1.5</b></a> |
  <a href="https://arxiv.org/abs/2608.23189"><b>📄 Echo-WM Paper</b></a> |
  <a href="https://echo-team-joy-future-academy-jd.github.io/Echo-1.5-Page/"><b>🌐 Project Page</b></a> |
  <a href="https://huggingface.co/jdopensource/JoyAI-Echo"><b>🤗 Long Video Hugging Face</b></a>
</p>
<p>
  <a href="https://huggingface.co/Echo-Team/Echo-WM"><b>🤗 World Model Hugging Face</b></a> |
  <a href="https://github.com/zhuang2002/ComfyUI_JoyAI_Echo"><b>🖥️ ComfyUI</b></a>
</p>

</div>

This repository holds two independent projects. Each has its own environment,
checkpoints, and entrypoint — pick the one you need and follow its README.

| Project | What it does | Guide |
|---|---|---|
| **Echo-LongVideo** (long video) | Long-horizon, multi-shot audio-visual generation. Up to ~5 minutes, with a paired audio-video memory bank carrying continuity across shots. | [`echo_longvideo/`](echo_longvideo/README.md) |
| **Echo-WM** (world model) | Omnimodal world model for generative media that responds to continuous navigation while video, environmental sound, music, and speech evolve together. | [`echo_wm/`](echo_wm/README.md) |

```text
JoyAI-Echo/
+-- echo_longvideo/   # long-video generation: inference.py, configs/, prompts/, ltx-*
`-- echo_wm/          # world model: inference_wm.py, Gradio demo, bundled ltx-*
```

The two do not share a Python environment or a checkpoint directory. `echo_wm/`
bundles its own copy of `ltx-core` and `ltx-pipelines`, so installing one project
never affects the other.

## Quickstart

Long video:

```bash
cd echo_longvideo
conda env create -f environment.yml && conda activate echo-long
```

World model:

```bash
cd echo_wm
conda create -n echo-wm python=3.11 -y && conda activate echo-wm
pip install -r requirements.txt
```

Checkpoints are downloaded separately in both cases. See each README for the
exact files and paths.

**For academic research and non-commercial use only.**

## Roadmap

The current release is built on **LTX-2.3**. Next we are bringing up **LTX-2.5**
as the backbone, then the serving stack around it.

| | Item | Status | Notes |
|---|---|---|---|
| **Backbone** | LTX-2.3 Base | ✅ shipped | Bidirectional audio-visual DiT used by Echo-LongVideo and Echo-WM Base |
| | LTX-2.3 Causal / Flash | ✅ shipped | Chunk-causal attention, KV-cache rollout, 4-step student — see [`echo_wm/README_CAUSAL.md`](echo_wm/README_CAUSAL.md) |
| | LTX-2.5 Base | 🗓️ planned | Load official LTX-2.5 weights (Gemma 4 TE, 2.5 VAE / DiT) into the existing bidirectional path |
| | LTX-2.5 Causal | 🗓️ planned | Same causal / Flash recipe on the 2.5 backbone: block-causal masks, sink+FIFO cache, few-step distillation |
| **Accel** | Few-step distillation | 🗓️ planned | Push 2.5 Base and Causal onto a shared consistency / DMD student |
| | Attention kernels | 🗓️ planned | FlashAttention / FlashInfer for video, audio, and UCPE branches |
| | KV-cache infra | 🗓️ planned | Paged / variable-length cache, RoPE + UCPE rebase on eviction |
| | Runtime compile | 🗓️ planned | FP8 + `torch.compile` / TensorRT execution for the DiT forward |
| | Serving | 🗓️ planned | Multi-GPU data-parallel inference and host-side prep overlap |

```text
LTX-2.3  ── Base ✅ ── Causal / Flash ✅
                │
                ▼
LTX-2.5  ── Base 🗓️ ── Causal 🗓️ ── Accel infra 🗓️
```

Nothing in the planned rows is available in this repo yet. Tracking only —
APIs and configs will land with the corresponding drop.

## Citation

If JoyAI-Echo helps your research or products, please cite:

```bibtex
@article{li2026joyai,
  title={JoyAI-Echo: Pushing the Frontier of Long Audio-Visual Generation},
  author={Li, Haoran and Li, Fredreic and Ma, Shichen and Huang, Jie and Liu, Yijun and Shi, Jiaqi and Ma, Yanwen},
  year={2026}
}

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

## License

This project is based on LTX-2 by Lightricks Ltd.

Portions of the original LTX-2 codebase have been modified by JD.com for academic and research purposes only.
This project is not intended for commercial use. For commercial use of LTX-2 or its derivatives, please contact Lightricks Ltd.

All original copyright, license, patent, trademark, and attribution notices from LTX-2 are retained.
This project remains subject to the LTX-2 Community License Agreement.
