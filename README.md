<p align="center">
  <img src="assets/teaser.png" alt="JoyAI-Echo generated video gallery" width="100%">
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

## 📰 News

- 🚀 **2026-08-28** — Released [JoyAI-Echo 1.5 / Echo-LongVideo](echo_longvideo/README.md), including long-horizon generation, consumer-GPU inference profiles, and the Director Agent.
- 🌍 **2026-08-26** — Released [Echo-WM](echo_wm/README.md), our omnimodal world model for interactive audio-visual generation.
- 🎬 **2026-06-22** — Released [JoyAI-Echo 1.0](https://github.com/jd-opensource/JoyAI-Echo/tree/echo1.0), now preserved on the `echo1.0` archive branch.

This repository holds two independent projects. Each has its own environment,
checkpoints, and entrypoint — pick the one you need and follow its README.

| Project | What it does | Guide |
|---|---|---|
| **Echo-LongVideo** (long video) | Long-horizon, multi-shot audio-visual generation. Up to ~5 minutes, with a paired audio-video memory bank carrying continuity across shots. | [`echo_longvideo/`](echo_longvideo/README.md) |
| **Echo-WM** (world model) | Omnimodal world model for generative media that responds to continuous navigation while video, environmental sound, music, and speech evolve together. | [`echo_wm/`](echo_wm/README.md) |

```text
JoyAI-Echo/
├── echo_longvideo/   # long-video generation: inference.py, configs/, prompts/, ltx-*
└── echo_wm/          # world model: inference_wm.py, Gradio demo, bundled ltx-*
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

## Echo-WM Roadmap

Echo-WM is on **LTX-2.3** today. Next we move Base and Causal onto **LTX-2.5**,
then cut long-rollout cost with sparse attention and a tighter cache / runtime
stack.

### Backbone

- [x] **LTX-2.3 · Base** — bidirectional audio-visual DiT used by Echo-WM Base (~10 s).
- [x] **LTX-2.3 · Flash Preview / Causal** — current public preview with chunk-causal attention, KV-cache rollout, and 4-step inference. See [`echo_wm/README_CAUSAL.md`](echo_wm/README_CAUSAL.md).
- [ ] **LTX-2.5 · Base** — load official LTX-2.5 weights (Gemma 4 TE, 2.5 VAE / DiT) into the existing bidirectional path.
- [ ] **LTX-2.5 · Causal** — the same Flash recipe on 2.5: block-causal masks, sink+FIFO cache, few-step student.

### Accel

- [ ] **Sparse attention** — SageAttention and similar sparse / low-bit kernels on video, audio, and UCPE branches.
- [ ] **FlashAttention / FlashInfer** — fused attention for long causal windows without blowing up HBM.
- [ ] **Paged KV-cache** — variable-length cache so rollouts stay bounded; rebase RoPE and UCPE when tokens evict.
- [ ] **FP8 / TensorRT** — compile the DiT forward at lower precision for decode-time throughput.

## Citation

If JoyAI-Echo helps your research or products, please cite:

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

@article{li2026joyai,
  title  = {JoyAI-Echo: Pushing the Frontier of Long Audio-Visual Generation},
  author = {Li, Haoran and Li, Fredreic and Ma, Shichen and Huang, Jie and Liu, Yijun and Shi, Jiaqi and Ma, Yanwen},
  year   = {2026}
}
```

## License

This project is based on LTX-2 by Lightricks Ltd.

Portions of the original LTX-2 codebase have been modified by JD.com for academic and research purposes only.
This project is not intended for commercial use. For commercial use of LTX-2 or its derivatives, please contact Lightricks Ltd.

All original copyright, license, patent, trademark, and attribution notices from LTX-2 are retained.
This project remains subject to the LTX-2 Community License Agreement.
