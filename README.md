<p align="center">
  <img src="echo_longvideo/assets/image.png" alt="JoyAI-Echo generated video gallery" width="100%">
</p>

<div align="center">

<h1>JoyAI-Echo</h1>

<p><strong>🎬  Long-Horizon Audio-Visual Generation for Persistent Stories and Interactive Worlds</strong></p>

<p>
  <a href="https://www.researchgate.net/publication/405770309_JoyAI-Echo_Pushing_the_Frontier_of_Long_Audio-Visual_Generation"><b>📄 Paper 1.0</b></a> |
  <a href="https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-1.5-Page/blob/main/Doc/joyai-echo-15.pdf"><b>📄 Paper 1.5</b></a> |
  <a href="https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-1.5-Page/blob/main/Doc/Echo_WM.pdf"><b>📄 Echo-WM Paper</b></a> |
  <a href="https://echo-team-joy-future-academy-jd.github.io/Echo-1.5-Page/"><b>🌐 Project Page</b></a> |
  <a href="https://huggingface.co/jdopensource/JoyAI-Echo"><b>🤗 Hugging Face</b></a> |
  <a href="https://github.com/zhuang2002/ComfyUI_JoyAI_Echo"><b>🖥️ ComfyUI</b></a>
</p>

</div>

This repository holds two independent projects. Each has its own environment,
checkpoints, and entrypoint — pick the one you need and follow its README.

| Project | What it does | Guide |
|---|---|---|
| **JoyAI-Echo** (long video) | Long-horizon, multi-shot audio-visual generation. Up to ~5 minutes, with a paired audio-video memory bank carrying continuity across shots. | [`echo_longvideo/`](echo_longvideo/README.md) |
| **Echo-WM** (world model) | Omnimodal world model with camera/WASD action control. ~10 s clips from a first frame plus a prompt, video and audio together. | [`echo_wm/`](echo_wm/README.md) |

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

## Citation

If JoyAI-Echo helps your research or products, please cite:

```bibtex
@techreport{echo2026joyai,
  title        = {JoyAI-Echo: Pushing the Frontier of Long Video Generation},
  author       = {{Echo Team @ Joy Future Academy, JD}},
  institution  = {Joy Future Academy, JD},
  year         = {2026},
  month        = {May}
}
```

For Echo-WM, see the citation block in [`echo_wm/README.md`](echo_wm/README.md).

## License

This project is based on LTX-2 by Lightricks Ltd.

Portions of the original LTX-2 codebase have been modified by JD.com for academic and research purposes only.
This project is not intended for commercial use. For commercial use of LTX-2 or its derivatives, please contact Lightricks Ltd.

All original copyright, license, patent, trademark, and attribution notices from LTX-2 are retained.
This project remains subject to the LTX-2 Community License Agreement.
