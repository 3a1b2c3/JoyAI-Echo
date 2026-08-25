# Checkpoints

This directory holds the weights consumed by `inference.py`. Files are
**not** tracked by git — see `.gitignore`. The release authors will publish
download links separately.

Expected layout (matches `configs/inference.yaml`):

```
checkpoints/
├── echo-longvideo-release.safetensors   # long-video release model
└── gemma-3-12b/                         # Gemma 3 12B text encoder
    ├── config.json
    ├── tokenizer.json
    └── ...
```

If your weights live elsewhere, override the paths in `configs/inference.yaml`
or via CLI flags (e.g. `python inference.py --checkpoint /path/to/model.safetensors`).
