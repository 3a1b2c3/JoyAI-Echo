# Model files

Model weights are downloaded separately and ignored by Git.

```text
checkpoints/
├── echo15_full_dmd/
│   ├── checkpoint.json
│   └── model.safetensors
├── echo15_fp8/
│   ├── checkpoint.json
│   └── model.safetensors
├── echo15_fp4/
│   ├── checkpoint.json
│   ├── components.safetensors
│   └── transformer_modelopt.pt
├── msst/
│   └── model_bandit_plus_dnr_sdr_11.47.chpt
└── gemma-3-12b/
```

Each directory is one public checkpoint and contains a `checkpoint.json`
manifest. The loader rejects any checkpoint name outside `echo15_full_dmd`,
`echo15_fp8`, and `echo15_fp4`. The FP4 package is standalone:
`components.safetensors` contains the VAE, vocoder and embedding processor,
while `transformer_modelopt.pt` contains the packed FP4 DiT. Callers still provide only the directory path.

Run `python scripts/setup_msst.py` to install and verify the MSST Bandit model.
Its matching configuration remains in the pinned MSST-WebUI checkout under
`third_party/MSST-WebUI/`.
