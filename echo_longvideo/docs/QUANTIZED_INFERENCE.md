# BF16, FP8 and FP4 inference

Echo 1.5 uses one inference pipeline with three public generator precisions.
Only model construction is precision-specific.

| Release checkpoint | Precision | Loader backend |
| --- | --- | --- |
| `echo15_full_dmd` | BF16 | complete DMD-merged model |
| `echo15_fp8` | FP8 | E4M3 weights, checkpoint scales, dynamic activation scaling and `torch._scaled_mm` |
| `echo15_fp4` | FP4 | ModelOpt LTX-2 plugin with packed FP4 transformer weights |

All three are self-contained directory checkpoints selected by one
`paths.checkpoint` value. The manifest binds the public name, precision and
internal files together.

## BF16

Use `configs/inference.bf16.yaml`. DMD is merged once while preparing the release;
runtime directly loads `echo15_full_dmd/model.safetensors`.

## FP8

Use `configs/inference.fp8.yaml`. Before model
construction, the loader reads the safetensors header and discovers every
Linear layer that has a `weight_scale`. Construction fails if that set does not
match the LTX transformer. E4M3 weights and FP32 scales retain their checkpoint
dtypes when the model moves between CPU and CUDA.

FP8 requires a PyTorch/CUDA build that provides `torch._scaled_mm`. The
reference environment is PyTorch 2.8 with CUDA 12.8.

## FP4

Install the optional backend and use `configs/inference.fp4.yaml`:

```bash
uv pip install -r requirements-fp4.txt
python inference.py --config configs/inference.fp4.yaml
```

The FP4 release contains `components.safetensors` and
`transformer_modelopt.pt`. The first file contains the embedding processor,
video/audio VAEs and vocoder. The second contains the packed FP4 DiT and its
ModelOpt graph metadata.

The loader constructs the DiT topology on the meta device, restores ModelOpt's
LTX-2 QuantLinear graph, and assigns the packed tensors directly. It never
materializes a BF16 DiT and does not require `echo15_full_dmd`.

The ModelOpt state is executable framework state and must be obtained from a
trusted release source. ModelOpt names the underlying E2M1 block-scaled format
NVFP4; that technical format name does not mean fake quantization. Native FP4
execution is hardware dependent. Do not substitute the much larger fake-quant
calibration checkpoint for the packed release artifact.

## CLI override

```bash
python inference.py \
  --config configs/inference.fp8.yaml \
  --checkpoint /models/echo15_fp8

python inference.py \
  --config configs/inference.fp4.yaml \
  --checkpoint /models/echo15_fp4
```

Run metadata records the public checkpoint, precision and exact backend. The
loader does not accept loose base, DMD, or ModelOpt paths.

## RTX 5090 with DiT layerwise offload

The following measurements use the same R2V case on a Windows RTX 5090: 241
frames at 1280x736, `resident_blocks=0`, `prefetch_blocks=1`, and tiled VAE
decode. Times are in seconds, GPU memory is the peak NVML reading, and RAM is
the peak process RSS.

| Checkpoint | Cold start | Online R2V | DMD | Tiled VAE | Peak GPU memory | Peak process RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 161.55 s | 138.07 s | 115.21 s | 8.68 s | 16.36 GiB | 70.75 GiB |
| FP8 | 163.11 s | 144.90 s | 125.33 s | 8.66 s | 17.17 GiB | 36.26 GiB |
| FP4 standalone v2 | 160.67 s | 144.09 s | 123.67 s | 8.75 s | 16.29 GiB | 26.51 GiB |

With full DiT offload, activation memory dominates the GPU peak, so all three
precisions remain near 16-17 GiB. The standalone FP4 v2 release is the default
FP4 checkpoint going forward: it preserves generation speed while reducing
peak process RAM to 26.51 GiB.

## H20 with resident DiT

The resident baseline uses one NVIDIA H20D, the same R2V request and output
shape as the RTX 5090 benchmark, no DiT layerwise offload, and tiled VAE decode.
The conditioning bundle is prepared before loading the generator, matching the
recommended online cache-hit path. Each checkpoint processes the request twice
in one server runtime; the second request is the steady-state result after both
the DiT and decoders are resident on CUDA.

| Checkpoint | Weight load | First R2V | Steady-state R2V | DMD | Tiled VAE | Peak GPU memory | Peak process RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 25.33 s | 47.67 s | 44.62 s | 39.36 s | 3.17 s | 47.57 GiB | 38.26 GiB |
| FP8 | 14.65 s | 54.77 s | 51.94 s | 46.42 s | 3.45 s | 33.12 GiB | 20.98 GiB |
| FP4 standalone v2 | 14.12 s | 51.63 s | 48.09 s | 42.50 s | 3.46 s | 25.32 GiB | 16.38 GiB |

GPU memory is the peak NVML reading during the steady-state request. Process
RAM is the highest sampled RSS during weight loading, which is the lifecycle
peak for these resident runs. The request produced 241-frame, 1280x736 H.264
video with 48 kHz stereo AAC for all three checkpoints. BF16 is the fastest
resident mode on H20; FP4 provides the lowest resident GPU and host-memory
footprint, while the current FP8 scaled-matmul backend prioritizes memory
reduction rather than latency.
