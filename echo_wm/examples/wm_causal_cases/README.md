# 4-Step WM Test Cases

Each case is a 4-step first-frame I2V test from WBench Navi. `case.json`
contains only public semantic controls; model paths and generated media stay
outside this directory.

The case directory names match their WBench IDs. Each contains an `input.jpg`
and a compact `case.json` with the original WBench caption in `prompt`, plus
`action` and optional `fov_deg`/`seed`. The caption is passed directly to the
Gemma text encoder; no runtime prompt enhancement or six-field rewrite is
applied.

From the `echo_wm` directory, run one case with:

```bash
python scripts/run_wm_case_causal.py \
  --case examples/wm_causal_cases/0079 \
  --checkpoint checkpoints/echo-wm-flash.safetensors \
  --gemma-path checkpoints/gemma-3
```

Use `--list` to see available cases. Use `--dry-run` to print the resolved
`inference_wm_causal.py` command without loading a model.

Each checked-in case has four 96-frame actions: 4 seconds per action at 24 fps,
or 385 output frames including the first frame.

| Case | Motion | Input |
| --- | --- | --- |
| `0024` | yaw right/forward alternating | `input.jpg` |
| `0075` | forward four times | `input.jpg` |
| `0079` | yaw right four times | `input.jpg` |
| `0081` | pitch down/up, then backward/forward | `input.jpg` |
| `0122` | strafe left/right alternating | `input.jpg` |
| `0170` | forward/backward, strafe left, yaw right | `input.jpg` |

The shared WBench camera and bounded-cache defaults live in
`configs/inference_wm_causal.yaml`.

To run a fast validation instead of the default 16-second case:

```bash
python scripts/run_wm_case_causal.py \
  --case examples/wm_causal_cases/0079 \
  --checkpoint checkpoints/echo-wm-flash.safetensors \
  --gemma-path checkpoints/gemma-3 \
  --num-frames 25 --width 256 --height 128
```
