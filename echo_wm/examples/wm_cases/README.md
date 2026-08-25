# WM Test Cases

Each case is a single first-frame I2V test. `case.json` contains only public
semantic controls; model paths and generated media stay outside this directory.

The case directory names are intentionally short. Each contains an `input.png`
and a compact `case.json` with `prompt`, `action`, and optional `fov_deg`/`seed`.

Run one case with:

```bash
python echo_wm/scripts/run_wm_case.py \
  --case echo_wm/examples/wm_cases/0004 \
  --checkpoint echo_wm/checkpoints/echo-wm-base.safetensors \
  --gemma-path echo_wm/checkpoints/gemma-3
```

Use `--list` to see available cases. Use `--dry-run` to print the resolved
`inference_wm.py` command without loading a model.

The checked-in cases are intentionally single-shot I2V inputs:

| Case | Motion | Input |
| --- | --- | --- |
| `0004` | indoor forward + left/right navigation | `input.png` |
| `0009` | outdoor forward + left/right navigation | `input.png` |
| `0010` | green canyon, forward + left/right navigation | `input.png` |

To run a fast validation instead of the default 10-second generation:

```bash
python echo_wm/scripts/run_wm_case.py --case echo_wm/examples/wm_cases/0004 \
  --checkpoint echo_wm/checkpoints/echo-wm-base.safetensors \
  --gemma-path echo_wm/checkpoints/gemma-3 --num-frames 9 --width 256 --height 144 --steps 4
```
