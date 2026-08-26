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

| Case | Scene | Action |
| --- | --- | --- |
| `0004` | whimsical interior | `w-60,a-60,w-60,d-60` |
| `0009` | autumn torii grove | `w-60,a-60,w-60,d-60` |
| `0010` | green fantasy canyon | `w-60,a-60,w-60,d-60` |
| `0011` | alpine wingsuit flight | `w-60,wl-60,wj-60,w-60` |
| `0012` | jungle river, first-person bird | `w-60,wl-60,w-60,wj-60` |
| `0013` | meadow cabin, dog from behind | `w-60,wl-60,wj-60,w-60` |

`0004`-`0010` strafe with `a`/`d`; `0011`-`0013` hold `w` and steer with `l`/`j`,
so forward motion and yaw are combined in the same segment.

To run a fast validation instead of the default 10-second generation:

```bash
python echo_wm/scripts/run_wm_case.py --case echo_wm/examples/wm_cases/0004 \
  --checkpoint echo_wm/checkpoints/echo-wm-base.safetensors \
  --gemma-path echo_wm/checkpoints/gemma-3 --num-frames 9 --width 256 --height 144 --steps 4
```
