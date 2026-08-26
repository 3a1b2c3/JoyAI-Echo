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

| Case | Scene |
| --- | --- |
| `0004` | magic workshop interior |
| `0010` | limestone canyon pool |
| `0011` | alpine wingsuit flight |
| `0013` | meadow cabin, dog from behind |
| `0014` | giant piano ridge above clouds |

Each case's camera route is in its own `case.json`. Some strafe with `a`/`d`;
others hold `w` and steer with `l`/`j`, combining forward motion and yaw.