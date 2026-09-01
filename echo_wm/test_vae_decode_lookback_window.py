"""Empirically determine the VideoDecoder's real temporal lookback window,
to answer TROUBLESHOOTING.md item 0.5's open question: instead of guessing
a window size by hand-tracing every up_block's temporal stride/kernel
config (real risk: a wrong guess produces silently corrupted preview
frames, not a crash), decode with progressively smaller trailing windows
and check where the newest output frames stop changing relative to a
full-prefix decode -- the real decoder, not a reimplementation.

Uses a synthetic random latent tensor of realistic shape (not a real
generation) -- valid for this specific question, since the required
lookback window is a property of the network's convolution architecture
(kernel sizes, strides, causal padding), not of the input values. Do NOT
use this script's output to judge visual quality -- only to find the
window size where windowed and full decode agree numerically.

Needs a real checkpoint (VideoDecoder weights) and ideally a real GPU (CPU
works but will be slow for a full-size decoder) -- run on the GB300 box:

Run: python test_vae_decode_lookback_window.py --checkpoint checkpoints/echo-wm-flash.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "ltx-core" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "ltx-pipelines" / "src"))
from ltx_core.model.video_vae import decode_video  # noqa: E402
from ltx_pipelines.utils.model_ledger import ModelLedger  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="checkpoints/echo-wm-flash.safetensors")
parser.add_argument("--full-latent-frames", type=int, default=31, help="Latent frames for a full ~241-frame clip (1 + (241-1)/8).")
parser.add_argument("--latent-h", type=int, default=9)
parser.add_argument("--latent-w", type=int, default=16)
parser.add_argument("--tail-frames-to-check", type=int, default=3, help="How many of the newest OUTPUT frames must match to call a window size 'safe'.")
parser.add_argument("--atol", type=float, default=1e-3)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("[test] No CUDA device -- running on CPU, will be slow for a full decoder.")

dtype = torch.bfloat16
print(f"[test] Loading VideoDecoder from {args.checkpoint} ...", flush=True)
ledger = ModelLedger(dtype=dtype, device=device, checkpoint_path=args.checkpoint)
video_decoder = ledger.video_decoder()
print("[test] VideoDecoder loaded.", flush=True)

torch.manual_seed(0)
LATENT_CHANNELS = 128
full_latent = torch.randn(
    1, LATENT_CHANNELS, args.full_latent_frames, args.latent_h, args.latent_w,
    device=device, dtype=dtype,
)

print(f"[test] Decoding FULL latent ({args.full_latent_frames} latent frames) as reference...", flush=True)
with torch.inference_mode():
    full_output = next(decode_video(full_latent, video_decoder, tiling_config=None))
print(f"[test] Full decode output shape: {tuple(full_output.shape)} (f, h, w, c)", flush=True)

full_output_f = full_output.float()
tail_check = args.tail_frames_to_check
reference_tail = full_output_f[-tail_check:]

print(
    f"\n[test] Checking windowed decode vs. full decode's last {tail_check} output frame(s), "
    f"decreasing window size from {args.full_latent_frames} latent frames down to 1...\n",
    flush=True,
)

results = []
for window in range(args.full_latent_frames, 0, -1):
    windowed_latent = full_latent[:, :, -window:]
    with torch.inference_mode():
        windowed_output = next(decode_video(windowed_latent, video_decoder, tiling_config=None))
    windowed_output_f = windowed_output.float()
    if windowed_output_f.shape[0] < tail_check:
        print(f"[test] window={window}: output too short ({windowed_output_f.shape[0]} frames) to check tail -- stopping.", flush=True)
        break
    windowed_tail = windowed_output_f[-tail_check:]
    max_diff = (windowed_tail - reference_tail).abs().max().item()
    matches = torch.allclose(windowed_tail, reference_tail, atol=args.atol)
    results.append((window, max_diff, matches))
    print(f"[test] window={window:3d} latent frames: max_diff={max_diff:.6f} {'MATCH' if matches else 'differs'}", flush=True)
    if not matches and window < args.full_latent_frames:
        # Once we've found a mismatch, one more confirming step below is enough --
        # no need to keep shrinking further, the boundary is what matters.
        break

print("\n[test] Summary:", flush=True)
safe_windows = [w for w, _, m in results if m]
if safe_windows:
    min_safe = min(safe_windows)
    print(
        f"[test] Smallest window that still matched the full decode: {min_safe} latent frames "
        f"(~{min_safe * 8} output frames, per the encoder's 8x temporal compression). "
        f"Add a safety margin before using this in real code -- this is one synthetic sample, "
        f"not a proof for all inputs.",
        flush=True,
    )
else:
    print(
        "[test] No window smaller than the full latent matched within atol="
        f"{args.atol} -- either the decoder's real receptive field is close to the full "
        "clip length for this configuration, or atol is too strict for this dtype "
        "(try a looser --atol, e.g. 1e-2, given bfloat16 precision).",
        flush=True,
    )
