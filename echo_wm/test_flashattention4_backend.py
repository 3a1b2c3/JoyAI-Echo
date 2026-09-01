"""Decisive test: does FlashAttention-4 actually work on this GPU?

Item -10 in TROUBLESHOOTING.md flagged flash-attn-4 as a real, specific
hardware match for GB300 (SM100/103) but never tested it, over an
unresolved concern about arbitrary bias-tensor support. That concern is
moot now: item -14 confirmed every mask reaching attention in this
pipeline is an all-zero no-op, already stripped to `mask=None` before any
fast-path library is tried (`_mask_arg()` in attention.py) -- so this test
only needs to confirm a plain unmasked call works, same as the
SageAttention/FlashInfer tests already run tonight.

Install first (only beta releases exist, exact pin required):
    pip install flash-attn-4==4.0.0b28

Run: python test_flashattention4_backend.py
"""

from __future__ import annotations

import torch

print(f"[test] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[test] GPU: {torch.cuda.get_device_name(0)}, "
          f"compute capability: {torch.cuda.get_device_capability(0)}")
print(f"[test] torch version: {torch.__version__}")

try:
    import flash_attn_4  # noqa: F401 -- just checking the module name/import path
    print(f"[test] flash_attn_4 module: {flash_attn_4.__file__}")
except ImportError as exc:
    print(f"[test] flash_attn_4 (bare module name): NOT INSTALLED or different import name ({exc})")
    flash_attn_4 = None

# The PyPI package is `flash-attn-4` but the actual importable module name
# is unconfirmed -- try the most likely candidates rather than guessing once.
func = None
tried_names = []
for module_name, attr_name in [
    ("flash_attn_4", "flash_attn_func"),
    ("flash_attn_interface_4", "flash_attn_func"),
    ("flash_attn4", "flash_attn_func"),
]:
    tried_names.append(f"{module_name}.{attr_name}")
    try:
        mod = __import__(module_name, fromlist=[attr_name])
        func = getattr(mod, attr_name, None)
        if func is not None:
            print(f"[test] Found callable: {module_name}.{attr_name}")
            break
    except ImportError:
        continue

if func is None:
    print(f"[test] FlashAttention-4: NOT INSTALLED or import path differs from all tried "
          f"names ({tried_names}). If `pip install flash-attn-4==4.0.0b28` succeeded, "
          f"introspect the actual package layout: python -c \"import flash_attn_4; "
          f"print([n for n in dir(flash_attn_4) if not n.startswith('_')])\"")
    raise SystemExit(0)

if not torch.cuda.is_available():
    print("[test] No CUDA device -- can't test a real kernel call.")
    raise SystemExit(0)

device = torch.device("cuda")
batch, seq_len, heads, head_dim = 1, 1024, 32, 128
q = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=torch.bfloat16)
k = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=torch.bfloat16)
v = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=torch.bfloat16)

try:
    out = func(q, k, v, causal=False)
    torch.cuda.synchronize()
    print(f"[test] FlashAttention-4: SUCCEEDED, output shape {tuple(out.shape)}, dtype {out.dtype}")
    print("[test] Real kernel call worked -- worth wiring into "
          "ltx-core/.../attention.py as a real AttentionCallable (mirror "
          "the SageAttention/FlashAttention2/3 classes) and re-running the "
          "block-timing benchmark against the SDPA baseline.")
except Exception as exc:  # noqa: BLE001 - report whatever the real kernel call raises
    print(f"[test] FlashAttention-4: FAILED on real kernel call: {type(exc).__name__}: {exc}")
