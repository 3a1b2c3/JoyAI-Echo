"""Decisive test: does SageAttention actually work on this GPU?

Mirrors test_attention_backend.py's pattern (used tonight for xformers/
FlashAttention/FlashInfer) -- run this after any install/build attempt to
get one clear answer (import succeeded + real kernel call succeeded, or
the exact error) instead of a multi-round debugging session.

Known caveat as of tonight's research: SageAttention's kernel source has
no compute-capability 10.0/10.3 (GB300/Blackwell-datacenter) target
defined yet (open GitHub issue #237) -- if that's still true, this will
fail with a "no kernel" style error regardless of how it was installed,
since the problem is missing kernel *source*, not a build/packaging
issue like xformers turned out to be.

Run: python test_sageattention_backend.py
"""

from __future__ import annotations

import torch

print(f"[test] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[test] GPU: {torch.cuda.get_device_name(0)}, "
          f"compute capability: {torch.cuda.get_device_capability(0)}")
print(f"[test] torch version: {torch.__version__}")

try:
    import sageattention
    print(f"[test] sageattention module: {sageattention.__file__}")
    print(f"[test] sageattention version: {getattr(sageattention, '__version__', 'unknown')}")
except ImportError as exc:
    print(f"[test] SageAttention: NOT INSTALLED ({exc})")
    raise SystemExit(0)

try:
    from sageattention import sageattn
except ImportError as exc:
    print(f"[test] sageattention.sageattn: NOT FOUND in this build ({exc})")
    print(f"[test] module contents: {[n for n in dir(sageattention) if not n.startswith('_')]}")
    raise SystemExit(0)

if not torch.cuda.is_available():
    print("[test] No CUDA device -- can't test a real kernel call.")
    raise SystemExit(0)

device = torch.device("cuda")
batch, heads, seq_len, head_dim = 1, 32, 1024, 128
q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)

try:
    out = sageattn(q, k, v, is_causal=False)
    torch.cuda.synchronize()
    print(f"[test] SageAttention: SUCCEEDED, output shape {tuple(out.shape)}, dtype {out.dtype}")
    print("[test] Real kernel call worked -- worth wiring into "
          "ltx-core/.../attention.py as a real AttentionCallable and "
          "re-running the block-timing benchmark against the SDPA baseline.")
except Exception as exc:  # noqa: BLE001 - report whatever the real kernel call raises
    print(f"[test] SageAttention: FAILED on real kernel call: {type(exc).__name__}: {exc}")
    print("[test] If this mentions capability/architecture/'no kernel', it likely "
          "confirms no sm_100/103 kernel exists in this build's source -- not fixable "
          "by reinstalling, matches the known GitHub issue #237 gap.")
