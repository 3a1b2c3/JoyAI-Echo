"""Which attention backend actually succeeds for a representative shape,
on whatever torch/CUDA/library combination is currently installed.

Doesn't touch the real model -- just calls each AttentionCallable directly
with dummy tensors matching the shape seen in this model's real attention
calls (q_shape=(1, 1024, 4096), 32 heads, 128 dim_head), with mask=None
(this model's masks are always either literally None or an all-zero no-op
-- see TROUBLESHOOTING.md item -14 -- so None is the representative case).

Run: python test_attention_backend.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "ltx-core/src")
from ltx_core.model.transformer import attention as A  # noqa: E402

print(f"[test] CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[test] GPU: {torch.cuda.get_device_name(0)}, "
          f"compute capability: {torch.cuda.get_device_capability(0)}", flush=True)
print(f"[test] torch version: {torch.__version__}", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
q = torch.randn(1, 1024, 4096, device=device, dtype=torch.bfloat16)
k = torch.randn(1, 1024, 4096, device=device, dtype=torch.bfloat16)
v = torch.randn(1, 1024, 4096, device=device, dtype=torch.bfloat16)
heads = 32

backends = [
    ("xformers", A.XFormersAttention, A.memory_efficient_attention is not None),
    ("FlashAttention3", A.FlashAttention3, A.flash_attn_interface is not None),
    ("FlashAttention2", A.FlashAttention2, A.flash_attn_func is not None),
    ("FlashInfer", A.FlashInferAttention, A.flashinfer_single_prefill is not None),
    ("PyTorch SDPA", A.PytorchAttention, True),
]

for name, cls, avail in backends:
    if not avail:
        print(f"[test] {name}: NOT INSTALLED", flush=True)
        continue
    try:
        out = cls()(q, k, v, heads, None)
        print(f"[test] {name}: SUCCEEDED, output shape {tuple(out.shape)}", flush=True)
    except Exception as exc:  # noqa: BLE001 - report whatever breaks, don't hide it
        print(f"[test] {name}: FAILED - {type(exc).__name__}: {exc}", flush=True)

print("\n[test] Also checking what AttentionFunction.DEFAULT actually picks:", flush=True)
out = A.AttentionFunction.DEFAULT(q, k, v, heads, None)
print(f"[test] AttentionFunction.DEFAULT returned shape {tuple(out.shape)} "
      f"(check stdout above/before this for which backend's print, if any, fired)", flush=True)
