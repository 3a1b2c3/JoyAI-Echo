"""Standalone check: does FlashInfer accept our exact attention pattern
(real additive bias tensor, not just a boolean causal mask) at all?

This model's attention.py passes a real bias tensor into every single
attention call (see XFormersAttention/PytorchAttention's `mask` handling in
ltx_core/model/transformer/attention.py) -- every other fast-attention
library tried this session (xformers, FlashAttention-3/4) either has no
kernel for this GPU's compute capability, or flatly refuses a bias tensor
at all. This script isolates the one open question before writing any real
integration code: does flashinfer.single_prefill_with_kv_cache (or whatever
the actual matching API turns out to be) accept a real additive bias tensor
and produce numerically correct output vs. a plain PyTorch SDPA reference,
on THIS box's actual GPU.

Not a benchmark -- purely a correctness/API-shape smoke test. If this
fails, there's no point writing a real FlashInferAttention class for
attention.py; if it passes, that's the green light to do the real
integration.

Run on the GB300 box (inside echo_wm/.venv):
    pip install flashinfer-python
    python test_flashinfer_bias.py
"""

from __future__ import annotations

import torch

print(f"[test] CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[test] GPU: {torch.cuda.get_device_name(0)}, "
          f"compute capability: {torch.cuda.get_device_capability(0)}", flush=True)

try:
    import flashinfer
    print(f"[test] flashinfer imported OK, version: {getattr(flashinfer, '__version__', '?')}", flush=True)
except ImportError as exc:
    raise SystemExit(f"[test] flashinfer not installed: {exc}\nRun: pip install flashinfer-python") from exc

# Small, representative shapes -- matches the query/key/value shapes seen in
# this session's xformers rejection log: (1, 1024, 32, 128) -- [batch,
# tokens, heads, dim_head]. Real additive bias tensor (not boolean), which
# is exactly the pattern every other library has choked on.
batch, tokens, heads, dim_head = 1, 1024, 32, 128
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

torch.manual_seed(0)
q = torch.randn(batch, tokens, heads, dim_head, device=device, dtype=dtype)
k = torch.randn(batch, tokens, heads, dim_head, device=device, dtype=dtype)
v = torch.randn(batch, tokens, heads, dim_head, device=device, dtype=dtype)
# A real additive bias tensor -- e.g. representing a local-attention-window
# penalty (large negative outside the window), not just True/False.
bias = torch.zeros(batch, heads, tokens, tokens, device=device, dtype=dtype)
window = 128
idx = torch.arange(tokens, device=device)
outside_window = (idx[None, :] - idx[:, None]).abs() > window
bias[:, :, outside_window] = float("-1e4")

print(f"[test] q/k/v shape: {q.shape}, bias shape: {bias.shape}", flush=True)

# --- Reference: plain PyTorch SDPA (same math this codebase's PytorchAttention uses) ---
q_sdpa = q.transpose(1, 2)  # [B, H, T, D]
k_sdpa = k.transpose(1, 2)
v_sdpa = v.transpose(1, 2)
ref = torch.nn.functional.scaled_dot_product_attention(
    q_sdpa, k_sdpa, v_sdpa, attn_mask=bias, dropout_p=0.0, is_causal=False
)
print(f"[test] SDPA reference output shape: {ref.shape}, "
      f"mean={ref.float().mean().item():.4f}, std={ref.float().std().item():.4f}", flush=True)

# --- FlashInfer attempt: try the most likely-matching API for this pattern ---
# flashinfer's API surface varies by version; this tries the single-request
# prefill path with a custom additive bias, per the LogitsTransform/
# custom_mask support found in the docs. If this specific call signature is
# wrong for the installed version, the error message itself is useful
# (tells us the real API to target), not just a pass/fail.
try:
    from flashinfer import single_prefill_with_kv_cache

    q_fi = q[0]  # flashinfer's single-request API typically drops the batch dim: [T, H, D]
    k_fi = k[0]
    v_fi = v[0]
    bias_fi = bias[0]  # [H, T, T]

    out = single_prefill_with_kv_cache(
        q_fi, k_fi, v_fi,
        custom_mask=None,
        # Different flashinfer versions expose this differently -- logits_soft_cap,
        # a custom LogitsTransform callback, or a direct additive-bias kwarg. Try the
        # most literal name first; the resulting error (if any) tells us the real one.
        # This is intentionally a first attempt, not a confirmed-correct call.
    )
    print(f"[test] FlashInfer call succeeded (WITHOUT bias -- API smoke test only), "
          f"output shape: {out.shape}", flush=True)
    print("[test] NOTE: this call did NOT pass the bias tensor -- the API for custom "
          "additive bias needs to be found from flashinfer's actual docs/source for "
          "the installed version before a real correctness comparison is possible.",
          flush=True)
except ImportError as exc:
    print(f"[test] single_prefill_with_kv_cache not found in this flashinfer version: {exc}", flush=True)
    print("[test] Check `python -c \"import flashinfer; help(flashinfer)\"` "
          "for the actual API surface of the installed version.", flush=True)
except Exception as exc:  # noqa: BLE001 - we want to see exactly what breaks
    print(f"[test] FlashInfer call failed: {type(exc).__name__}: {exc}", flush=True)

print("[test] Done. Key question still open: does the installed flashinfer version's "
      "actual API accept a real additive bias tensor (not just a boolean mask)? "
      "Check the printed API surface / error above and flashinfer's docs for this "
      "specific version.", flush=True)
