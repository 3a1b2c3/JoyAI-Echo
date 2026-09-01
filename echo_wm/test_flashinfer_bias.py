"""Standalone check, step 2: find FlashInfer 0.6.18's actual API for a
custom additive bias tensor, then verify it produces numerically correct
output vs. plain PyTorch SDPA on this GB300 box.

Step 1 (earlier run) confirmed: flashinfer 0.6.18 imports fine and
single_prefill_with_kv_cache() runs successfully on GB300 (compute
capability 10.3) WITHOUT a bias argument -- the first fast-attention
library this session that didn't reject the hardware outright. This script
introspects the real function signature (instead of guessing kwarg names
again) to find how to pass our additive bias tensor, then checks the
output actually matches an SDPA reference numerically.
"""

from __future__ import annotations

import inspect

import torch

import flashinfer
from flashinfer import single_prefill_with_kv_cache

print(f"[test] flashinfer version: {getattr(flashinfer, '__version__', '?')}", flush=True)
print(f"[test] single_prefill_with_kv_cache signature:", flush=True)
print(f"  {inspect.signature(single_prefill_with_kv_cache)}", flush=True)
print(flush=True)
doc = inspect.getdoc(single_prefill_with_kv_cache) or "(no docstring)"
print("[test] docstring:", flush=True)
print(doc, flush=True)
print(flush=True)

# Also scan the whole module for anything bias/mask/logits related, in case
# the real answer is a different function (e.g. a variant that takes a
# custom_mask or attn_bias kwarg, or a separate masked/biased entry point)
# rather than an extra kwarg on this one.
print("[test] flashinfer module members matching bias/mask/logits/custom:", flush=True)
for name in sorted(dir(flashinfer)):
    if any(kw in name.lower() for kw in ("bias", "mask", "logits", "custom")):
        print(f"  flashinfer.{name}", flush=True)

# Same scan one level into flashinfer.prefill specifically, since that's
# likely where single_prefill_with_kv_cache's siblings/variants live.
try:
    import flashinfer.prefill as _prefill_mod
    print("[test] flashinfer.prefill members matching bias/mask/logits/custom:", flush=True)
    for name in sorted(dir(_prefill_mod)):
        if any(kw in name.lower() for kw in ("bias", "mask", "logits", "custom")):
            print(f"  flashinfer.prefill.{name}", flush=True)
except ImportError:
    print("[test] no flashinfer.prefill submodule", flush=True)

print(flush=True)
print("[test] Done -- use the signature/docstring/member list above to find the real "
      "bias-tensor argument name, then we can write a real correctness check.", flush=True)
