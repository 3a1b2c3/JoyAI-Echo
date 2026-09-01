"""Does capturing the cross-modal RoPE branch of Attention.forward() at one
block's kv_cache_start, then replaying with a *different* block's data,
silently apply the WRONG RoPE slice?

Concern (traced, not yet proven): in ltx_core/model/transformer/attention.py,
the a2v/v2a cross-modal branch does

    query_slice = kv_cache["local_cross_q_slices"].get((kv_cache_start, kv_cache_start + new_keys))
    q = apply_rotary_emb(q, _slice_rope(local_q_pe, *query_slice), self.rope_type)

`kv_cache_start` is a plain Python int, different every block. Under
torch.cuda.graph() capture, this dict lookup runs once, at capture time --
its result (query_slice) gets baked into the recorded op sequence. If
replaying the captured graph for a *different* block (different
kv_cache_start, different real data) still uses the *captured* block's
query_slice, RoPE would be silently wrong for every block except the one
captured -- a correctness bug, not a crash.

This script isolates exactly that branch (importing the real
apply_rotary_emb/_slice_rope/update_kv_cache from attention.py, not a
reimplementation) with tiny synthetic tensors, and empirically checks:
does graph-replay-with-swapped-data match a fresh eager run for the SAME
new block, or does it silently reuse the captured block's RoPE slice?

Needs a real CUDA GPU (torch.cuda.graph() capture doesn't work on CPU) --
run on the GB300 box, not locally.

Run: python test_graph_capture_cross_modal_rope.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "ltx-core" / "src"))
from ltx_core.model.transformer.attention import _slice_rope, update_kv_cache  # noqa: E402
from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb  # noqa: E402

if not torch.cuda.is_available():
    print("[test] No CUDA device -- this test needs a real GPU (torch.cuda.graph() "
          "capture isn't supported on CPU). Run on the GB300 box.")
    raise SystemExit(0)

torch.set_grad_enabled(False)
device = torch.device("cuda")
DIM = 8  # small, even (rope needs even dim for interleaved pairs)
HEADS_DIM = DIM  # single "head" worth of channels for this isolated test


def make_cache(capacity: int, local_q_pe, local_k_pe, q_slices: dict) -> dict:
    return {
        "k": torch.zeros(1, capacity, DIM, device=device),
        "v": torch.zeros(1, capacity, DIM, device=device),
        "positions": torch.full((capacity,), -1, device=device, dtype=torch.long),
        "length": 0,
        "local_attn_size": capacity,
        "sink_tokens": 0,
        "local_cross_q_rope_pe": local_q_pe,
        "local_cross_k_rope_pe": local_k_pe,
        "local_cross_q_slices": q_slices,
    }


def cross_modal_step(
    kv_cache: dict, kv_cache_start: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Exact copy of Attention.forward()'s
    `elif local_q_pe is not None or local_k_pe is not None:` branch
    (attention.py) -- the piece under test. Returns q after RoPE, which is
    what would flow into self.attention_function(q, k, v, ...) next; k/v
    also get RoPE'd for completeness but q is what we diff."""
    local_q_pe = kv_cache["local_cross_q_rope_pe"]
    local_k_pe = kv_cache["local_cross_k_rope_pe"]
    new_keys = k.shape[1]
    k, v = update_kv_cache(kv_cache, kv_cache_start, k, v)
    query_slice = kv_cache["local_cross_q_slices"].get((kv_cache_start, kv_cache_start + new_keys))
    if query_slice is None:
        raise ValueError(f"missing local cross-modal query RoPE slice for {(kv_cache_start, kv_cache_start + new_keys)}")
    q = apply_rotary_emb(q, _slice_rope(local_q_pe, *query_slice), LTXRopeType.INTERLEAVED)
    return q


# Two distinct "blocks": different absolute start, different real q/k/v data.
CHUNK = 4
BLOCK_A_START, BLOCK_B_START = 0, 8
CAPACITY = 16

# RoPE tables covering enough absolute range for both blocks' query slices.
# Shape (B, T, D) cos/sin, broadcastable against q/k of shape (1, T, D).
max_t = 32
angles = torch.linspace(0, 3.14, max_t, device=device).unsqueeze(0).unsqueeze(-1).expand(1, max_t, DIM)
local_q_pe = (torch.cos(angles), torch.sin(angles))
local_k_pe = (torch.cos(angles), torch.sin(angles))

q_slices = {
    (BLOCK_A_START, BLOCK_A_START + CHUNK): (BLOCK_A_START, BLOCK_A_START + CHUNK),
    (BLOCK_B_START, BLOCK_B_START + CHUNK): (BLOCK_B_START, BLOCK_B_START + CHUNK),
}

torch.manual_seed(0)
q_a = torch.randn(1, CHUNK, DIM, device=device)
k_a = torch.randn(1, CHUNK, DIM, device=device)
v_a = torch.randn(1, CHUNK, DIM, device=device)
torch.manual_seed(1)
q_b = torch.randn(1, CHUNK, DIM, device=device)
k_b = torch.randn(1, CHUNK, DIM, device=device)
v_b = torch.randn(1, CHUNK, DIM, device=device)

# --- Reference 1: eager run for block A ---
cache_a_eager = make_cache(CAPACITY, local_q_pe, local_k_pe, q_slices)
q_a_out_eager = cross_modal_step(cache_a_eager, BLOCK_A_START, q_a.clone(), k_a.clone(), v_a.clone())

# --- Reference 2: eager run for block B (the ground truth we want replay to match) ---
cache_b_eager = make_cache(CAPACITY, local_q_pe, local_k_pe, q_slices)
q_b_out_eager = cross_modal_step(cache_b_eager, BLOCK_B_START, q_b.clone(), k_b.clone(), v_b.clone())

# --- Graph capture at block A's kv_cache_start, using static buffers ---
cache_capture = make_cache(CAPACITY, local_q_pe, local_k_pe, q_slices)
static_q = q_a.clone()
static_k = k_a.clone()
static_v = v_a.clone()

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        cross_modal_step(cache_capture, BLOCK_A_START, static_q, static_k, static_v)
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_q_out = cross_modal_step(cache_capture, BLOCK_A_START, static_q, static_k, static_v)

print("[test] Capture at block A succeeded. Replaying with block B's data "
      "swapped into the static buffers (kv_cache_start still baked as A)...")

# --- Replay with block B's DATA swapped in, but kv_cache_start is baked as A ---
static_q.copy_(q_b)
static_k.copy_(k_b)
static_v.copy_(v_b)
g.replay()
torch.cuda.synchronize()
q_replay_out = static_q_out.clone()

match = torch.allclose(q_replay_out, q_b_out_eager, atol=1e-5, rtol=1e-5)
print(f"[test] replay-with-swapped-data vs block-B-eager: {'MATCH' if match else 'MISMATCH'}")
if match:
    print("[test] UNEXPECTED (given the traced concern) -- graph replay with "
          "swapped data matched the correct block-B answer. Either the concern "
          "doesn't apply the way it was reasoned, or this particular RoPE table "
          "happens to be symmetric in a way that hides it -- investigate further "
          "before trusting this as a general result.")
else:
    print("[test] CONFIRMED: graph-capturing this branch at one block's "
          "kv_cache_start and replaying with another block's data silently "
          "applies the WRONG RoPE slice. Real integration must NOT capture "
          "this branch as-is -- query_slice needs to be computed from a tensor "
          "value (not a baked Python dict lookup) before this is safe to graph.")
    print(f"  replay q[0,0,:4]     = {q_replay_out[0, 0, :4].tolist()}")
    print(f"  block-B eager q[0,0,:4] = {q_b_out_eager[0, 0, :4].tolist()}")
