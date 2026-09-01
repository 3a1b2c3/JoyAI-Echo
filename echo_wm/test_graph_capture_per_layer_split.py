"""Feasibility test for TROUBLESHOOTING.md item -1.0's per-layer-split fix
for item -0.7's graph-capture blocker.

Background: item -0.7 proved that graph-capturing the a2v/v2a cross-modal
branch is unsafe, because its `query_slice` comes from a Python dict lookup
keyed on `kv_cache_start` (a plain int that's different every block) --
capture bakes in whichever slice was live at capture time.

Item -1.0 confirmed video_self/audio_self does NOT have this problem: its
slice bounds come from `update_kv_cache`'s returned `active` count, which is
`k.shape[1]`-derived (a static shape, not a captured Python-side value) and,
crucially, is CONSTANT across steady-state blocks once the sink+window cache
is full (`_update_kv_cache_fixed_chunk` always returns `active == capacity`
once `local == capacity`, see attention.py update_kv_cache). So self-attn's
`_slice_rope(local_pe, active - q_len, active)` bounds don't change block to
block in steady state, unlike a2v/v2a's per-block-varying query_slice.

That means the only architecturally valid version of fix (b) is a
PER-LAYER SPLIT: capture self-attn+cache-update, drop to eager Python for
that layer's a2v/v2a, re-enter capture for the next layer's self-attn --
repeated once per layer. This has never been tried in this codebase. The
existing ECHO_WM_GRAPH_CAPTURE_TEST (rollout.py) only tests ONE capture
region around the *whole* forward() call, which is exactly what's broken.

Open question this script actually tests: does interleaving TWO separate
torch.cuda.graph() capture regions with real eager Python work executed
between them -- on the same cache dict / device state -- produce correct
results across two different real blocks (different kv_cache_start each),
or does something about re-entering capture, memory pool reuse, or stream
state between the two regions silently break either region's correctness?
This is a different, previously-untested risk from item -0.7's single-
region bug.

Simulates a 2-"layer" stack (self-attn capture -> eager cross-modal ->
self-attn capture, twice) processing two different blocks (kv_cache_start
0 and 4, CHUNK=4, steady-state fixed-chunk cache), using the REAL
update_kv_cache/apply_rotary_emb/_slice_rope functions from attention.py
(not reimplemented), and diffs the split-capture path's output against a
fully eager reference for the same two blocks.

Needs a real CUDA GPU (torch.cuda.graph() capture doesn't work on CPU) --
run on the GB300 box, not locally.

Run: python test_graph_capture_per_layer_split.py
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
CHUNK = 4
CAPACITY = 8  # local_attn_size == capacity, so fixed-chunk path activates
              # once the cache has seen one advance (is_advance requires a
              # prior call, so block 0 uses the general path, block 1+ uses
              # fixed-chunk -- matches real steady-state behavior).


def make_self_attn_cache(capacity: int, local_pe) -> dict:
    return {
        "k": torch.zeros(1, capacity, DIM, device=device),
        "v": torch.zeros(1, capacity, DIM, device=device),
        "positions": torch.full((capacity,), -1, device=device, dtype=torch.long),
        "length": 0,
        "local_attn_size": capacity,
        "sink_tokens": 0,
        "local_rope_pe": local_pe,
    }


def make_cross_modal_cache(capacity: int, local_q_pe, local_k_pe, q_slices: dict) -> dict:
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


def self_attn_step(
    cache: dict, start: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Exact copy of Attention.forward()'s `if local_pe is not None:` branch
    (video_self/audio_self) -- the piece this test claims IS graph-safe."""
    local_pe = cache["local_rope_pe"]
    k, v = update_kv_cache(cache, start, k, v)
    active = k.shape[1]
    q_len = q.shape[1]
    q = apply_rotary_emb(q, _slice_rope(local_pe, active - q_len, active), LTXRopeType.INTERLEAVED)
    return q


def cross_modal_step(
    cache: dict, start: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Exact copy of Attention.forward()'s a2v/v2a branch -- run EAGERLY
    (never captured) in the split-capture design, so this is just the
    known-correct real behavior, not what's under test."""
    local_q_pe = cache["local_cross_q_rope_pe"]
    local_k_pe = cache["local_cross_k_rope_pe"]
    new_keys = k.shape[1]
    k, v = update_kv_cache(cache, start, k, v)
    query_slice = cache["local_cross_q_slices"].get((start, start + new_keys))
    if query_slice is None:
        raise ValueError(f"missing query RoPE slice for {(start, start + new_keys)}")
    q = apply_rotary_emb(q, _slice_rope(local_q_pe, *query_slice), LTXRopeType.INTERLEAVED)
    return q


max_t = 32
angles = torch.linspace(0, 3.14, max_t, device=device).unsqueeze(0).unsqueeze(-1).expand(1, max_t, DIM)
local_pe = (torch.cos(angles), torch.sin(angles))
local_q_pe = (torch.cos(angles), torch.sin(angles))
local_k_pe = (torch.cos(angles), torch.sin(angles))

BLOCK_0_START, BLOCK_1_START = 0, 4
q_slices = {
    (BLOCK_0_START, BLOCK_0_START + CHUNK): (BLOCK_0_START, BLOCK_0_START + CHUNK),
    (BLOCK_1_START, BLOCK_1_START + CHUNK): (BLOCK_1_START, BLOCK_1_START + CHUNK),
}

torch.manual_seed(0)
q0_l1, k0_l1, v0_l1 = (torch.randn(1, CHUNK, DIM, device=device) for _ in range(3))
q0_cm, k0_cm, v0_cm = (torch.randn(1, CHUNK, DIM, device=device) for _ in range(3))
q0_l2, k0_l2, v0_l2 = (torch.randn(1, CHUNK, DIM, device=device) for _ in range(3))
torch.manual_seed(1)
q1_l1, k1_l1, v1_l1 = (torch.randn(1, CHUNK, DIM, device=device) for _ in range(3))
q1_cm, k1_cm, v1_cm = (torch.randn(1, CHUNK, DIM, device=device) for _ in range(3))
q1_l2, k1_l2, v1_l2 = (torch.randn(1, CHUNK, DIM, device=device) for _ in range(3))


def run_two_layers_eager(block_start, q_l1, k_l1, v_l1, q_cm, k_cm, v_cm, q_l2, k_l2, v_l2):
    cache_l1 = make_self_attn_cache(CAPACITY, local_pe)
    cache_cm = make_cross_modal_cache(CAPACITY, local_q_pe, local_k_pe, q_slices)
    cache_l2 = make_self_attn_cache(CAPACITY, local_pe)
    out_l1 = self_attn_step(cache_l1, block_start, q_l1, k_l1, v_l1)
    out_cm = cross_modal_step(cache_cm, block_start, q_cm, k_cm, v_cm)
    out_l2 = self_attn_step(cache_l2, block_start, q_l2, k_l2, v_l2)
    return out_l1, out_cm, out_l2


# --- Reference: fully eager, block 0 then block 1 (fresh caches each --
# isolates per-block correctness of each piece independently, matching how
# this test's split-capture path also uses fresh per-block-start caches
# rather than modeling full multi-block cache accumulation, which is a
# simplification worth flagging if this test is extended further). ---
ref0 = run_two_layers_eager(BLOCK_0_START, q0_l1, k0_l1, v0_l1, q0_cm, k0_cm, v0_cm, q0_l2, k0_l2, v0_l2)
ref1 = run_two_layers_eager(BLOCK_1_START, q1_l1, k1_l1, v1_l1, q1_cm, k1_cm, v1_cm, q1_l2, k1_l2, v1_l2)

# --- Split-capture path: two independent CUDAGraph regions (layer 1 self-attn,
# layer 2 self-attn), eager cross-modal code actually running BETWEEN them,
# for real -- not simulated. Captured once (at block 0), replayed for block 1
# with block 1's real data swapped into the static buffers. ---
cache_l1_cap = make_self_attn_cache(CAPACITY, local_pe)
cache_l2_cap = make_self_attn_cache(CAPACITY, local_pe)
cache_cm_cap = make_cross_modal_cache(CAPACITY, local_q_pe, local_k_pe, q_slices)

static_q_l1, static_k_l1, static_v_l1 = q0_l1.clone(), k0_l1.clone(), v0_l1.clone()
static_q_l2, static_k_l2, static_v_l2 = q0_l2.clone(), k0_l2.clone(), v0_l2.clone()

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        self_attn_step(cache_l1_cap, BLOCK_0_START, static_q_l1, static_k_l1, static_v_l1)
torch.cuda.current_stream().wait_stream(s)

g1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1):
    static_out_l1 = self_attn_step(cache_l1_cap, BLOCK_0_START, static_q_l1, static_k_l1, static_v_l1)

print("[test] Layer-1 self-attn capture (region A) succeeded.")

s2 = torch.cuda.Stream()
s2.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s2):
    for _ in range(3):
        self_attn_step(cache_l2_cap, BLOCK_0_START, static_q_l2, static_k_l2, static_v_l2)
torch.cuda.current_stream().wait_stream(s2)

g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2):
    static_out_l2 = self_attn_step(cache_l2_cap, BLOCK_0_START, static_q_l2, static_k_l2, static_v_l2)

print("[test] Layer-2 self-attn capture (region B, captured AFTER region A "
      "with eager work already run on cache_l1_cap in between) succeeded.")

print("[test] Replaying region A -> real eager cross-modal -> region B for "
      "block 1, checking against block-1 eager reference...")

# Block 1: swap real data into region A's static buffers, replay, run real
# eager cross-modal, swap data into region B's static buffers, replay.
static_q_l1.copy_(q1_l1)
static_k_l1.copy_(k1_l1)
static_v_l1.copy_(v1_l1)
g1.replay()
torch.cuda.synchronize()
out_l1_split = static_out_l1.clone()

out_cm_split = cross_modal_step(cache_cm_cap, BLOCK_1_START, q1_cm.clone(), k1_cm.clone(), v1_cm.clone())

static_q_l2.copy_(q1_l2)
static_k_l2.copy_(k1_l2)
static_v_l2.copy_(v1_l2)
g2.replay()
torch.cuda.synchronize()
out_l2_split = static_out_l2.clone()

ref_l1, ref_cm, ref_l2 = ref1
match_l1 = torch.allclose(out_l1_split, ref_l1, atol=1e-5, rtol=1e-5)
match_cm = torch.allclose(out_cm_split, ref_cm, atol=1e-5, rtol=1e-5)
match_l2 = torch.allclose(out_l2_split, ref_l2, atol=1e-5, rtol=1e-5)

print(f"[test] layer-1 (captured, region A) block-1 replay vs eager: {'MATCH' if match_l1 else 'MISMATCH'}")
print(f"[test] cross-modal (eager, between regions) block-1 vs eager:  {'MATCH' if match_cm else 'MISMATCH'}")
print(f"[test] layer-2 (captured, region B) block-1 replay vs eager: {'MATCH' if match_l2 else 'MISMATCH'}")

if match_l1 and match_cm and match_l2:
    print("\n[test] CONFIRMED FEASIBLE: two independent torch.cuda.graph() regions "
          "with real eager Python work interleaved between them, replayed for a "
          "different block than they were captured at, both produced correct "
          "results. The per-layer-split version of fix (b) is not blocked by "
          "this specific risk -- real integration into transformer.py's layer "
          "loop is the next step, not further isolated feasibility testing.")
else:
    print("\n[test] BLOCKED: at least one region's replay did not match the eager "
          "reference. Something about re-entering capture / interleaved eager "
          "work / memory pool reuse between two graph regions breaks correctness "
          "here -- needs root-causing before attempting real integration.")
    if not match_l1:
        print(f"  region A: replay={out_l1_split[0,0,:4].tolist()} eager={ref_l1[0,0,:4].tolist()}")
    if not match_cm:
        print(f"  cross-modal: eager-between={out_cm_split[0,0,:4].tolist()} eager-ref={ref_cm[0,0,:4].tolist()}")
    if not match_l2:
        print(f"  region B: replay={out_l2_split[0,0,:4].tolist()} eager={ref_l2[0,0,:4].tolist()}")
