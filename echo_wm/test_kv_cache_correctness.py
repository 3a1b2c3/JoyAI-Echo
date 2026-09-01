"""Correctness check: does the current real update_kv_cache in attention.py
(searchsorted-based general path + the newer BlockKVCache-style fixed-chunk
graph-capturable path, ported from flashdreams' core/attention/kvcache.py --
see TROUBLESHOOTING.md) produce byte-identical output to the original
boolean-mask version, across a realistic sequence of calls (cache fill-up,
then steady-state windowing, including repeated same-range "denoising step"
overwrites, and Echo-WM's irregular first-block-width-1-then-width-3
pattern)?

This doesn't need a GPU or the real model -- update_kv_cache only depends
on plain tensor ops. Run: python test_kv_cache_correctness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "ltx-core" / "src"))
from ltx_core.model.transformer.attention import update_kv_cache as update_kv_cache_real  # noqa: E402


def update_kv_cache_old(cache: dict, start: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Original boolean-mask-indexing version, for comparison."""
    length = int(cache["length"])
    old_positions = cache["positions"][:length]
    old_k = cache["k"][:, :length]
    old_v = cache["v"][:, :length]
    end = start + k.shape[1]

    keep_old = old_positions < start
    positions = torch.cat([old_positions[keep_old], torch.arange(start, end, device=k.device)], dim=0)
    merged_k = torch.cat([old_k[:, keep_old], k], dim=1)
    merged_v = torch.cat([old_v[:, keep_old], v], dim=1)

    local = int(cache.get("local_attn_size", -1))
    sink = int(cache.get("sink_tokens", 0))
    if local >= 0 and positions.numel() > local:
        if not 0 <= sink < local:
            raise ValueError(f"expected 0 <= sink_tokens < local_attn_size, got {sink}/{local}")
        sink_mask = positions < sink
        recent_budget = local - int(sink_mask.sum())
        recent_start = max(sink, end - recent_budget)
        keep = sink_mask | (positions >= recent_start)
        positions = positions[keep]
        merged_k = merged_k[:, keep]
        merged_v = merged_v[:, keep]

    active = positions.numel()
    cache["k"][:, :active].copy_(merged_k)
    cache["v"][:, :active].copy_(merged_v)
    cache["positions"][:active].copy_(positions)
    cache["length"] = active
    return cache["k"][:, :active].clone(), cache["v"][:, :active].clone()


def update_kv_cache_new(cache: dict, start: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """New searchsorted-based version -- copy of the real one in attention.py."""
    length = int(cache["length"])
    old_positions = cache["positions"][:length]
    old_k = cache["k"][:, :length]
    old_v = cache["v"][:, :length]
    end = start + k.shape[1]

    n_keep_old = int(torch.searchsorted(old_positions, start).item()) if length > 0 else 0
    positions = torch.cat([old_positions[:n_keep_old], torch.arange(start, end, device=k.device)], dim=0)
    merged_k = torch.cat([old_k[:, :n_keep_old], k], dim=1)
    merged_v = torch.cat([old_v[:, :n_keep_old], v], dim=1)

    local = int(cache.get("local_attn_size", -1))
    sink = int(cache.get("sink_tokens", 0))
    if local >= 0 and positions.numel() > local:
        if not 0 <= sink < local:
            raise ValueError(f"expected 0 <= sink_tokens < local_attn_size, got {sink}/{local}")
        recent_budget = local - sink
        recent_start = max(sink, end - recent_budget)
        recent_start_idx = int(torch.searchsorted(positions, recent_start).item())
        positions = torch.cat([positions[:sink], positions[recent_start_idx:]], dim=0)
        merged_k = torch.cat([merged_k[:, :sink], merged_k[:, recent_start_idx:]], dim=1)
        merged_v = torch.cat([merged_v[:, :sink], merged_v[:, recent_start_idx:]], dim=1)

    active = positions.numel()
    cache["k"][:, :active].copy_(merged_k)
    cache["v"][:, :active].copy_(merged_v)
    cache["positions"][:active].copy_(positions)
    cache["length"] = active
    return cache["k"][:, :active].clone(), cache["v"][:, :active].clone()


def make_cache(capacity: int, dim: int, local_attn_size: int, sink_tokens: int) -> dict:
    return {
        "k": torch.zeros(1, capacity, dim),
        "v": torch.zeros(1, capacity, dim),
        "positions": torch.zeros(capacity, dtype=torch.long),
        "length": 0,
        "local_attn_size": local_attn_size,
        "sink_tokens": sink_tokens,
    }


def _compare(cache_old: dict, cache_new: dict, cache_real: dict, label: str) -> bool:
    ok = True
    for name, other in (("new (searchsorted)", cache_new), ("real (attention.py, incl. fixed-chunk path)", cache_real)):
        k_match = torch.equal(cache_old["k"][:, : cache_old["length"]], other["k"][:, : other["length"]])
        v_match = torch.equal(cache_old["v"][:, : cache_old["length"]], other["v"][:, : other["length"]])
        pos_match = torch.equal(
            cache_old["positions"][: cache_old["length"]], other["positions"][: other["length"]]
        )
        len_match = cache_old["length"] == other["length"]
        if not (k_match and v_match and pos_match and len_match):
            print(f"[MISMATCH] {label} vs {name}: k={k_match} v={v_match} pos={pos_match} len={len_match}")
            print(f"  old positions:  {cache_old['positions'][:cache_old['length']].tolist()}")
            print(f"  {name} positions: {other['positions'][:other['length']].tolist()}")
            ok = False
    return ok


def run_scenario(
    local_attn_size: int, sink_tokens: int, chunk_size: int, n_blocks: int, redo_last: bool,
    capacity: int | None = None,
) -> bool:
    """Simulates n_blocks worth of causal generation: for each block,
    optionally "redo" it once (simulating a repeated denoising step
    overwriting the same [start, end) range before the final clean pass),
    then advance. Compares the original boolean-mask implementation against
    both the searchsorted rewrite and the real current implementation in
    attention.py (which includes the newer BlockKVCache-style fixed-chunk
    path). capacity=None defaults to exactly local_attn_size, matching real
    model.py::allocate() -- required for the fixed-chunk path to activate at
    all (it only engages when local_attn_size == cache capacity)."""
    if capacity is None:
        capacity = local_attn_size
    dim = 4
    torch.manual_seed(0)

    cache_old = make_cache(capacity, dim, local_attn_size, sink_tokens)
    cache_new = make_cache(capacity, dim, local_attn_size, sink_tokens)
    cache_real = make_cache(capacity, dim, local_attn_size, sink_tokens)

    start = 0
    all_match = True
    for block in range(n_blocks):
        end = start + chunk_size
        steps = [0, 1] if redo_last else [0]  # simulate a redone step, then commit
        for _step in steps:
            k = torch.randn(1, chunk_size, dim)
            v = torch.randn(1, chunk_size, dim)
            update_kv_cache_old(cache_old, start, k.clone(), v.clone())
            update_kv_cache_new(cache_new, start, k.clone(), v.clone())
            update_kv_cache_real(cache_real, start, k.clone(), v.clone())

            if not _compare(cache_old, cache_new, cache_real, f"local={local_attn_size} sink={sink_tokens} "
                             f"chunk={chunk_size} block={block} start={start} end={end}"):
                all_match = False
        start = end
    return all_match


def run_irregular_first_block_scenario(
    local_attn_size: int, sink_tokens: int, chunk_size: int, n_blocks: int
) -> bool:
    """Mirrors Echo-WM's real block layout (causal_video_blocks): a one-off
    width-1 first block, then every later block at the fixed chunk_size --
    exactly the pattern that makes update_kv_cache's fixed-chunk path only
    activate from the *second* call onward (first call always falls back to
    the general path, since its width never matches the steady chunk_size)."""
    capacity = local_attn_size
    dim = 4
    torch.manual_seed(1)

    cache_old = make_cache(capacity, dim, local_attn_size, sink_tokens)
    cache_new = make_cache(capacity, dim, local_attn_size, sink_tokens)
    cache_real = make_cache(capacity, dim, local_attn_size, sink_tokens)

    widths = [1] + [chunk_size] * n_blocks
    start = 0
    all_match = True
    for block, width in enumerate(widths):
        k = torch.randn(1, width, dim)
        v = torch.randn(1, width, dim)
        update_kv_cache_old(cache_old, start, k.clone(), v.clone())
        update_kv_cache_new(cache_new, start, k.clone(), v.clone())
        update_kv_cache_real(cache_real, start, k.clone(), v.clone())
        if not _compare(cache_old, cache_new, cache_real,
                         f"irregular-first local={local_attn_size} sink={sink_tokens} block={block} start={start}"):
            all_match = False
        start += width
    return all_match


def run_overlapping_starts_scenario() -> bool:
    """The exact call sequence from tests/test_echo_wm_causal.py::
    test_sink_plus_fifo_cache_rollover_and_block_replacement -- overlapping
    (non-chunk-aligned) starts (0, 2, 5, 8) with capacity=7, sink=2. Also
    asserts against that test's own hardcoded expected positions, not just
    old-vs-new agreement, since this is the one scenario with independent
    ground truth."""
    capacity, sink, dim = 7, 2, 1
    cache_old = make_cache(capacity, dim, capacity, sink)
    cache_new = make_cache(capacity, dim, capacity, sink)
    cache_real = make_cache(capacity, dim, capacity, sink)

    all_match = True
    for start in (0, 2, 5, 8):
        values = torch.arange(start, start + 3).view(1, 3, 1).float()
        update_kv_cache_old(cache_old, start, values.clone(), values.clone())
        update_kv_cache_new(cache_new, start, values.clone(), values.clone())
        update_kv_cache_real(cache_real, start, values.clone(), values.clone())
        if not _compare(cache_old, cache_new, cache_real, f"overlapping-starts start={start}"):
            all_match = False

    expected = [0, 1, 6, 7, 8, 9, 10]
    for name, cache in (("old", cache_old), ("new", cache_new), ("real", cache_real)):
        got = cache["positions"][: cache["length"]].tolist()
        if got != expected:
            print(f"[MISMATCH] overlapping-starts {name} final positions {got} != expected {expected}")
            all_match = False

    replacement = torch.full((1, 3, 1), 99.0)
    active_k_old, _ = update_kv_cache_old(cache_old, 8, replacement.clone(), replacement.clone())
    active_k_new, _ = update_kv_cache_new(cache_new, 8, replacement.clone(), replacement.clone())
    active_k_real, _ = update_kv_cache_real(cache_real, 8, replacement.clone(), replacement.clone())
    for name, active_k in (("old", active_k_old), ("new", active_k_new), ("real", active_k_real)):
        got = active_k[0, -3:, 0].tolist()
        if got != [99.0, 99.0, 99.0]:
            print(f"[MISMATCH] overlapping-starts {name} redo trailing values {got} != [99.0, 99.0, 99.0]")
            all_match = False
    return all_match


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    scenarios = [
        # (local_attn_size, sink_tokens, chunk_size, n_blocks, redo_last)
        (10, 4, 3, 20, False),
        (10, 4, 3, 20, True),   # with repeated/redone denoising steps per block
        (7, 1, 3, 20, False),
        (7, 1, 3, 20, True),
        (4, 1, 3, 20, False),   # the minimum valid window
        (4, 1, 3, 20, True),
        (19, 7, 3, 15, True),   # native/default window
    ]
    all_ok = True
    for local_attn_size, sink, chunk, n_blocks, redo in scenarios:
        ok = run_scenario(local_attn_size, sink, chunk, n_blocks, redo)
        status = "OK" if ok else "FAILED"
        print(f"[test] local_attn_size={local_attn_size} sink={sink} chunk={chunk} "
              f"n_blocks={n_blocks} redo_last={redo} (capacity==local, exercises fixed-chunk path): {status}")
        all_ok = all_ok and ok

    for local_attn_size, sink, chunk, n_blocks in [(10, 4, 3, 20), (7, 1, 3, 20), (19, 7, 3, 15)]:
        ok = run_irregular_first_block_scenario(local_attn_size, sink, chunk, n_blocks)
        status = "OK" if ok else "FAILED"
        print(f"[test] irregular-first-block local={local_attn_size} sink={sink} chunk={chunk}: {status}")
        all_ok = all_ok and ok

    ok = run_overlapping_starts_scenario()
    status = "OK" if ok else "FAILED"
    print(f"[test] overlapping-starts (tests/test_echo_wm_causal.py scenario, incl. hardcoded ground truth): {status}")
    all_ok = all_ok and ok

    print()
    if all_ok:
        print("[test] ALL SCENARIOS MATCHED -- both the searchsorted rewrite and the real "
              "attention.py implementation (incl. the new BlockKVCache-style fixed-chunk "
              "path) are byte-identical to the original boolean-mask version across all "
              "tested configurations.")
    else:
        print("[test] MISMATCH FOUND -- do NOT trust the current update_kv_cache yet, "
              "see failures above.")
