"""Correctness check: does the searchsorted-based update_kv_cache (rewritten
to avoid nonzero()-triggering boolean-mask indexing -- see
TROUBLESHOOTING.md) produce byte-identical output to the original
boolean-mask version, across a realistic sequence of calls (cache fill-up,
then steady-state windowing, including repeated same-range "denoising step"
overwrites)?

This doesn't need a GPU or the real model -- update_kv_cache only depends
on plain tensor ops. Run: python test_kv_cache_correctness.py
"""

from __future__ import annotations

import torch


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


def run_scenario(local_attn_size: int, sink_tokens: int, chunk_size: int, n_blocks: int, redo_last: bool) -> bool:
    """Simulates n_blocks worth of causal generation: for each block,
    optionally "redo" it once (simulating a repeated denoising step
    overwriting the same [start, end) range before the final clean pass),
    then advance. Returns True if old and new implementations matched at
    every single step."""
    capacity = local_attn_size + chunk_size + 8  # generous headroom
    dim = 4
    torch.manual_seed(0)

    cache_old = make_cache(capacity, dim, local_attn_size, sink_tokens)
    cache_new = make_cache(capacity, dim, local_attn_size, sink_tokens)

    start = 0
    all_match = True
    for block in range(n_blocks):
        end = start + chunk_size
        steps = [0, 1] if redo_last else [0]  # simulate a redone step, then commit
        for _step in steps:
            k = torch.randn(1, chunk_size, dim)
            v = torch.randn(1, chunk_size, dim)
            out_k_old, out_v_old = update_kv_cache_old(cache_old, start, k.clone(), v.clone())
            out_k_new, out_v_new = update_kv_cache_new(cache_new, start, k.clone(), v.clone())

            k_match = torch.equal(out_k_old, out_k_new)
            v_match = torch.equal(out_v_old, out_v_new)
            pos_match = torch.equal(
                cache_old["positions"][: cache_old["length"]], cache_new["positions"][: cache_new["length"]]
            )
            len_match = cache_old["length"] == cache_new["length"]

            if not (k_match and v_match and pos_match and len_match):
                print(f"[MISMATCH] local_attn_size={local_attn_size} sink={sink_tokens} "
                      f"chunk={chunk_size} block={block} start={start} end={end}: "
                      f"k_match={k_match} v_match={v_match} pos_match={pos_match} len_match={len_match}")
                print(f"  old positions: {cache_old['positions'][:cache_old['length']].tolist()}")
                print(f"  new positions: {cache_new['positions'][:cache_new['length']].tolist()}")
                all_match = False
        start = end
    return all_match


if __name__ == "__main__":
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
              f"n_blocks={n_blocks} redo_last={redo}: {status}")
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("[test] ALL SCENARIOS MATCHED -- searchsorted rewrite is byte-identical "
              "to the original boolean-mask version across all tested configurations.")
    else:
        print("[test] MISMATCH FOUND -- do NOT trust the searchsorted rewrite yet, "
              "see failures above.")
