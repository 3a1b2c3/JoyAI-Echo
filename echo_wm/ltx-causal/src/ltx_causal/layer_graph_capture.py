"""Per-layer CUDA graph capture wrapper for TransformerLayer's split phases
(see TROUBLESHOOTING.md item -1.0).

WIRED into LTXModel._process_transformer_blocks (model.py) and gated behind
ECHO_WM_GRAPH_CAPTURE_LAYERS=1 (CausalTI2VidPipeline, causal_ti2vid.py),
default off -- but NOT YET TESTED END-TO-END on real hardware as of this
writing. Only the isolated 2-toy-layer feasibility test
(test_graph_capture_per_layer_split.py) has actually been run on a GPU; the
real model.py integration has not. Treat ECHO_WM_GRAPH_CAPTURE_LAYERS=1 as
unverified and watch closely for wrong/corrupted generation output, not
just crashes, before trusting it -- this is exactly the failure mode item
-0.7 proved possible (silently wrong, not a crash) for a closely related
piece of code.

Design: only `vx`/`ax` (the residual-stream tensors) are treated as
per-call-varying static buffers. Every other argument to a captured phase
(video/audio's other TransformerArgs fields, kv_cache dict, perturbations,
kv_cache_start ints, ucpe tensors) is assumed FIXED across all replays of
one captured graph -- true within one generation call at one (layer_index,
shape) key, since positional embeddings/context/masks are computed once per
call and kv_cache_start only varies in the sense that a NEW graph must be
captured whenever it changes (this class does not attempt the item -0.7
fix of making kv_cache_start itself graph-traceable -- that's still the
cross-modal phase's job to stay eager for).

If any of those "fixed" arguments actually change between two calls at the
same (layer_index, shape) key -- e.g. a different prompt without a fresh
CausalTI2VidPipeline instance -- this class has NO way to detect that and
will silently replay stale-argument results. Whatever wires this in must
guarantee those arguments are stable for the lifetime of one captured
region, the same way CausalTI2VidPipeline's per-resolution _model_cache
(item -6) already has to guarantee shape stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class _CapturedRegion:
    graph: torch.cuda.CUDAGraph
    static_vx_in: torch.Tensor | None
    static_ax_in: torch.Tensor | None
    static_vx_out: torch.Tensor | None
    static_ax_out: torch.Tensor | None


class LayerPhaseGraphCapture:
    """Captures one (layer_index, vx/ax shape) combination's phase function
    once, replays it on every subsequent call with matching shapes.

    Usage (once transformer.py's phase split is confirmed correct):

        capture = LayerPhaseGraphCapture()
        vx, ax = capture.run(
            layer_index, layer._forward_self_attn_phase_tensors_only,
            vx, ax, warmup=3,
        )

    `phase_fn` must be a callable taking exactly (vx, ax) and returning
    exactly (vx, ax) -- i.e. every other argument (video/audio metadata,
    kv_cache, perturbations, ...) must already be bound via closure/partial
    before being passed here, since this class only manages vx/ax as
    graph-static buffers. This deliberately does NOT attempt to make
    kv_cache_start or any other Python-side value graph-traceable (see
    module docstring and item -0.7) -- callers needing per-block-varying
    behavior beyond vx/ax must not use this for that phase.

    IMPORTANT real mismatch to reconcile at call sites, not yet done: the
    real _forward_self_attn_phase (transformer.py) takes `video`/`audio`
    TransformerArgs objects (not raw vx/ax) and returns a 3-tuple
    `(vx, ax, flags)`, not the 2-tuple this class expects. `flags` is
    deterministic Python metadata derived from `video`/`audio` object
    identity (run_vx/run_ax/run_a2v/run_v2a/perturbations) -- it does NOT
    depend on vx/ax's runtime tensor values, so the correct fix is a
    wrapping closure that calls the real phase once eagerly outside any
    graph to obtain `flags`, then hands THIS class a closure of the form
    `lambda vx, ax: layer._forward_self_attn_phase(replace(video, x=vx), replace(audio, x=ax), ...)[:2]`
    -- i.e. adapt the real 3-arg-in/3-out phase into a 2-in/2-out one at
    the call site, not inside this class.

    Wired into model.py's _process_transformer_blocks as of item -1.0's
    integration (LTXModel.set_layer_graph_capture), using string region
    keys like "L3_self_attn"/"L3_mlp" rather than plain layer indices, so
    `region_key` below accepts any hashable, not just int.
    """

    def __init__(self) -> None:
        self._regions: dict[tuple[object, tuple[int, ...] | None, tuple[int, ...] | None], _CapturedRegion] = {}

    def run(
        self,
        region_key: object,
        phase_fn: Callable[[torch.Tensor | None, torch.Tensor | None], tuple[torch.Tensor | None, torch.Tensor | None]],
        vx: torch.Tensor | None,
        ax: torch.Tensor | None,
        warmup: int = 3,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        vx_shape = tuple(vx.shape) if vx is not None else None
        ax_shape = tuple(ax.shape) if ax is not None else None
        key = (region_key, vx_shape, ax_shape)

        region = self._regions.get(key)
        if region is None:
            region = self._capture(phase_fn, vx, ax, warmup)
            self._regions[key] = region

        if region.static_vx_in is not None:
            region.static_vx_in.copy_(vx)
        if region.static_ax_in is not None:
            region.static_ax_in.copy_(ax)

        region.graph.replay()
        torch.cuda.synchronize()

        out_vx = region.static_vx_out.clone() if region.static_vx_out is not None else None
        out_ax = region.static_ax_out.clone() if region.static_ax_out is not None else None
        return out_vx, out_ax

    def _capture(
        self,
        phase_fn: Callable[[torch.Tensor | None, torch.Tensor | None], tuple[torch.Tensor | None, torch.Tensor | None]],
        vx: torch.Tensor | None,
        ax: torch.Tensor | None,
        warmup: int,
    ) -> _CapturedRegion:
        static_vx_in = vx.clone() if vx is not None else None
        static_ax_in = ax.clone() if ax is not None else None

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                phase_fn(static_vx_in, static_ax_in)
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_vx_out, static_ax_out = phase_fn(static_vx_in, static_ax_in)

        return _CapturedRegion(
            graph=graph,
            static_vx_in=static_vx_in,
            static_ax_in=static_ax_in,
            static_vx_out=static_vx_out,
            static_ax_out=static_ax_out,
        )

    def clear(self) -> None:
        """Drop all captured regions -- call when kv_cache/prompt/other
        assumed-fixed state actually changes (e.g. new generation call),
        since this class cannot detect that on its own (see module
        docstring)."""
        self._regions.clear()
