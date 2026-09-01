import os
from enum import Enum
from typing import Protocol

import torch

from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb

memory_efficient_attention = None
flash_attn_interface = None
flash_attn_func = None
_xformers_unusable = False
_flash_attn3_unusable = False
_flash_attn2_unusable = False
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None
try:
    # Previously gated on `if memory_efficient_attention is None`, i.e. only
    # attempted when xformers wasn't importable at all -- wrong assumption
    # for GPUs (like compute capability 10.3/12.0 Blackwell) where xformers
    # *imports* fine but has no working kernel and fails at call time (see
    # AttentionFunction.DEFAULT below). Import unconditionally so flash-attn
    # is available as a fallback even when xformers "looks" installed but
    # doesn't actually work for this hardware.
    #
    # FlashAttention-3's module is `flash_attn_interface`, a *separate*
    # package/build from the mainline `flash-attn` PyPI package (which is
    # FlashAttention-2 and exposes `flash_attn` instead) -- confirmed on
    # real hardware that `pip install flash-attn` gives the latter, not
    # this one. Both are attempted independently below since either might
    # be the one actually installed.
    import flash_attn_interface
except ImportError:
    flash_attn_interface = None
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

flashinfer_single_prefill = None
_flashinfer_unusable = False
try:
    # Confirmed on real hardware (GB300, compute capability 10.3): imports
    # and runs successfully with no bias/mask argument -- unlike xformers
    # (no kernel for this compute capability at all) and FlashAttention-2/3
    # (either no working kernel here, or an ABI mismatch against the pinned
    # torch build -- see TROUBLESHOOTING.md items -10/-12). This model's
    # causal streaming attention calls never actually pass a mask (also
    # confirmed on real hardware), so FlashInfer's boolean-only custom_mask
    # API isn't even needed here -- a plain unmasked call suffices.
    from flashinfer import single_prefill_with_kv_cache as flashinfer_single_prefill
except ImportError:
    flashinfer_single_prefill = None


def _slice_rope(
    pe: tuple[torch.Tensor, torch.Tensor], start: int, end: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return pe[0][..., start:end, :], pe[1][..., start:end, :]


def update_kv_cache(
    cache: dict,
    start: int,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert one global token range and return the active sink+FIFO window.

    The cache is deliberately inference-only. Repeated denoising forwards for
    the same block replace that block in place; the clean refresh forward then
    replaces it once more before the next block starts.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("causal KV caches are inference-only")
    if k.shape != v.shape or k.ndim != 3:
        raise ValueError("KV tensors must have matching [batch, tokens, dim] shapes")

    length = int(cache["length"])
    old_positions = cache["positions"][:length]
    old_k = cache["k"][:, :length]
    old_v = cache["v"][:, :length]
    end = start + k.shape[1]

    # A denoising step is a transaction over [start, end): discard a previous
    # noisy version of that range while retaining earlier committed history.
    #
    # old_positions is always sorted ascending (tokens are appended in
    # strictly increasing position order, never reordered) -- so
    # `old_positions < start` always selects a contiguous *prefix*, not a
    # scattered subset. Boolean-mask indexing (`old_k[:, keep_old]`, as
    # this used to be written) doesn't know that, and internally calls
    # nonzero() on the mask to find which indices to gather -- a real,
    # measured cost confirmed via torch.profiler (720 aten::nonzero calls
    # in a single forward pass, each forcing a CPU-GPU sync -- see
    # TROUBLESHOOTING.md). Replaced with searchsorted (finds the prefix
    # length directly) + plain slicing, which needs only one small sync
    # (the .item() call) instead of nonzero()'s per-call sync-and-gather.
    n_keep_old = int(torch.searchsorted(old_positions, start).item()) if length > 0 else 0
    positions = torch.cat(
        [old_positions[:n_keep_old], torch.arange(start, end, device=k.device)], dim=0
    )
    merged_k = torch.cat([old_k[:, :n_keep_old], k], dim=1)
    merged_v = torch.cat([old_v[:, :n_keep_old], v], dim=1)

    local = int(cache.get("local_attn_size", -1))
    sink = int(cache.get("sink_tokens", 0))
    if local >= 0 and positions.numel() > local:
        if not 0 <= sink < local:
            raise ValueError(f"expected 0 <= sink_tokens < local_attn_size, got {sink}/{local}")
        # Same reasoning as above: positions is sorted and starts at 0, and
        # sink tokens are never dropped by this function once inserted, so
        # the sink prefix is exactly positions[:sink] -- same result as the
        # original `sink_mask = positions < sink; sink_mask.sum()`, without
        # needing the boolean mask or its sum. The "recent" portion is a
        # suffix (positions >= recent_start), found the same
        # searchsorted-instead-of-nonzero way as above.
        recent_budget = local - sink
        recent_start = max(sink, end - recent_budget)
        recent_start_idx = int(torch.searchsorted(positions, recent_start).item())
        positions = torch.cat([positions[:sink], positions[recent_start_idx:]], dim=0)
        merged_k = torch.cat([merged_k[:, :sink], merged_k[:, recent_start_idx:]], dim=1)
        merged_v = torch.cat([merged_v[:, :sink], merged_v[:, recent_start_idx:]], dim=1)

    active = positions.numel()
    if active > cache["k"].shape[1]:
        raise ValueError(f"KV cache overflow: {active} active tokens exceed capacity {cache['k'].shape[1]}")
    cache["k"][:, :active].copy_(merged_k)
    cache["v"][:, :active].copy_(merged_v)
    cache["positions"][:active].copy_(positions)
    cache["length"] = active
    return cache["k"][:, :active].clone(), cache["v"][:, :active].clone()


class AttentionCallable(Protocol):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


_sdpa_backend_logged = False


def _mask_is_effectively_none(mask: torch.Tensor | None) -> bool:
    """True if mask is None, or a real tensor that's a complete no-op (all
    zeros -- adds zero bias everywhere, restricts nothing). Confirmed on
    real hardware: every mask reaching PytorchAttention in this model's
    causal streaming path is exactly this -- a real (non-None) tensor
    object, but with a single unique value of 0.0. FlashAttention/FlashInfer's
    `if mask is not None: raise` checks were treating these as "has a mask"
    and skipping the fast path even though the mask has zero actual effect;
    this lets AttentionFunction.DEFAULT correctly recognize them as safe to
    fast-path too."""
    if mask is None:
        return True
    return bool(torch.all(mask == 0.0))


def _pytorch_attention_core(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None
) -> torch.Tensor:
    """The reshape -> SDPA -> reshape sequence, factored out so it can
    optionally be torch.compile-d as one unit (see ECHO_WM_COMPILE_ATTENTION
    below) -- compiling only the single SDPA call wouldn't help (it's
    already one fused op), but the surrounding view/transpose/reshape calls
    are genuine separate dispatched ops that compilation could fuse away."""
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    # SDPA has multiple backends (FLASH_ATTENTION, EFFICIENT_ATTENTION,
    # MATH, CUDNN_ATTENTION). Flash's kernel often can't take an
    # arbitrary (non-causal) bias tensor at all and PyTorch silently
    # falls back to MATH -- the slowest backend, always correct but with
    # no fast kernel, which is genuine per-call compute cost that no
    # amount of caching or warmup fixes. Explicitly prefer
    # EFFICIENT_ATTENTION/CUDNN_ATTENTION (both support a bias tensor)
    # over letting PyTorch's default priority silently pick MATH.
    global _sdpa_backend_logged
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
            if not _sdpa_backend_logged:
                print("[attention] using SDPA with explicit backend priority "
                      "[EFFICIENT_ATTENTION, CUDNN_ATTENTION, MATH] (skipping FLASH_ATTENTION, "
                      "which can't take a non-causal bias tensor and would silently fall through "
                      "to MATH -- the slow backend -- if left in the default priority order)", flush=True)
                _sdpa_backend_logged = True
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
            )
    except ImportError:
        # Older torch without torch.nn.attention.sdpa_kernel -- fall back
        # to whatever the default backend priority picks.
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


# Opt-in (ECHO_WM_COMPILE_ATTENTION=1): torch.compile just this one function,
# not the whole model -- unlike the whole-model attempt in item -4 that hit
# a recompilation storm (from a self.idx guard living elsewhere in the
# model, outside this function entirely), scoping compilation to only this
# small, self-contained reshape+SDPA+reshape sequence may avoid that guard
# altogether. mode="default" (not "reduce-overhead"/CUDA graphs -- item -20
# found an unpinned CPU-GPU copy elsewhere in the model breaks graph
# capture; unclear if that's reachable from here, not worth the risk).
# dynamic=True is set upfront (not left to dynamo's default
# guess-then-recompile-once-it-notices behavior) since this model's
# sequence length genuinely varies during KV-cache fill-up before
# stabilizing at the windowed size -- starting dynamic avoids paying that
# discovery-recompile on the first few real blocks. NOT YET BENCHMARKED.
_COMPILE_ATTENTION = os.environ.get("ECHO_WM_COMPILE_ATTENTION", "0") == "1"
_compiled_pytorch_attention_core = None
if _COMPILE_ATTENTION:
    _compiled_pytorch_attention_core = torch.compile(_pytorch_attention_core, dynamic=True)


class PytorchAttention(AttentionCallable):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        fn = _compiled_pytorch_attention_core if _compiled_pytorch_attention_core is not None else _pytorch_attention_core
        return fn(q, k, v, heads, mask)


class XFormersAttention(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        # xformers expects [B, M, H, K]
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            # add a singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # pad to a multiple of 8
            pad = 8 - mask.shape[-1] % 8
            # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
            # but when using separated heads, the shape has to be (B, H, Nq, Nk)
            # in flux, this matrix ends up being over 1GB
            # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            mask_out[..., : mask.shape[-1]] = mask
            # doesn't this remove the padding again??
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            raise NotImplementedError("Mask is not supported for FlashAttention3")

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention2(AttentionCallable):
    """FlashAttention-2, from the mainline `flash-attn` PyPI package
    (module `flash_attn`) -- a different package/module from
    FlashAttention3's `flash_attn_interface` above. Same no-mask
    limitation as FA3 (FA2's `flash_attn_func` has no general bias-tensor
    argument either, only `causal`/`window_size`, neither used here)."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flash_attn_func is None:
            raise RuntimeError("FlashAttention2 was selected but `flash-attn` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            raise NotImplementedError("Mask is not supported for FlashAttention2")

        out = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, causal=False)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashInferAttention(AttentionCallable):
    """FlashInfer's single_prefill_with_kv_cache -- confirmed importable and
    runnable on GB300 (compute capability 10.3) with no mask/bias argument,
    unlike xformers (no kernel for this capability) or FlashAttention-2/3
    (ABI mismatch against the pinned torch build here). Unlike those two,
    this doesn't need mask support at all: this model's causal streaming
    attention calls never actually pass a mask (confirmed on real
    hardware), so a plain unmasked call suffices.

    single_prefill_with_kv_cache's API is per-request (no batch dimension:
    q/k/v are [seqlen, num_heads, head_dim]), unlike the [B, T, H*D] shape
    every other AttentionCallable here takes -- this model only ever runs
    batch=1, so that's handled by indexing/unsqueezing around the call
    rather than a real batching loop.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flashinfer_single_prefill is None:
            raise RuntimeError("FlashInferAttention was selected but `flashinfer` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads
        if b != 1:
            raise RuntimeError(
                f"FlashInferAttention only supports batch_size=1 (single_prefill_with_kv_cache "
                f"has no batch dimension), got batch_size={b}"
            )

        if mask is not None:
            raise NotImplementedError("Mask is not supported by this FlashInferAttention wiring")

        q, k, v = (t.view(b, -1, heads, dim_head)[0] for t in (q, k, v))  # [T, H, D], batch dim dropped
        out = flashinfer_single_prefill(q.to(v.dtype), k.to(v.dtype), v, causal=False)
        out = out.unsqueeze(0).reshape(b, -1, heads * dim_head)
        return out


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASH_INFER = "flash_infer"
    DEFAULT = "default"

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()(q, k, v, heads, mask)
        elif self is AttentionFunction.FLASH_ATTENTION_2:
            return FlashAttention2()(q, k, v, heads, mask)
        elif self is AttentionFunction.FLASH_INFER:
            return FlashInferAttention()(q, k, v, heads, mask)
        else:
            # Strip masks that are real tensors but a complete no-op (all
            # zeros -- see _mask_is_effectively_none) down to None before
            # trying FlashAttention2/3/FlashInfer below: their `if mask is
            # not None: raise` checks otherwise skip these calls even
            # though the mask has zero actual effect. xformers/PyTorch SDPA
            # don't have this issue (they just use the mask as-is, whether
            # it's a no-op or not), so this only matters for those three.
            mask_arg = None if _mask_is_effectively_none(mask) else mask
            # Default behavior: XFormers if installed else PyTorch. "Installed"
            # only means importable, not that it has a working kernel for this
            # GPU -- xformers builds ship a fixed set of prebuilt kernels
            # (fa3/fa2/cutlassF etc.), each gated on specific compute
            # capability ranges, and reject anything outside them at *call*
            # time (NotImplementedError), not import time. A GPU newer than
            # every kernel's supported range (e.g. compute capability 12.0
            # consumer/workstation Blackwell cards on some xformers builds)
            # imports fine and then fails on the very first real call. Try it
            # once; if it doesn't work for this GPU, remember that and use
            # PyTorch's own scaled_dot_product_attention for the rest of the
            # process instead of failing every call.
            global _xformers_unusable
            if memory_efficient_attention is not None and not _xformers_unusable:
                try:
                    return XFormersAttention()(q, k, v, heads, mask)
                except NotImplementedError as exc:
                    _xformers_unusable = True
                    print(
                        f"[attention] xformers has no working kernel for this GPU "
                        f"(falling back to PyTorch SDPA for the rest of this process): {exc}",
                        flush=True,
                    )

            # Second choice, tried only after xformers is confirmed unusable
            # (or never installed): flash-attn's own kernels are a separate
            # question from xformers' bundled fa3/cutlassF kernels -- xformers
            # rejecting this GPU doesn't mean flash-attn does too.
            #
            # Two different failure modes need different handling here:
            # - mask is not None: FlashAttention3.__call__ always rejects
            #   this (NotImplementedError), but it's specific to *this call*
            #   -- other (unmasked) calls elsewhere in the model may still
            #   work fine, so this must NOT permanently disable flash-attn.
            # - Anything else (RuntimeError etc.): a genuine hardware/kernel
            #   incompatibility, which -- like xformers -- will fail every
            #   future call the same way. Remember it, same
            #   try-once-remember-forever pattern as xformers above.
            global _flash_attn3_unusable
            if flash_attn_interface is not None and not _flash_attn3_unusable and mask_arg is None:
                try:
                    return FlashAttention3()(q, k, v, heads, mask_arg)
                except RuntimeError as exc:
                    _flash_attn3_unusable = True
                    print(
                        f"[attention] flash-attn (FlashAttention3) has no working kernel "
                        f"for this GPU (falling back to PyTorch SDPA for the rest of this "
                        f"process): {exc}",
                        flush=True,
                    )

            # Third choice: FlashAttention-2 (mainline `flash-attn` PyPI
            # package, module `flash_attn`) -- a different package from
            # FlashAttention3's `flash_attn_interface` above, tried
            # independently since either might be the one actually
            # installed. Same call-specific-vs-permanent failure handling
            # as FlashAttention3.
            global _flash_attn2_unusable
            if flash_attn_func is not None and not _flash_attn2_unusable and mask_arg is None:
                try:
                    return FlashAttention2()(q, k, v, heads, mask_arg)
                except RuntimeError as exc:
                    _flash_attn2_unusable = True
                    print(
                        f"[attention] flash-attn (FlashAttention2) has no working kernel "
                        f"for this GPU (falling back to PyTorch SDPA for the rest of this "
                        f"process): {exc}",
                        flush=True,
                    )

            # Fourth choice: FlashInfer -- DISABLED by default (set
            # ECHO_WM_FLASHINFER=1 to re-enable for testing). Confirmed
            # working correctly on GB300 (real output, right shape) but
            # confirmed SLOWER than PyTorch SDPA in a first-ever/cold
            # generation (~1.9-2.0s/block vs. SDPA's ~1.6-1.7s, one block
            # spiked to 3.3s) -- see TROUBLESHOOTING.md item -16. Not yet
            # ruled out: FlashInfer's docs mention JIT compilation of
            # kernels, so that measurement may include a one-time
            # per-shape compile cost baked into the very first generation
            # -- untested whether a *second* generation in the same
            # process (same shapes, kernel already compiled) is actually
            # faster than SDPA. Re-enable and compare a second run's block
            # timing (not the first) against the SDPA baseline before
            # concluding anything further.
            _flashinfer_enabled = os.environ.get("ECHO_WM_FLASHINFER", "0") == "1"
            global _flashinfer_unusable
            if (
                _flashinfer_enabled
                and flashinfer_single_prefill is not None
                and not _flashinfer_unusable
                and mask_arg is None
            ):
                try:
                    return FlashInferAttention()(q, k, v, heads, mask_arg)
                except RuntimeError as exc:
                    _flashinfer_unusable = True
                    print(
                        f"[attention] flashinfer has no working kernel for this GPU "
                        f"(falling back to PyTorch SDPA for the rest of this process): {exc}",
                        flush=True,
                    )

            return PytorchAttention()(q, k, v, heads, mask)


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        kv_cache: dict | None = None,
        kv_cache_start: int = 0,
        crossattn_cache: dict | None = None,
    ) -> torch.Tensor:
        """Multi-head attention with optional RoPE, perturbation masking, and per-head gating.
        When ``perturbation_mask`` is all zeros, the expensive query/key path
        (linear projections, RMSNorm, RoPE) is skipped entirely and only the
        value projection is used as a pass-through.
        Args:
            x: Query input tensor of shape ``(B, T, query_dim)``.
            context: Key/value context tensor of shape ``(B, S, context_dim)``.
                Falls back to ``x`` (self-attention) when *None*.
            mask: Optional attention mask. Interpretation depends on the attention
                backend (additive bias for xformers/PyTorch SDPA).
            pe: Rotary positional embeddings applied to both ``q`` and ``k``.
            k_pe: Separate rotary positional embeddings for ``k`` only. When
                *None*, ``pe`` is reused for keys.
            perturbation_mask: Optional mask in ``[0, 1]`` that
                blends the attention output with the raw value projection:
                ``out = attn_out * mask + v * (1 - mask)``.
                **1** keeps the full attention output, **0** bypasses attention
                and passes the value projection through unchanged.
                *None* or all-ones means standard attention; all-zeros skips
                the query/key path entirely for efficiency.
            all_perturbed: Whether all perturbations are active for this block.
        Returns:
            Output tensor of shape ``(B, T, query_dim)``.
        """
        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q = self.q_norm(self.to_q(x))
            if crossattn_cache is not None:
                if not crossattn_cache["is_init"]:
                    cached_k = self.k_norm(self.to_k(context))
                    size = cached_k.shape[1]
                    crossattn_cache["k"][:, :size].copy_(cached_k)
                    crossattn_cache["v"][:, :size].copy_(v)
                    crossattn_cache["length"] = size
                    crossattn_cache["is_init"] = True
                size = int(crossattn_cache["length"])
                k = crossattn_cache["k"][:, :size]
                v = crossattn_cache["v"][:, :size]
                if pe is not None:
                    q = apply_rotary_emb(q, pe, self.rope_type)
            else:
                k = self.k_norm(self.to_k(context))
                local_pe = kv_cache.get("local_rope_pe") if kv_cache is not None else None
                local_q_pe = kv_cache.get("local_cross_q_rope_pe") if kv_cache is not None else None
                local_k_pe = kv_cache.get("local_cross_k_rope_pe") if kv_cache is not None else None
                if local_pe is not None:
                    k, v = update_kv_cache(kv_cache, kv_cache_start, k, v)
                    active = k.shape[1]
                    q_len = q.shape[1]
                    q = apply_rotary_emb(q, _slice_rope(local_pe, active - q_len, active), self.rope_type)
                    k = apply_rotary_emb(k, _slice_rope(local_pe, 0, active), self.rope_type)
                elif local_q_pe is not None or local_k_pe is not None:
                    if local_q_pe is None or local_k_pe is None:
                        raise ValueError("cross-modal RoPE rebase requires both query and key templates")
                    new_keys = k.shape[1]
                    k, v = update_kv_cache(kv_cache, kv_cache_start, k, v)
                    query_slice = kv_cache["local_cross_q_slices"].get((kv_cache_start, kv_cache_start + new_keys))
                    if query_slice is None:
                        raise ValueError("missing local cross-modal query RoPE slice")
                    q = apply_rotary_emb(q, _slice_rope(local_q_pe, *query_slice), self.rope_type)
                    k = apply_rotary_emb(k, _slice_rope(local_k_pe, 0, k.shape[1]), self.rope_type)
                else:
                    if pe is not None:
                        q = apply_rotary_emb(q, pe, self.rope_type)
                        k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)
                    if kv_cache is not None:
                        k, v = update_kv_cache(kv_cache, kv_cache_start, k, v)

            out = self.attention_function(q, k, v, self.heads, mask)  # (B, T, H*D)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        # Apply per-head gating if enabled
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, T, H)
            b, t, _ = out.shape
            # Reshape to (B, T, H, D) for per-head gating
            out = out.view(b, t, self.heads, self.dim_head)
            # Apply gating: 2 * sigmoid(x) so that zero-init gives identity (2 * 0.5 = 1.0)
            gates = 2.0 * torch.sigmoid(gate_logits)  # (B, T, H)
            out = out * gates.unsqueeze(-1)  # (B, T, H, D) * (B, T, H, 1)
            # Reshape back to (B, T, H*D)
            out = out.view(b, t, self.heads * self.dim_head)

        return self.to_out(out)
