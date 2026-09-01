# Troubleshooting: `gradio_echo_wm.py` (Flash Preview / streaming UI)

## -15. torch/CUDA environment corruption chasing FlashAttention-2 (fixed, new DGX-specific requirements file)

Chasing item -12's FlashAttention-2 install (`pip install flash-attn
--force-reinstall --no-cache-dir`) dragged in a cascade of mismatched
dependency upgrades on the GB300 box: `torch` silently jumped to a
**nightly dev build** (`2.15.0.dev20260830+cu132`), with `torchaudio`
(2.11.0) and `torchvision` (0.30.0.dev) landing on different, inconsistent
versions -- none matching this repo's pin (`torch==2.9.1`,
`requirements.txt`). First symptom: `torchaudio` import crashed outright
(`RuntimeError: PyTorch and TorchAudio were compiled with different CUDA
versions`), blocking the app from starting at all.

**Recovery took multiple attempts, two of which made things worse:**
- Reinstalling the pinned versions via the **cu128** index (per
  `requirements.txt`'s own comment) failed to find a matching wheel --
  this box is **aarch64** (GB300/Grace), and cu128 has no aarch64 builds
  for this platform.
- Reinstalling via **cu132**, unpinned (`pip install torch torchaudio
  torchvision --index-url .../cu132`, no version numbers): silently
  produced a **CPU-only** build (`torch 2.9.1+cpu`, `CUDA available:
  False`) -- no error, no warning, just silently dropped GPU support. This
  is what produced the misleading "masks are always None" finding in item
  -12/-13's earlier measurement -- almost certainly measured against this
  broken CPU build, not a real GPU run.
- The actual fix: `setup_and_run.sh` (the script that originally
  provisioned this exact box) already had the right answer --
  `pip install --index-url https://download.pytorch.org/whl/cu130
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1`, explicitly
  excluding xformers entirely on aarch64 (`grep -v '^xformers'
  requirements.txt`, since there's no aarch64 xformers wheel at all --
  independently consistent with everything else found about xformers not
  working on this GPU regardless of wheel availability).
- That cu130/2.9.1 combination *did* reinstall successfully, but a
  subsequent (unintended) unpinned reinstall onto **cu132** landed on
  `torch==2.14.0+cu132` / `torchaudio==2.11.0` / `torchvision==0.29.0`
  instead -- a different, newer, but apparently GPU-working combination.
  Rather than fight back to exactly 2.9.1 a third time, this became the
  new documented baseline for this box (see below) since it's the one
  that's actually confirmed to have `torch.cuda.is_available() == True`.

**New file**: `requirements_dgx.txt` -- a DGX-GB300-specific (aarch64,
CUDA 13.2) variant of `requirements.txt`, pinning `torch==2.14`,
`torchaudio==2.11`, `torchvision==0.29.0` (install via the `cu132` index,
documented in the file itself), no `xformers` (no aarch64 wheel, ever, on
any torch version -- unrelated to this specific ABI saga), and
`flashinfer-python` included. **Always verify after any torch-adjacent
install on this box** with
`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
-- this session hit two different *silent* failure modes (ABI mismatch
crash, and silent CPU-only fallback) from mismatched CUDA-tagged indices
before landing on a working combination.

## -14. Masks are real tensors but all-zero (a no-op) -- FlashAttention/FlashInfer were rejecting them anyway (fixed)

Once the torch/CUDA environment (item -15) was working again, the
mask-diagnostic from item -12/-13 actually fired for the first time and
gave a real, different answer than "always `None`": every mask reaching
`PytorchAttention` is a genuine (non-`None`) tensor, but with exactly one
unique value: `0.0`. An all-zero additive bias restricts nothing -- it's
mathematically a complete no-op, identical in effect to no mask at all.

This matters because `FlashAttention2`/`FlashAttention3`/`FlashInferAttention`
all hard-reject on `mask is not None` (raising `NotImplementedError`) --
so every single call was hitting that rejection and falling back to SDPA,
even though the mask had zero actual effect and the fast kernels would
have produced identical output. The earlier "masks are always None"
conclusion (item -12) was measured on a *different, broken* torch install
(the CPU-only fallback -- see item -15) where the model may not have
reached these particular code paths at all; this is the first measurement
taken with a genuinely working GPU torch build.

**Fixed**: added `_mask_is_effectively_none(mask)` (`attention.py`) --
`True` for `None` or an all-zero tensor. `AttentionFunction.DEFAULT` now
computes `mask_arg = None if _mask_is_effectively_none(mask) else mask`
once, and passes `mask_arg` (not the raw `mask`) into the
FlashAttention2/3/FlashInfer attempts -- so a no-op mask gets stripped to
a real `None` before reaching their hard-rejection checks, letting the
fast paths actually engage instead of always falling through to SDPA.
xformers/PyTorch SDPA are unaffected (no mask-presence hard-rejection to
begin with, so passing the original `mask` there is harmless either way).

The mask-diagnostic instrumentation itself was removed after this
confirmed the answer (no longer needed).

## -13. FlashInfer wired in as a fourth attention fallback (implemented, not yet benchmarked)

Direct continuation of item -12's FlashAttention-2 attempt, which turned
out to be a dead end (see below): item -10's original FlashInfer
investigation found it supports SM100/SM103 (GB300's compute capability
10.3) and, unlike xformers/FlashAttention, has a real (if boolean-only)
custom-mask API -- but that investigation got shelved once FlashAttention
briefly looked more promising. Since item -12 established this model's
causal streaming attention calls **never actually pass a mask at all**,
FlashInfer's mask API question is moot -- a plain unmasked call is all
that's needed, and that exact call
(`flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False)`) was
already confirmed importable and runnable on the real GB300 box in a
standalone test (`echo_wm/test_flashinfer_bias.py`) before any of the
FlashAttention detour happened.

**FlashAttention-2, tried in between, is a confirmed dead end** -- not a
missing-package or wrong-build-flag issue, but incompatible at the
compiled-CUDA-extension ABI level against the pinned `torch==2.9.1`:
- `pip install flash-attn` (prebuilt wheel): `undefined symbol:
  ..._c10_cuda_check_implementation...`
- `pip uninstall -y flash-attn && pip install flash-attn
  --no-build-isolation --force-reinstall --no-cache-dir` (forced source
  rebuild against the actual installed torch): still failed, but with a
  **different** undefined symbol (`materialize_cow_storage`) -- proof the
  rebuild genuinely happened and still doesn't match. flash-attn
  2.8.3.post1 does not currently work against this torch build, full stop.
  A `FlashAttention2` class (mirroring `FlashAttention3`'s structure,
  using `flash_attn.flash_attn_func`) was still added to `attention.py`
  for completeness/future use -- it gracefully no-ops (via the existing
  `except ImportError` guard) since the import fails, so it costs nothing
  to leave in.

**FlashInfer integration** (`FlashInferAttention` class in `attention.py`):
mirrors the `FlashAttention2`/`3` structure, with one real difference --
`single_prefill_with_kv_cache`'s API has **no batch dimension** (`q`/`k`/`v`
are `[seqlen, num_heads, head_dim]`, not `[B, T, H*D]` like every other
`AttentionCallable` here), handled by indexing out batch 0 before the call
and unsqueezing back after (this model only ever runs `batch_size=1`;
`batch_size != 1` raises `RuntimeError`, treated as a permanent
incompatibility like a real hardware/kernel failure would be, not a
call-specific one). Wired as the fourth stage in `AttentionFunction.DEFAULT`:
xformers -> FlashAttention3 -> FlashAttention2 -> **FlashInfer** -> SDPA.
Added `flashinfer-python` to `requirements.txt` (uncommented, since it's
confirmed importable/runnable here, unlike the commented-out
confirmed-broken `flash-attn`).

**Not yet benchmarked end-to-end through a real generation** -- confirmed
working at the standalone-script level (item -10), not yet confirmed via
`[rollout]` timing that it's actually engaging (watch for the *absence* of
both the xformers-rejection and any flash-attn-rejection messages, i.e.
`grep -i "attention\]" gradio_debug.log` showing nothing at all instead of
the `using SDPA...` line) or that it produces a real speed change.

Unrelated regression hit while testing this, not caused by any of the
above: `pip install flash-attn --force-reinstall --no-cache-dir` appears
to have pulled in a `torchaudio` dependency resolution that no longer
matches the installed `torch` build's CUDA version (`torch` reports CUDA
13.0, `torchaudio` reports CUDA 13.2) --
`RuntimeError: Detected that PyTorch and TorchAudio were compiled with
different CUDA versions`. Blocks the app from starting at all. Not yet
fixed -- needs `torchaudio` reinstalled from whatever CUDA-13.0-tagged
index `torch` itself was originally installed from (matching this repo's
own `torch`/`xformers` CUDA-alignment pattern elsewhere in
`requirements.txt`), not investigated further this session since the
FlashInfer work took priority.

## -12. FlashAttention-3 re-opened as an option: masks are always None on the causal streaming path (implemented, not yet tested)

Item -10's FlashAttention-4 investigation assumed every attention call in
this model passes a real bias/mask tensor, based on the original xformers
rejection log (`attn_bias: <class 'torch.Tensor'>`). **That assumption was
wrong for the causal streaming path specifically.** Added a diagnostic to
`PytorchAttention.__call__` (the backend actually running, since xformers
has no working kernel here) logging any non-None mask it receives -- across
20 real blocks (two full 10-block generations) on the GB300 box, **zero**
non-None masks were observed. The original rejection log's real bias
tensor almost certainly came from a different code path (likely the base/
non-causal engine's CFG-batched bidirectional attention, never confirmed
directly). Diagnostic removed after confirming this (was briefly in
`transformer.py` at the wrong call site first -- self_attention_mask
there is *also* always None for this path, the KV-cache windowing is done
by physically truncating cache tensors in `update_kv_cache`, not by
masking a full-length attention matrix).

Since FlashAttention-3's only hard limitation in this codebase
(`if mask is not None: raise NotImplementedError`) never triggers when
mask is always None, **`FlashAttention3` -- already fully implemented in
`attention.py`, previously just unreachable -- is now genuinely viable**
for this GPU, pending only real kernel/hardware compatibility (separate
question from xformers' bundled-kernel rejection).

Two real fixes needed to actually reach it:
1. **Import guard bug**: `flash_attn_interface` was only ever imported
   `if memory_efficient_attention is None` -- i.e. only when xformers
   failed to *import* at all. Wrong for GPUs where xformers imports fine
   but fails at *call* time (exactly this GPU's situation) -- meant
   flash-attn was never even attempted, regardless of whether it was
   installed. Now imported unconditionally.
2. **`AttentionFunction.DEFAULT` fallback chain extended**: xformers ->
   FlashAttention3 (only when `mask is None`, since a masked call failing
   is call-specific, not a hardware-incompatibility signal -- doesn't
   permanently disable it) -> PyTorch SDPA. A genuine `RuntimeError` from
   FlashAttention3 (real kernel/hardware incompatibility) *does*
   permanently disable it for the rest of the process, same
   try-once-remember-forever pattern as the existing xformers fallback.

**Update: FlashAttention-3 specifically is a dead end, but FlashAttention-2
picked up the torch and is now wired in too.** `pip install flash-attn` on
the GB300 box gave `flash_attn` 2.8.3.post1 -- **FlashAttention-2**, whose
module is `flash_attn` (function `flash_attn_func`), not FlashAttention-3's
`flash_attn_interface` (a separate package/build entirely, not what plain
`pip install flash-attn` provides). The FA3 code path in `attention.py`
correctly detected `flash_attn_interface` was unavailable and silently
skipped it (confirmed via `grep -i "attention\]" gradio_debug.log` --
only the SDPA fallback message appeared, no FA3-failure message either,
meaning it was never attempted at all).

Added a real `FlashAttention2` class (mirrors `FlashAttention3`'s
structure, calls `flash_attn.flash_attn_func(q, k, v, causal=False)`) and
a third fallback stage in `AttentionFunction.DEFAULT`: xformers ->
FlashAttention3 -> **FlashAttention2** -> PyTorch SDPA. Same call-specific
vs. permanent-failure handling as FA3 (`mask is not None` only skips that
one call; a real `RuntimeError` permanently disables it for the rest of
the process). **Not yet tested on real hardware as of this writing** --
restart and check `grep -i "attention\]" gradio_debug.log`: either
`[attention] flash-attn (FlashAttention2) has no working kernel for this
GPU...` (dead end too, falls back to SDPA same as before) or no such
message at all (meaning it's actually being used) plus a real speed
change in `[rollout]` timing.

## -11. `ltx_core.loader` <-> `ltx_core.quantization` order-dependent circular import (fixed)

Hit when adding the `ECHO_WM_FP8` import (item -9):
```
ImportError: cannot import name 'calculate_weight_float8' from partially
initialized module 'ltx_core.quantization.fp8_cast' (most likely due to a
circular import)
```
Real pre-existing circular dependency in `ltx_core` itself:
`ltx_core/loader/fuse_loras.py` imports from `ltx_core.quantization.fp8_cast`,
which imports from `ltx_core.loader.module_ops` -- a genuine cycle between
the two packages. It only "works" elsewhere in the codebase (e.g.
`causal_ti2vid.py`) by accident of import order: whichever package's
`__init__.py` starts executing *first* determines whether the cycle
resolves cleanly or not. `causal_ti2vid.py` imports `ltx_core.loader`
(line 12) before `ltx_core.quantization` (line 16), which happens to
work. `gradio_echo_wm.py`'s new `from ltx_core.quantization import
QuantizationPolicy` was the *first* thing to touch either package,
entering the cycle from the opposite direction -- and failed, because
`fp8_cast.py`'s `calculate_weight_float8` function (needed by
`fuse_loras.py`) isn't defined yet at the point the cycle loops back to
it from that direction.

**Fixed** by reordering (not restructuring `ltx_core`'s actual circular
dependency, which is out of scope): moved the `QuantizationPolicy` import
in `gradio_echo_wm.py` to *after* `from ltx_pipelines.causal_ti2vid import
CausalTI2VidPipeline`, so `ltx_core.quantization` is already safely
resolved (via `causal_ti2vid.py`'s own working import order) by the time
`gradio_echo_wm.py` touches it directly. **General lesson for this repo**:
any future top-level import of `ltx_core.quantization` in this file must
come after something that imports `ltx_core.loader` first (any
`ltx_pipelines.*` import already does), not before.

## -10. FlashAttention-4 investigated as a possible xformers replacement on GB300 (not integrated)

xformers has no working kernel for compute capability 10.3 (GB300) or 12.0
(horde) at all (item -3) -- checked whether a newer xformers release fixed
this: the box already had **xformers 0.0.35** installed (newer than this
repo's pinned `0.0.33.post2` in `requirements.txt`), and it *still* rejects
this GPU the same way. Confirmed via web search that xformers itself has
not added Blackwell (sm_120 consumer, and status unclear for sm_100/103
datacenter) kernel support as of this writing.

Separately found: **FlashAttention-4** (`flash-attn-4` on PyPI -- a
different package from `flash-attn`/xformers, not a version bump of
either) explicitly targets NVIDIA's **SM100/SM103** datacenter Blackwell
GPUs, i.e. B200/B300/**GB300** by name -- a real, specific hardware match
for the GB300 box (compute capability 10.3), confirmed via web search, not
a guess. Requires CUDA 12.8+ (earlier CUDA builds carry no Blackwell
kernels at all per the project's own setup.py gencode logic). Only beta
releases exist on PyPI as of this writing (`4.0.0b3` through `4.0.0b28`,
no stable release) -- `pip install flash-attn-4` fails outright without
`--pre` or an exact version pin (`pip install flash-attn-4==4.0.0b28`,
confirmed installable this way on the GB300 box).

**Real, unresolved risk found before any integration work started:**
looking back at the original xformers rejection log (item -3), **FA2 was
already rejected for a reason independent of compute capability**:
```
`fa2F@2.5.7-pt` is not supported because:
    attn_bias type is <class 'torch.Tensor'>
```
This model passes a real additive bias tensor into attention (not a
boolean causal mask), and FlashAttention kernels have historically had
limited/no support for arbitrary bias tensors -- optimized for
causal/no-mask patterns instead. It's unconfirmed whether FlashAttention-4
broadened bias-tensor support enough to accept this model's masks, or
whether it would reject them the same structural way regardless of
matching the GPU generation perfectly. **Not yet integrated** --
`flash-attn-4` is listed (commented out) in `requirements.txt` for
reference, but nothing in `ltx-core/.../attention.py` calls it yet. Before
sinking real integration effort (a third attention backend path,
analogous to the existing xformers-then-SDPA fallback) into this, worth a
cheap standalone check of whether FA4's Python API even accepts a
tensor-shaped bias argument at all -- not yet done.

## -9. `ECHO_WM_FP8=1`: opt-in fp8 weight storage (implemented, not yet benchmarked)

`ltx_core/quantization/` already has a working, dependency-free
`QuantizationPolicy.fp8_cast()` -- discovered while looking into whether fp8
quantization was worth pursuing after item -7's "confirmed real, not a
timing artifact" cache-update finding raised the possibility that some of
our fixed costs might be memory-bandwidth-bound rather than purely
compute-bound. Very low effort to wire in: `CausalTI2VidPipeline.__init__`
already accepted a `quantization: QuantizationPolicy | None` param
(unused by `EchoWMCausalEngine` until now), so this was just threading
`QuantizationPolicy.fp8_cast()` through when `ECHO_WM_FP8=1`
(`gradio_echo_wm.py`'s `EchoWMCausalEngine.__init__`) -- no new
dependencies, `ModelLedger.transformer()` already had the sd_ops/module_ops
application logic in place.

**Important nuance, not "real" fp8 compute:** `fp8_cast()` downcasts
transformer linear weights to `float8_e4m3fn` **for storage only** --
`UPCAST_DURING_INFERENCE` (`fp8_cast.py`) replaces each `nn.Linear`'s
forward to upcast weights back to bf16 immediately before every matmul.
So this halves weight memory footprint/bandwidth, but the actual multiply
still happens in bf16 -- it targets memory-bandwidth-bound cost, not
FLOPs. (A true native-fp8-compute path exists too --
`QuantizationPolicy.fp8_scaled_mm()` -- but requires `tensorrt_llm`, not
installed, out of scope for a quick test.) Real, if modest, precision
loss from the fp8 rounding step, additive to (not a replacement for) the
timesteps/attn_window quality trades already in place.

Default off (`ECHO_WM_FP8=0` in `run_gradio.sh`). **Not yet tested on real
hardware** -- unknown whether this actually moves the needle at all,
given item -7's confirmed-flat cache-update cost might not be
memory-bandwidth-bound in the way that would make this help.

## -8. Live-preview audio drop-outs between blocks (fixed, second attempt)

Video and audio decode on independent cadences in this pipeline (separate
"newly available" checks in `causal_ti2vid.py`'s `raw_on_block`) -- some
blocks carry real new video frames but no new audio yet (it catches up in
a later block). `StreamingEncoder.write_chunk()` only muxes audio when
given a real chunk, so those video-only blocks left a genuine gap in the
audio track's timeline while video kept advancing -- audible as
silence/drop-outs, confirmed by the reporting user as "silence" specifically
(not clicks/pops), which ruled out a timestamp-continuity/artifact
explanation in favor of a real missing-content one.

**First attempt (reverted, broke audio entirely):** filled gaps with
hardcoded digital silence at a guessed 48000 Hz sample rate. Broke audio
completely, not just left it gap-y. Root cause (inferred, not directly
confirmed): `StreamingEncoder` uses one **persistent** `AudioResampler`
across the whole generation; feeding it real chunks at their actual rate
and silence chunks declared at a different, mismatched rate on the same
stateful resampler instance very likely corrupted its internal state
(`libswresample`, which PyAV's resampler wraps, isn't built to handle the
declared input rate changing mid-stream cleanly).

**Second attempt (current):** instead of synthesizing silence, loop the
most recent *real* audio chunk (`last_audio` dict: waveform + its actual
`sampling_rate`, updated whenever a real chunk arrives) to fill a
video-only block's exact duration, always using that chunk's real,
already-consistent sampling rate -- never a guessed one. Falls back to
skipping audio only when no real chunk has arrived yet at all (e.g. the
very first block). Trades a possibly-audible loop/stutter during gaps for
guaranteed timeline continuity and no resampler-rate-mismatch risk. Not
yet confirmed working on real hardware as of this writing.

## -7. Speed tuning summary + step-count/attention-window exposed as live UI dropdowns

Cumulative result of this session's speed work (measured via `[rollout]`
per-block timing and the Status panel's "avg generation speed"):
**~6.88fps -> ~9.80fps** average generation speed, and steady-state
per-block time **~3.5s -> 1.5s** -- landing almost exactly on the ~1.5s
real-time budget at 512x288/16fps. In order of real, confirmed impact:
1. Attention window `video_local_attn_size`/`video_sink_size` 19/7 -> 10/4
   -> **4/1** (the minimum `CausalCacheConfig.validate()` allows -- see
   its constraints: `0 < sink < local_attn`, `local_attn - sink >=
   video_chunk_size`, both `1 + n*video_chunk_size`). Confirmed real (not
   a `torch.compile`-bug artifact like the step-count numbers initially
   were): denoise dropped ~1.0-1.1s -> ~0.9s across these three cuts.
2. Denoising steps 4 (native) -> 2 -- see item -4 for how this was
   initially miscalibrated by the `torch.compile` recompilation bug and
   only became trustworthy once that was fixed.
3. Encode work pipelined off the rollout thread (item -6).
4. Warmup resolution fix (item -5) -- doesn't affect steady-state speed,
   but stops the cold-start cost from landing on a real user's first
   request.

**Confirmed real, not a timing artifact:** cache-update `forward()`'s ~0.5s
cost stayed flat across the 19/7 -> 10/4 -> 4/1 attention-window cuts, even
though it's literally the same model call as denoise (which did respond).
Initially suspected a CUDA-async measurement artifact (Python's
`time.time()` without `torch.cuda.synchronize()` calls between segments
can attribute one segment's actual GPU-busy time to a different segment,
since CUDA kernels launch asynchronously) -- tested this directly by
temporarily bracketing each timed segment with `torch.cuda.synchronize()`
in `rollout.py`. Result with honest, synchronized measurement: still flat
~0.5s, total block time barely changed (~1.6-1.7s vs. the unsynchronized
~1.7s). **Theory refuted -- this is a real, structural fixed cost**, not a
measurement bug. Whatever's expensive isn't the windowed self-attention
(`video_local_attn_size`/`video_sink_size`); more likely candidates:
cross-attention against the full (unwindowed) text/action context, or the
KV-cache write/bookkeeping itself. Would need real `torch.profiler`
instrumentation inside that specific `forward()` call to pin down further
-- not attempted, since we're already near the ~1.5s real-time budget and
this isn't reachable through any lever already exposed in the UI/config.
The diagnostic `torch.cuda.synchronize()` calls were removed from
`rollout.py` after answering this (they add real overhead, not something
to leave running permanently).

**UI exposure:** both step count and attention window are now live
dropdowns in the causal UI's "Video Settings" (`STEP_PRESETS`/
`ATTENTION_PRESETS` in `gradio_echo_wm.py`, "(config default)" plus a few
presets each) instead of only being editable via `configs/*.yaml` +
restart. Attention window required extending `CausalTI2VidPipeline`'s
per-resolution model cache (`self._model_cache`, added in item -4) to key
on `(width, height, video_local_attn_size, video_sink_size)` instead of
just `(width, height)` -- different windows need different-sized KV-cache
tensors (see `CausalModelWrapper.init_caches`), so they can't share a
cached model instance. `CausalTI2VidPipeline.__call__` gained an
`attn_window: tuple[int, int] | None` param that, when given, builds a
one-off `CausalCacheConfig` for that call instead of using
`self.cache_config` (validated the same way).

## -6. Encode work pipelined off the rollout thread (confirmed working)

Measured steady-state per-block breakdown (once the `torch.compile` bug in
item -4 was ruled out): `denoise (~1.0-1.1s) -> on_block callback (~0.4s)
-> cache-update forward() (~0.4s)` = ~1.9s/block total, all strictly
sequential on one thread. The callback's real work
(`StreamingEncoder.write_chunk()` -- CPU-bound numpy conversion +
libx264/AAC encode, then a WebSocket push) only depends on *this* block's
already-denoised output -- nothing about it needs to finish before
cache-update `forward()` or the next block's denoise can start.

Fixed in `EchoWMCausalEngine.generate()`: `on_block` no longer calls
`stream_encoder.write_chunk()` directly. Instead it hands
`(video_chunk, audio_chunk)` to a FIFO `encode_queue`, consumed by a
dedicated background thread (`_encode_worker`) that does the actual
encode+push. `on_block` returns almost immediately, so the rollout
thread moves straight to cache-update `forward()` (and the next block's
denoise) while encoding happens concurrently. FIFO ordering matters here:
muxed byte order must match block order, so a single consumer thread (not
a thread pool) processes the queue serially. At generation end (both the
normal-completion and error paths), a `None` sentinel is pushed and the
encode thread is joined *before* calling `stream_encoder.close()` --
otherwise `close()`'s flush could race ahead of a still-pending block's
`write_chunk()`.

**Confirmed on real hardware.** `on_block callback returned` time dropped
from ~0.4s to ~0.1-0.2s (not fully to ~0s -- the overlap isn't perfect,
some GIL/scheduling contention as flagged as a risk below, but still a
real, substantial win). Combined with the other changes in item -7,
steady-state total block time reached ~1.5-1.6s, at or very near the
~1.5s real-time budget.

## -5. Warmup used a different resolution than the real config, so it didn't actually warm anything (fixed)

Confirmed on real hardware: even with warmup running to completion before
the server started accepting requests, the **first real user generation**
still paid a huge cold-start hit -- block 0 took 21.2s and block 1 took
6.0s before settling to ~1.0-1.1s/block steady state. Cause: `_warmup()`
was hardcoded to 128x64, while the real config is 512x288 -- cuDNN/cuBLAS
kernel selection and CUDA JIT compilation are shape-dependent, so warming
up at one resolution doesn't warm the kernels needed at another. Fixed:
`_warmup()` now reads `width`/`height`/`fps` from the engine's actual
loaded config (falling back to 512/288/16 if unset) instead of a
hardcoded throwaway size, while deliberately keeping `num_frames=25`
(minimal valid causal block count) -- block count doesn't affect
per-block kernel shape, only width/height/`video_chunk_size` do, so more
warmup blocks would add time without warming anything new.

## -4. `ECHO_WM_COMPILE=1`: opt-in torch.compile for the causal transformer (tried, confirmed net loss -- default off)

`ModelLedger.transformer()` (`utils/model_ledger.py:219`) builds a
brand-new model object from scratch on **every** `CausalTI2VidPipeline.__call__`
-- naively wrapping that in `torch.compile()` would mean paying full
graph-tracing cost on every single generation (same failure mode as the
CUDA-cache-size tuning in item -3.6 that turned out not to help). Fixed the
precondition for compiling to pay off at all: `causal_ti2vid.py` caches the
built model per `(width, height)` (`self._model_cache`) -- built once,
reused across every later generation at that resolution -- and, only when
`ECHO_WM_COMPILE=1` is set, wraps `x0_model.velocity_model` in
`torch.compile()` (default mode, deliberately not `reduce-overhead`/
CUDA-graphs, for the reasons below).

**Confirmed net loss on real hardware (`10.74.11.118`).** Exactly the risk
predicted before testing: `torch._dynamo` hit a **repeated recompilation
storm**, not a one-time compile cost --
```
W ... torch/_dynamo/convert_frame.py:2048] [14/8] User stack trace:
W ...   File ".../ltx_core/model/transformer/transformer.py", line 440, in torch_dynamo_resume_in_forward_at_405
W ...     if self.idx >= int(self.num_layers * 0.7)
```
(HINT from dynamo: "torch.compile considers integer attributes of the
nn.Module to be static... you might want to make this integer dynamic").
Warmup got stuck on **block 0 of a 1-block warmup run** -- normally ~2s --
for **>100s and still climbing** (heartbeats every 2s past t=107.7s) before
being interrupted. This is a real regression, not "maybe slightly better,
maybe a wash" as originally estimated -- **default is now `ECHO_WM_COMPILE=0`**
in `run_gradio.sh`. Leaving the model-caching infrastructure in place (that
part is harmless/correct on its own, unrelated to the compile-specific
recompilation bug) in case this is revisited later with a fix for the
`self.idx` guard (e.g. `torch._dynamo.config.allow_unspec_int_on_nn_module = True`,
per dynamo's own hint, or restructuring that comparison to avoid a
per-layer Python-int guard entirely) -- not attempted this session.

## -3. Blackwell consumer/workstation GPUs (RTX 5090, RTX PRO 6000 -- compute capability 12.0): xformers/SDPA gotchas (fixed)

Hit on `kschmid-4vvboh` ("horde"), a compute-capability-**12.0** GPU
(`sm_120`, RTX 5090/PRO 6000-class -- distinct from the GB300 box used
elsewhere in this repo's docs, which is `sm_103`).

**(a) xformers has no working kernel for this GPU at all (fixed).**
`memory_efficient_attention` imports fine (so `AttentionFunction.DEFAULT`
picks it) but crashes on the *first real call* with
`NotImplementedError: No operator found ... requires device with
capability <= (9, 0)/== (8, 0) but your GPU has capability (12, 0) (too
new)` -- every bundled kernel (fa3F, fa2F, cutlassF) explicitly excludes
this capability. Fixed in `ltx-core/.../model/transformer/attention.py`:
`AttentionFunction.DEFAULT` now tries xformers once, catches
`NotImplementedError`, and falls back to `PytorchAttention` (SDPA) for the
rest of the process (`_xformers_unusable` module flag) instead of crashing
every call. Confirmed harmless for GPUs where xformers *does* work --
still tried first every time, unchanged behavior there.

**(b) SDPA's own default backend priority can silently pick the slow MATH
kernel (fixed).** Once xformers falls back to `PytorchAttention`, it calls
`torch.nn.functional.scaled_dot_product_attention` with a real (non-causal)
`attn_mask` tensor. FLASH_ATTENTION's kernel generally can't take an
arbitrary bias tensor, so PyTorch's default backend priority can silently
fall through to MATH -- correct, but with no fast kernel, meaning genuine
per-call compute cost that no warmup/caching fixes (confirmed: warmup time
did not improve across multiple repeated restarts before this fix, ruling
out a one-time-JIT-cost explanation). Fixed: explicitly wrap the SDPA call
in `torch.nn.attention.sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH])` so EFFICIENT_ATTENTION/
CUDNN_ATTENTION (both support a bias tensor) get a chance before MATH.
Logs which backend priority is active on first use
(`[attention] using SDPA with explicit backend priority ...`) -- note this
only confirms which backends are *allowed*, not which one actually
engaged internally; if EFFICIENT_ATTENTION/CUDNN_ATTENTION also lack
sm_120 kernels in this torch build, it would still silently fall through
to MATH regardless of this fix. Not yet confirmed which backend actually
engages on this GPU -- only that total generation time did drop after this
change (see item (d)).

**(c) `pip install sageatten` / `pip install flash_atten` fail — typos, not
missing packages.** Correct names: `sageattention`, `flash-attn`
(hyphenated). Neither installed/evaluated yet this session; unconfirmed
whether either has real sm_120 kernel support (same class of gap as (a) is
a real risk before assuming either fixes anything).

**(d) End-to-end confirmed working** after (a)+(b)+the `num_frames` fix in
item -3.5 below: real generation, ~4s/block (denoise ~2.2s + callback/cache
overhead ~1.8s) across 10 blocks, live preview showing real streaming
content in the browser.

## -3.5. Invalid `num_frames=2` in causal warmup (fixed)

A later "shorten warmup" edit set the causal engine's warmup
`num_frames=2` (comment claimed "2 frames is the minimum that still
produces >1 block") -- **mathematically impossible** for this config and
crashed immediately (`ValueError: causal --num-frames must be 1 + 8*n
output frames`). The causal pipeline requires `num_frames = 1 + 8*n`
*and* (combined with `video_chunk_size=3` in the cache config) the
resulting latent length must be `1 + 3*m` -- together, valid
non-degenerate values are `1 + 24*k`. **25 (k=1) is the actual minimum**
that produces more than zero real blocks; nothing smaller works at all.
Reverted to `num_frames=25` for the causal engine's warmup. (The base,
non-causal engine's warmup has no such constraint and was left at
`num_frames=2`, untouched.)

## -3.6. CUDA JIT cache tuning (applied, unclear if it actually helped)

Enlarged `CUDA_CACHE_MAXSIZE` (default 1GB -> 4GB) and set
`CUDA_CACHE_PATH` explicitly in `run_gradio.sh`, on the theory that the
model's many distinct kernel shapes were evicting the default-size cache,
forcing every server restart to pay full JIT compilation cost again.
**Later evidence undercut this theory**: warmup time did not improve
across multiple repeated restarts even with the larger cache -- pointing
at (b) above (genuine per-call compute cost from a slow SDPA backend) as
the real bottleneck instead of JIT compilation. The larger cache size is
harmless and still applied, but likely wasn't the fix that mattered.

## -2. Live preview fades/goes to black near the end (mitigated, not fully diagnosed)

Reported: the live preview appears to fade to black by the end of
generation. Not yet isolated to a specific cause -- the reporting user
interrupted (Ctrl+C) runs before reaching completion each time, so it was
never confirmed whether the **final downloaded result** also ends in
black (a real generation artifact, would need investigating the model/
decode path) or whether it's specific to the **live preview** component
(a UI transition quirk).

**Mitigation applied**, addresses one plausible cause regardless of root
cause: `stream_video` didn't have `loop=True`. A short block clip that
finishes playing before the next block's update arrives has nothing
telling the `<video>` element to keep showing content, which could read as
"faded to black" depending on browser behavior at end-of-playback. Added
`loop=True` so it keeps replaying the last block's content instead.

**Still needed to fully diagnose:** let a generation run to completion
(no Ctrl+C) and check whether the *downloaded* final video also ends in
black. If it does, this isn't a UI issue at all -- the model/decode
pipeline is producing genuinely black frames near the clip's end, a
different and more serious problem.

## -1. Visible stutter/gap between block updates in the live preview (fixed via WebSocket + MediaSource)

Even with item 0 below fixed (real incremental blocks now arrive), each
new block caused a visible pause: `stream_video` (`gr.Video`) did a full
`<video src=...>` swap per block, which meant a full HTTP fetch + decode +
rebuffer in the browser every time, not a seamless append.

**First attempt, reverted:** flipped `stream_video` to
`gr.Video(streaming=True)` plus a `fragmented=True` option on
`encode_video()`, assuming `streaming=True` does client-side MSE append of
fragmented MP4 chunks. **Wrong assumption, confirmed by testing:** this
Gradio version's `streaming=True` actually drives an **HLS** player
(`hls.mjs`) that fetches a `playlist.m3u8` -- Gradio expects to run its own
server-side HLS segmentation, not receive already-fragmented MP4 directly.
Result: `HLS error: levelEmptyError`, fatal, video never played. Fully
reverted.

**Second attempt, this is what's in place now:** bypass `gr.Video` and
Gradio's file-serving entirely with a custom WebSocket + MediaSource
Extensions (MSE) player, mirroring `JoyAI-Video-Edit`'s own UI. Standalone
pieces validated first via throwaway scripts (`test_streaming_encode.py`,
`test_ws_stream_server.py`/`test_ws_stream_client.py`, `test_mse_player.html`)
before touching the real app, confirming: PyAV can produce a correctly
incrementally-flushing fragmented MP4 (critical setting:
`stream.gop_size = frames_per_chunk`, forcing a keyframe/fragment boundary
at each chunk -- without it fragments don't flush until `close()`), a
WebSocket can deliver those bytes progressively, and a real browser's
`MediaSource`/`SourceBuffer.appendBuffer()` plays them live.

**Architecture, as built:**
- `ltx_pipelines/utils/media_io.py`: new `StreamingEncoder` class --
  `write_chunk(video_chunk, audio_chunk)` returns only newly-flushed bytes
  since the last call; `close()` flushes and returns the trailer. Supports
  an optional second (AAC) stream for audio (`include_audio=True`), added
  at construction time (not lazily -- fragmented MP4 needs every stream
  declared before the first packet is muxed).
- `gradio_echo_wm.py`: `main()` no longer uses `demo.queue().launch(...)`
  -- it mounts Gradio into a FastAPI app (`gr.mount_gradio_app`) alongside
  a custom `/ws/stream/{run_id}` route, served via `uvicorn.run()` (needed
  since `.launch()` has no hook for adding routes). **`--share` no longer
  works in this mode** (Gradio's tunnel setup is internal to `.launch()`)
  -- prints a note and ignores the flag instead of failing silently.
  `EchoWMCausalEngine.generate()` takes an `on_stream_chunk(bytes | None)`
  callback, driving a per-generation `StreamingEncoder` from inside
  `on_block`; a thread-safe bridge (`_stream_queues` + `_push_stream_chunk`,
  using `loop.call_soon_threadsafe`) relays bytes from the sync worker
  thread to the async WebSocket handler. Frontend: `head=` JS
  (`_mse_stream_js`) polls a hidden trigger textbox for a fresh run id,
  then opens `MediaSource` + the WebSocket and appends chunks as they
  arrive into a raw `<video id="live-preview-video">` (replacing the old
  `stream_video` `gr.Video`).

**Two real bugs hit and fixed while wiring this into the actual app
(neither showed up in the standalone validation, since that used plain
HTML, not Gradio):**

**(a) The hidden trigger textbox never being detected at all.**
`stream_trigger = gr.Textbox(..., visible=False, elem_id="stream-trigger")`
seemed like the obvious way to hide it, but current Gradio's `visible=False`
means the component **is never mounted in the DOM at all** (conditional
`{#if visible}` render), not just CSS-hidden like older Gradio/most other
frameworks. The polling JS's `document.querySelector('#stream-trigger
textarea')` therefore always returned `null`, silently, forever --
`connect()` never ran, and confirmed via the server access log: **zero**
`/ws/stream/...` requests ever arrived. Fixed: `visible=True` (so Gradio
actually renders it) with `#stream-trigger { display: none !important; }`
in the injected `<style>` instead. General lesson for this codebase: don't
use `visible=False` for any component a `head=` script needs to observe.

**(b) `MediaSource.addSourceBuffer()` MIME/codec mismatch once audio was
added.** The codecs string (`'video/mp4; codecs="avc1.640028"'`) must
declare *exactly* the track set present in the actual byte stream, or every
`appendBuffer()` call fails, not just the `addSourceBuffer()` call itself.
Since whether a given generation has audio depends on the `gen_audio`
checkbox (runtime, not know-at-page-load), the trigger textbox's value is
now `"<run_id>|<1-or-0>"` (not just the bare run id) so the frontend knows
upfront whether to declare `mp4a.40.2` (AAC-LC) alongside `avc1.640028` --
`connect(runId, hasAudio)`.

**Autoplay-with-audio caveat (inherent to browsers, not fixable server-side):**
browsers block autoplay *with sound* unless the site already has "high
media engagement" or `play()` follows a real user gesture -- a live-updating
background preview can't manufacture that. `_mse_stream_js` tries unmuted
`play()` first; on failure it falls back to muted (so the picture still
plays) and a click on the video unmutes + resumes audio. If a generation
has no audio track at all (`gen_audio` off), the video is muted upfront
instead of attempting (and always failing) an unmuted play.

**Not yet confirmed working end-to-end on real hardware after the audio
addition** -- video-only was confirmed working (clean `appendBuffer`,
`loadeddata` fired, `endOfStream()` succeeded with no errors) on
`10.74.11.118` before audio support was added; audio itself hasn't been
tested yet as of this writing.

## 0.5. Preview decode is redundant/slow -- partially mitigated

`decode_video()`/`decode_audio()` (`ltx_core/model/video_vae`) have no
incremental/cached-state API -- every `on_block` call re-decodes the
*entire* accumulated prefix from scratch (not just the new slice), so cost
grows as generation progresses; a block near the end re-decodes almost
everything shown by earlier blocks all over again. This was already
happening even before item 0's fix (the decode always ran; only the
*output* was discarded for "zero frame" blocks).

**Mitigation applied** (`causal_ti2vid.py`, `raw_on_block`): added
`PREVIEW_DECODE_STRIDE = 3` -- only actually decode+preview every 3rd
block plus always the last one; other blocks report a zero-frame chunk
(cheap, no decode work) via the existing zero-frame skip path. Trades
preview granularity (fewer visible updates -> contributes to the stutter
in item -1) for real speedup (fewer expensive re-decodes).

**Not implemented, more correct fix:** since the decoder is causal (only
looks backward), it likely doesn't need the *entire* prefix from frame 0
to correctly decode the recent tail -- just some bounded lookback window.
Slicing the decode input to `[video_start - window : video_end]` instead
of `[0 : video_end]` would keep per-block decode cost roughly constant
instead of growing, allowing every block to decode+show (fixing both the
speed problem and the item -1 stutter's stride-induced gaps at once)
without a stride at all. Not attempted because the decoder's actual
required receptive field/lookback size hasn't been confirmed -- guessing
wrong risks visibly corrupted preview frames (not a crash, so it could go
unnoticed).

## 0. "Live preview only ever shows one block, then nothing" (fixed -- root cause confirmed)

**Symptom:** with `num_frames=241`, block 0's `on_block` callback carried
all 241 frames; every callback after that carried `shape=(0, H, W, C)`
(zero frames), correctly skipped by the zero-frame guard (see item 1(b))
but meaning the preview never advanced past block 0. Initially suspected
this was just because a short clip fits in "one decode window" -- **wrong**.
Confirmed wrong by testing a much longer clip (`num_frames=961`): still
only block 0 ever had content, and the whole run **OOM'd**
(`torch.OutOfMemoryError: ... Tried to allocate 177.37 GiB`) trying to
decode it.

**Actual root cause**, in `causal_ti2vid.py`'s `raw_on_block`: `rollout.py`
allocates `buffers.video_output` **once**, sized for the entire clip, and
each block writes into its own slice of it
(`buffers.video_output[:, video_start*ppf:video_end*ppf] = video_sample`)
-- the buffer's total length never grows. `raw_on_block` passed this
*entire* buffer (including the still-zero, not-yet-denoised tail) to
`vae_decode_video` on every single call. The VAE doesn't truncate its own
output just because the tail is zero -- it decodes the whole thing and
returns full-clip-length pixel output every time. So:
- The `seen["video_frames"]` diffing logic (`decoded_video_so_far[seen:]`)
  assumed the decoded length *grows* each call. It never did after the
  first call, since the decoder always returns the same total length --
  hence zero "new" frames on every callback after block 0.
- Decoding the *entire* buffer (not just what's been denoised so far) on
  every one of ~40 callbacks is why a 961-frame run OOM'd: decode cost
  scales with total clip length, not progress-so-far, and compounds across
  every callback.

**Fix** (`causal_ti2vid.py`): use the `video_block` tuple already passed
to the callback (previously received and discarded as `_video_block`) --
its `video_end` (in latent frames) tells you exactly how far this block's
denoising has actually progressed. Convert to a pixel-frame count with the
same `(latent_frames - 1) * 8 + 1` formula the pipeline itself uses for
`num_frames`, and truncate the decoded output to that range *before*
diffing against what's already been shown:
```python
_, video_end = video_block
pixel_end = (video_end - 1) * 8 + 1
decoded_so_far_valid = decoded_video_so_far[:pixel_end]
video_chunk = decoded_so_far_valid[seen["video_frames"]:].clone()
seen["video_frames"] = decoded_so_far_valid.shape[0]
```
This fixes the "preview never advances" bug (later blocks now report real
incremental content). It does **not** fix the OOM-at-long-clips issue --
the full buffer is still decoded every call, just no longer *shown* past
the valid range. A full fix for that would need to slice the *input*
latent (and matching `positions`/`denoise_mask`/`clean_latent` fields) to
`video_end` before decoding at all, not just the output -- not attempted
here due to the risk of getting those coupled tensor shapes wrong in code
shared with the real (non-preview) generation path.

Separately, added a heartbeat so the UI doesn't look frozen during the
internal denoising time between callbacks: `EchoWMCausalEngine.generate()`'s
`result_queue.get()` (previously an unbounded blocking wait) now uses a 2s
timeout and yields `("heartbeat", elapsed_seconds)` on each empty wait;
`on_generate` shows this in the Status textbox without touching the video
outputs. Note: this heartbeat's *text* was observed to stop rendering in
the browser after the first tick even though the backend kept incrementing
correctly (confirmed via server-side `nvidia-smi` + log timestamps) --
a separate, still-unresolved Gradio frontend delivery issue, low priority
since it doesn't affect the final result.

## 1. Block preview videos 403 (`File not allowed: .../blocks/block_NNN.mp4`)

Two distinct, easily-conflated problems, both hit on `pmgb300ws-0304`:

**(a) Gradio's own `allowed_paths` check rejecting genuinely-safe files (fixed).**
Gradio's `/gradio_api/file=<path>` endpoint gates every file read through
`gradio.utils.is_allowed_file(path, blocked_paths, allowed_paths, created_paths)`,
which returns `(allowed: bool, reason: str)`. Files under `OUTPUT_ROOT` were
correctly listed in `allowed_paths` (passed to `.launch()`), yet the check
still returned `False` for genuinely-safe paths -- never definitively root
caused (suspected symlink/realpath normalization mismatch specific to this
box), and not worth continuing to chase since the app already monkeypatches
this function for debugging.

Fixed by making the monkeypatch (`_trusted_is_allowed_file`, top of
`gradio_echo_wm.py`) authoritative instead of Gradio's own logic: it does
its own `Path(path).resolve()` containment check against `OUTPUT_ROOT`/
`EXAMPLES_DIR` and returns `(True, "trusted output/example root")`
immediately on a match, before ever consulting Gradio's original (buggy)
logic.

Gotcha hit while fixing this: this installed Gradio version's
`is_allowed_file` returns a **tuple** `(allowed, reason)`, not a bare
`bool` -- `gradio/routes.py`'s `file()` endpoint does
`allowed, reason = utils.is_allowed_file(...)`, so an override that
`return True` instead of `return True, "..."` crashes with
`TypeError: cannot unpack non-iterable bool object` on every file request.

**A full-URL approach (bypassing local-path serving entirely) is a dead
end** for this deployment: `gr.Video`/`gr.File`'s postprocess step
downloads `http(s)://` values **server-side** via `safehttpx`, which has an
SSRF guard that hard-rejects any private/loopback IP with no bypass -- and
this box is only reachable via a private LAN IP (`10.74.11.x`). Local
filesystem paths through `is_allowed_file` are the only viable serving
path here.

**(b) Some blocks legitimately carry zero video frames -- was being
encoded/shown anyway, causing a spurious 403 (fixed).**
Once (a) was fixed, block 0 of every generation (and the warmup run)
consistently produced a real file -- but every subsequent block's file was
missing from disk entirely (confirmed via `ls -la`, not just Python's
`exists()`), even though `on_block` completed with no exception. Root
cause, confirmed via a `video_chunk.shape` debug print: some `on_block`
calls carry a **zero-frame** tensor, e.g. `shape=(0, 512, 896, 3)`.

Traced to `ltx-pipelines/src/ltx_pipelines/causal_ti2vid.py`'s
`raw_on_block` (~line 141-189): each callback only emits the *newly
decoded tail* since the last callback --
`video_chunk = decoded_video_so_far[seen["video_frames"]:].clone()`. The
callback fires once per internal denoising/rollout step, but the causal
(backward-looking-only) video decoder doesn't necessarily have a new chunk
ready at every single step -- e.g. a step might only advance the audio
decode. When that happens the slice is empty. This is expected pipeline
behavior, not a bug -- PyAV's MP4 muxer just silently writes nothing for
zero frames (no exception), and the previous code was encoding/queuing a
UI update for these empty blocks anyway, pointing the live preview at a
file that was never written.

Fixed in `on_block` (`gradio_echo_wm.py`): skip encoding and skip queuing
a UI update entirely when `video_chunk.shape[0] == 0` -- nothing to encode,
nothing to show for that callback.

Separately noted, not a bug: the live block-by-block **preview** can show
visible noise/artifacts on the first block or two even when the final
result is clean -- the preview uses a separate incremental/causal decoder
(`preview_video_decoder` in `causal_ti2vid.py`) that decodes each chunk as
soon as it's denoised, with less accumulated temporal context than the
final decode pass gets. If it settles down after a block or two and the
final downloaded video is clean, this is expected, not something to chase.

## 2. No startup warmup (fixed)

`gradio_echo_wm.py` had no warmup at startup -- the first real user
request paid the full cost of lazily loading ~47GB of weights plus any
first-call CUDA kernel compilation. Fixed via `_warmup()` in
`gradio_echo_wm.py`, which runs one small (256x128, 25-frame) throwaway
generation using the first available example case's image before the
server announces it's ready. Disable with `--no-warmup`.

## 3. Debugging notes

- The app is normally launched via `run_ui.sh` (auto-detects
  checkpoint/engine, activates `echo_wm/.venv`) -> `run_gradio.sh`. As of
  this session, `run_gradio.sh` always tees its full output to
  `gradio_debug.log` in the `echo_wm/` directory -- no need to remember to
  redirect manually. `on_block` prints one `[block] wrote block_index=N/M
  path=... frames=K` line per real (non-empty) block written; zero-frame
  blocks print nothing since there's nothing written for them.
- Watch for **wrong venv active**: `JoyAI-Echo/.venv` (top-level, Python
  3.12) vs `JoyAI-Echo/echo_wm/.venv` (Python 3.11, the one this app
  actually needs) are two different venvs. A `torchaudio` native-library
  load failure (`OSError: Could not load this library: .../libtorchaudio.so`)
  in a traceback pointing at the *wrong* venv path is the tell -- check
  `which python` / `python --version` (expect `echo_wm/.venv/bin/python`,
  3.11.x) before debugging further.
- `run_causal_0001` (and other `run_causal_NNNN` directories) get reused
  across server restarts, since `run_counter` resets to 0 each time the
  process restarts. Don't assume every file in a `run_causal_0001/`
  directory came from the same run -- check timestamps.
