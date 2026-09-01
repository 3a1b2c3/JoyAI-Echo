# Troubleshooting: `gradio_echo_wm.py` (Flash Preview / streaming UI)

## -1.1. `ECHO_WM_GRAPH_CAPTURE_LAYERS=1`: CONFIRMED BROKEN on real hardware -- cache position bookkeeping frozen at capture time, do not use

Ran end-to-end on `pmgb300ws-0304`. Speed numbers looked spectacular
(`denoise` 1.05-1.07s -> 0.12s, `cache` 0.55s -> 0.09s, block total
~1.8s -> ~0.41s, i.e. *faster than real-time*) -- and that was the first red
flag, not a win: every replayed block's timing was suspiciously
bit-identical (0.120-0.121s denoise, 0.089-0.090s cache across 9 different
blocks -- real varying computation doesn't do that). User confirmed: no
real video output.

**Root cause, now understood precisely**: `torch.cuda.graph()` capture only
records GPU kernel launches -- it does NOT replay plain Python statements.
`update_kv_cache` (attention.py) mutates the cache dict's Python-side
bookkeeping as ordinary assignments: `cache["_prev_start"] = start`,
`cache["_prev_chunk"] = chunk`, `cache["length"] = active`. Those execute
**once**, during the capture trace (warmup + one capture call, both using
block 0's `start=0`), and are then **frozen** -- every subsequent
`graph.replay()` reruns the same captured GPU ops with new `vx`/`ax`
tensor *values* copied in, but `kv_cache_start` itself (also a plain
Python int, baked into the closure) stays `0` on every replay, and
`cache["_prev_start"]` never updates past its frozen value either. Result:
`is_repeat = (start == cache["_prev_start"])` evaluates `True` on every
single replay (baked `0` == frozen `0`), so `_update_kv_cache_fixed_chunk`
takes the "same-range redo" branch every time -- overwriting the exact same
cache slot with each new block's K/V instead of advancing the sink+FIFO
window forward. The model never sees a real, advancing generation
sequence; it re-processes quasi-block-0 forever. That explains the
identical-every-block timing (same captured op sequence, no real growth),
the impossible speedup (skipping the real incremental work), and "no
output" (nothing coherent to decode).

**This reveals a real gap in item -1.0's scoping, not just a wiring bug.**
Item -1.0 correctly confirmed self-attn's RoPE math (`_slice_rope(local_pe,
active - q_len, active)`) doesn't depend on captured Python-side state --
`active` is shape-derived, constant in steady state. But it wrongly
extended that safety to `update_kv_cache`'s cache-*position* bookkeeping,
which is a different piece of state with the same problem class item -0.7
already found for a2v/v2a's `query_slice` dict lookup -- just one level
deeper (inside the self-attn phase's own cache write, not the cross-modal
branch).

**Why `test_graph_capture_per_layer_split.py` said CONFIRMED FEASIBLE
without catching this**: that test only exercised ONE capture-then-replay
against a FRESH cache for the "block 1" comparison (both the split-capture
path and its eager reference started from an empty cache), so `active`
happened to equal `CHUNK` in both cases regardless of `start`'s value --
the test's RoPE-output diff passed correctly, but it never tested a
*sequence* of many replays against ONE persistently-advancing cache, which
is the real production pattern and exactly where the frozen-bookkeeping bug
lives. The test wasn't wrong about what it checked; it checked too narrow a
scenario.

**Second, more severe symptom, same broken integration**: a later run hit a
hard crash instead of silent corruption -- `torch.AcceleratorError: CUDA
error: an illegal memory access was encountered`, raised at
`LayerPhaseGraphCapture.run`'s `torch.cuda.synchronize()` call, inside
`_generate_audio_prefix`'s call to `forward()` (rollout.py) -- a different
call site than the main per-block loop, likely a different `vx`/`ax` shape
or a `video=None` pattern for audio-prefix generation. Suspected
contributing cause, on top of the frozen-bookkeeping bug above:
`LayerPhaseGraphCapture` creates a separate `torch.cuda.CUDAGraph()` per
`(region_key, shape)` with no shared memory pool
(`torch.cuda.graph_pool_handle()` was never used) -- multiple independently
-captured graphs can conflict over CUDA memory addresses without one, a
known real gotcha with `torch.cuda.graph()`. Combined with garbage cache
indices from the bookkeeping bug, this plausibly explains an out-of-bounds
access. Not confirmed -- no GPU access to reproduce or root-cause further.
An illegal memory access can poison the whole CUDA context (same class of
danger noted in TROUBLESHOOTING.md's other CUDA-graph items) -- restart the
server process after hitting this, don't assume subsequent runs on the same
process are unaffected even with the flag off.

**Fourth bug found while fixing the third**: the "get `flags`" call at the
top of the graph-capture branch (`model.py`) called the REAL
`_forward_self_attn_phase` just to read `run_vx`/`run_ax`/etc. -- itself a
real, mutating call against the live cache, executed BEFORE
`LayerPhaseGraphCapture.run()`'s warmup/capture even started. It would have
consumed the real advance transition on its own, independent of bug three.

**All four bugs now have real fixes implemented (NOT yet verified on real
hardware)**:
1. `_self_attn_caches_at_capacity()` (`model.py`) gates the graph-capture
   branch on the cache already being full -- blocks during the fill phase
   fall through to the plain eager `block(...)` call.
2. `LayerPhaseGraphCapture.__init__` now creates one shared
   `torch.cuda.graph_pool_handle()`, passed to every `torch.cuda.graph(...,
   pool=...)` call -- addresses the suspected illegal-memory-access cause.
3. `LayerPhaseGraphCapture.run()`/`_capture()` gained a `warmup_phase_fn`
   parameter: warmup now runs against a caller-supplied SCRATCH closure
   (built from `_clone_layer_cache_for_warmup()` in `model.py`, deep-
   copying `video_self`/`audio_self`/`video_ucpe`'s k/v/positions/length),
   while the real capture-trace call is the only one touching the real
   cache -- fixes the "warmup consumes the real transition" bug.
4. `compute_layer_phase_flags()` (`transformer.py`) extracted as a pure,
   side-effect-free function -- `_forward_self_attn_phase` now delegates to
   it (confirmed via `git diff`: identical logic, just relocated) instead
   of duplicating it, and `model.py`'s flags lookup calls this instead of
   the real (mutating) phase method.

**Status: `ECHO_WM_GRAPH_CAPTURE_LAYERS` still must not be trusted without
a fresh real-hardware test.** All four found bugs have reasoned, code-
reviewed fixes now in place (git diffs checked for correctness), but NONE
of this has run on a GPU yet -- every fix so far in this saga has revealed
a further bug on the next real run, so treat this as "ready for another
careful test," not "fixed." Before trusting it: run the same silent-
corruption check (compare real generated video against a flag-off run,
same seed/prompt/image) AND the same crash check
(`_generate_audio_prefix`'s call path, the one that previously hit the
illegal-memory-access error) -- both, not just one. If either fails again,
that's a fifth bug, not a reason to revert the first four fixes blindly. If
this route keeps failing, flashdreams' actual `BlockKVCache` (item -21) --
which avoids this whole bug class by using `torch.sym_min`/`sym_max`-style
symbolic bounds and an explicit before_update/update/after_update lifecycle
instead of ad-hoc Python dict mutation -- remains the structurally sound
fallback, at the cost of the full 1-2+ week integration already scoped
there.

## -1.0. Item -0.7's graph-capture blocker: scoped concretely; reordering ruled out, only per-layer split is valid; feasibility CONFIRMED (narrower than first thought -- see item -1.1), real integration wired but BROKEN

Follow-up scoping on item -0.7's cross-modal-RoPE graph-capture bug, done
before attempting any fix.

**Denoise and cache are both dispatch-bound on the same `forward()` code
path -- but the "N-steps-of-forward()" explanation below was WRONG,
corrected after actually running `ECHO_WM_PROFILE_DENOISE`.** Originally
assumed (from the ~1.9x timing ratio matching the 2-step preset) that
`_denoise_av_block` calls `forward()` twice per block while
`ECHO_WM_PROFILE_CACHE` profiles one call. The real profiler data
contradicts that: `ECHO_WM_PROFILE_DENOISE`'s trace shows `aten::linear` at
exactly **1756** calls -- identical to cache's single-`forward()` count,
not doubled. Totals: `Self CPU time total: 795.286ms`, `Self CUDA time
total: 81.964ms` (~9.7x ratio, same dispatch-bound pattern as cache).

Initially suspected denoise vs. cache route through different attention
backends (denoise's trace shows `aten::scaled_dot_product_attention` /
cudnn kernels; the earlier cache profile showed `FlashAttnFunc`/cutlass
kernels instead) -- **retracted**: those two profiles were captured in
different sessions under different `ECHO_WM_FLASH_ATTENTION_4` settings
(cache profile: FA4 enabled; this denoise profile: captured after
switching to SDPA-only per user request). Not a real per-call discrepancy
within one run -- just an artifact of comparing profiles taken under
different flags. Not a lead worth pursuing.

Both are still dispatch-bound in the sense that matters for item -1.0/-1.1's
graph-capture work: fixing that blocker would still speed up both denoise
and cache, since they both go through the same `forward()`/layer-loop code
path -- just not via the specific "2x calls" mechanism originally claimed.
Combined they're still the large majority of block time (~1.86s of ~2.2s
in the run that produced this profile, hardware/load-dependent -- see the
run's own `[rollout] block` lines), not just cache's ~30% alone.

**The existing `ECHO_WM_GRAPH_CAPTURE_TEST=1` test** (`rollout.py` lines
434-468) wraps the *entire* multi-layer `forward()` call in one
`torch.cuda.graph()` region -- confirmed by reading it, not assumed. That
whole-model capture is what item -0.7 found fails.

**Fix (a) (data-driven `query_slice`) is bigger than it looked.**
`kv_cache_start` (the Python int whose dict lookup gets baked as a graph
constant, see item -0.7) is threaded through the entire KV-cache system as
a plain int, not just this one call site. Making the lookup tensor-driven
means making `kv_cache_start` itself graph-traceable everywhere -- closer
in scope to the full `BlockKVCache` integration (item -21) than a
standalone fix.

**Fix (b) (exclude a2v/v2a from the captured region) requires a per-layer
split, confirmed necessary by reading `transformer.py`'s actual layer
loop**: each layer runs self-attention (video_self ~line 401, audio_self
~line 447) *before* its a2v/v2a cross-modal block (~line 470), and that
cross-modal block reads `vx`/`ax` -- the *same* layer's post-self-attention
hidden states, not the previous layer's. Those updated `vx`/`ax` become the
residual-stream input to the *next* layer's self-attention. So the real
computation is strictly interleaved per layer: self-attn(N) ->
cross-modal(N) -> self-attn(N+1) -> ...

**This rules out "capture all self-attn across all layers, then all
cross-modal after"** -- it would silently compute something different from
the real model (a correctness bug, not just a missed optimization), not a
viable reordering. The only architecturally valid version of fix (b) is a
**per-layer split**: capture self-attn+cache-update, break to eager Python
for that layer's a2v/v2a, re-enter capture for the next layer's self-attn --
repeated once per layer. This reintroduces real dispatch/launch overhead at
every layer boundary, which is exactly the cost graph capture exists to
remove.

**Revised speedup expectation: likely well under item -21's ~80-90%
estimate.** That estimate assumed one clean whole-`forward()` capture. Per-
layer boundaries eat into it -- rough unmeasured guess, capturing maybe
~60% of the ops (self-attn/cache-update; a2v/v2a stays eager) minus
re-entry overhead at each boundary. No real number exists until this is
implemented and measured.

**Feasibility test written, not yet run**: `test_graph_capture_per_layer_split.py`
(repo root) -- simulates a 2-"layer" stack (self-attn capture -> real eager
cross-modal -> self-attn capture again), using the real
`update_kv_cache`/`apply_rotary_emb`/`_slice_rope` from `attention.py`,
captured at one block and replayed at a different block, diffed against a
fully eager reference. Tests the specific open risk this fix introduces:
whether two independent `torch.cuda.graph()` regions with real eager work
interleaved between them (re-entering capture, memory pool reuse) stay
correct -- a different risk from item -0.7's single-region bug, and not
covered by the existing `ECHO_WM_GRAPH_CAPTURE_TEST`. Needs a real GPU
(`torch.cuda.graph()` capture doesn't work on CPU) -- run on the GB300 box:
```bash
python test_graph_capture_per_layer_split.py
```
If it reports CONFIRMED FEASIBLE, real integration into `transformer.py`'s
layer loop is the next step. If BLOCKED, root-cause before attempting
integration -- do not proceed on the assumption it'll probably work.

**Result: CONFIRMED FEASIBLE**, run on `pmgb300ws-0304`. All three checks
matched the eager reference for block 1 (captured at block 0): region A
(layer-1 self-attn, captured) MATCH, cross-modal (eager, run for real
between the two capture regions) MATCH, region B (layer-2 self-attn,
captured after eager work ran on the other cache in between) MATCH. Two
independent `torch.cuda.graph()` regions with real eager Python work
interleaved between them, replayed for a different block than captured,
produced correct results -- the interleaved-capture-regions risk this test
targeted is not a blocker.

**Real integration DONE (not yet benchmarked, not yet run end-to-end)**:
1. `transformer.py`'s `BasicAVTransformerBlock.forward()` split into three
   phase methods (`_forward_self_attn_phase` / `_forward_cross_modal_phase`
   / `_forward_mlp_phase`), called in sequence by a thin `forward()`
   wrapper -- confirmed via `git diff` to be a pure mechanical extraction,
   zero logic changes, so every existing (non-graph-capture) caller is
   unaffected.
2. `ltx_causal/layer_graph_capture.py`'s `LayerPhaseGraphCapture` -- generic
   per-`(region_key, shape)` capture/replay manager for `vx`/`ax`.
3. Wired into `model.py`'s `LTXModel._process_transformer_blocks` loop: a
   new `elif self._layer_graph_capture is not None` branch runs
   `_forward_self_attn_phase` and `_forward_mlp_phase` through
   `LayerPhaseGraphCapture` (keyed `f"L{layer_index}_self_attn"` /
   `f"L{layer_index}_mlp"`), `_forward_cross_modal_phase` eagerly in
   between, exactly matching the per-layer-split design confirmed feasible
   above.
4. Gated behind `ECHO_WM_GRAPH_CAPTURE_LAYERS=1` in `CausalTI2VidPipeline.__call__`
   (`causal_ti2vid.py`), default off, applied once per newly-built
   `x0_model` (same place/pattern as `ECHO_WM_COMPILE`).

**NOT yet run on real hardware end-to-end.** Only the isolated 2-toy-layer
feasibility test has actually executed on a GPU -- the real `model.py`
integration (N real layers, real `TransformerArgs`, real `kv_cache` dicts)
has not. Test with:
```bash
ECHO_WM_GRAPH_CAPTURE_LAYERS=1 bash run_gradio.sh 2>&1 | tee /tmp/graph_capture_test.log
```
**Watch closely for wrong/corrupted output, not just crashes/exceptions** --
per item -0.7, the dangerous failure mode here is silently wrong generation,
not a loud error. Compare output against a plain (flag-off) run of the same
seed/prompt/image before trusting this. If output diverges or generation
looks wrong, that's a real correctness bug in this integration, not
something to dismiss as "probably fine." Once confirmed correct, the
`[rollout] block ...` timing line gives the real (not estimated) speedup
number against the ~1.76-1.8s/block baseline.

## -0.9. `ECHO_WM_PROFILE_CACHE=1` re-run: reconfirms item -21's dispatch-bound finding; next steps

Re-ran `ECHO_WM_PROFILE_CACHE=1` (see item -21 for the original finding) to
check whether it still holds on current code. It does -- the op-count
signature matches almost call-for-call: `aten::linear` x1756, `aten::mul`
x3886, `aten::add` x3802, `aten::copy_` ~4259, same top offenders, same
shape (many small kernels, no single dominant Self CUDA% -- `aten::addmm`
tops out around 24%, most rows single digits). Still dispatch-bound, not
compute-bound, on this exact codebase as of this re-check.

**Totals confirmed**: `Self CPU time total: 777.676ms`, `Self CUDA time
total: 73.392ms` (~10.6x CPU:CUDA ratio) -- same order of magnitude as item
-21's original 828ms/86ms (~9.6x). Still solidly dispatch-bound.

**Next steps, in order of effort** (none started):
1. ~~Confirm the totals line~~ -- done, see above.
2. **Scope-only, 1-3 days**: cherry-pick just
   `KVCacheRelativeRotaryPositionEmbedding3D` (`flashdreams/core/attention/rope.py`)
   -- precomputed, device-resident RoPE frequencies, no per-call CPU-GPU
   copy. Smallest real piece of item -21's proposal; doesn't by itself fix
   the dispatch-count problem (still one Python op per tensor), but
   removes one class of overhead cleanly and independently.
3. **Full integration, 1-2+ weeks**: `BlockKVCache` + its paired
   `CUDAGraphWrapper` from `flashdreams/core/attention/kvcache.py`, which is
   what actually collapses the ~4000+ individually-dispatched small ops
   into one graph replay -- the only lever expected to meaningfully move
   the 828ms CPU dispatch number. Blocked on item -0.7 (full-block graph
   capture silently corrupts cross-modal RoPE) being solved first, via
   either of its two proposed fixes (data-driven `query_slice` instead of a
   dict lookup, or excluding the a2v/v2a branch from the captured region).
   Estimated (theoretical, unmeasured) payoff: up to ~80-90% off
   cache-update's ~0.53s/block cost, which would land total block time
   around ~1.3-1.35s -- under the 1.5s real-time budget, not just closer to
   it.

## -0.8. `ECHO_WM_VAE_CHANNELS_LAST=1`: opt-in channels_last_3d VAE decoder (implemented, BENCHMARKED: net negative, default off)

`ECHO_WM_PROFILE_CALLBACK=1` traces of the `on_block` streaming-preview
callback (VAE-decodes the accumulated video/audio buffers so far) showed
the decoder's conv stack dominating cost as expected (`conv3d`,
`conv_transpose1d`, `conv1d`, depthwise conv -- ~96ms CUDA total across 685
`aten::convolution` calls in one sample trace), but also a
`nchwToNhwcKernel` layout-conversion kernel firing 495 times (~8.6ms) --
evidence the decoder isn't running in a consistent memory format and is
paying a per-call NCHW->NHWC conversion inside the conv path.

Added `ModelLedger.video_decoder()`
(`ltx-pipelines/src/ltx_pipelines/utils/model_ledger.py`): behind
`ECHO_WM_VAE_CHANNELS_LAST=1`, converts the built decoder to
`torch.channels_last_3d` once at load time instead of per-call. Opt-in,
default off -- whether this actually helps depends on whether cuDNN has a
faster `channels_last_3d` conv3d kernel for this exact shape/dtype on this
GPU, which is only knowable by re-running the same profiler with the flag
on and diffing the trace:

```bash
ECHO_WM_PROFILE_CALLBACK=1 ECHO_WM_VAE_CHANNELS_LAST=1 bash run_gradio.sh 2>&1 | tee /tmp/profile_callback_cl.log
grep -A 30 '\[profile\] on_block' /tmp/profile_callback_cl.log
```

**Benchmarked on real hardware: net negative.** `ECHO_WM_VAE_CHANNELS_LAST=1`
run measured `callback` at ~0.26-0.28s steady-state, vs. the ~0.2s baseline
-- worse, not better, roughly +30-40%. Likely `channels_last_3d` isn't
actually a faster cuDNN conv3d path for this decoder's specific
shape/dtype/GPU combination, so the one-time format conversion just adds
overhead with no runtime payoff -- same verdict pattern as item -9's FP8
result. Leave `ECHO_WM_VAE_CHANNELS_LAST=0`. (Note: this benchmark run
produced 6 blocks, not the usual 10 -- same run-type ambiguity noted
elsewhere in this doc; the relative callback comparison should still hold
regardless, since it's a within-run steady-state number.)

Compare the `nchwToNhwcKernel` line and overall CUDA total against a
baseline trace (flag off). Not yet run on real hardware as of this entry --
mark this RESOLVED or REVERTED once that comparison happens either way.

## -0.7. CONFIRMED: full-block CUDA graph capture would silently corrupt cross-modal RoPE (empirically proven, not yet fixed)

Item -23's proposed full block-forward graph integration (never started)
was paused over a traced-but-unproven concern: the a2v/v2a cross-modal
attention branch in `Attention.forward()` (`attention.py`) does
```python
query_slice = kv_cache["local_cross_q_slices"].get((kv_cache_start, kv_cache_start + new_keys))
q = apply_rotary_emb(q, _slice_rope(local_q_pe, *query_slice), self.rope_type)
```
`kv_cache_start` is a plain Python int, different every block -- under
`torch.cuda.graph()` capture, this dict lookup runs once at capture time
and its result is baked into the recorded op sequence.

**Empirically confirmed with a standalone real-GPU repro**
(`test_graph_capture_cross_modal_rope.py`, isolates exactly this branch
with tiny synthetic tensors, importing the real `apply_rotary_emb`/
`_slice_rope`/`update_kv_cache` from `attention.py`): captured this
branch at block A's `kv_cache_start`, replayed the same graph with block
B's real data swapped into the static input buffers (but `kv_cache_start`
still baked as A, since that can't change post-capture) -- output
**MISMATCHED** block B's honest eager-run reference by a wide margin
(not a rounding difference: `[-0.296, 2.676, -0.141, -0.844]` replayed vs.
`[-2.143, 1.630, 0.514, -0.684]` expected). This is silent, not a crash --
exactly the dangerous failure mode: plausible-looking but wrong output,
which is why it was worth a dedicated correctness test instead of trusting
a guess before ever touching real generation.

**Conclusion: any future full-block-forward graph capture must NOT
naively include this branch as-is.** Two paths forward, neither attempted:
(a) compute `query_slice` from a tensor value (e.g. `torch.searchsorted`
against a precomputed boundary tensor) instead of a Python dict lookup, so
it's traced as data rather than baked as a constant, or (b) exclude the
a2v/v2a cross-modal path from the captured region entirely and run it
eagerly alongside the captured graph each block. Full block-forward graph
integration (item -23's plan) remains not started, and now has a known,
scoped blocker instead of an unverified guess.

## -0.6. SageAttention: confirmed working on this GPU (contradicts earlier research), but confirmed SLOWER than SDPA -- disabled by default

Earlier this session (see -16 below), research suggested no sm_100/103
kernel existed for SageAttention on this architecture (GitHub issue
#237). **Wrong, or since fixed upstream**: `pip install sageattention`
(1.0.6) on the real GB300 box, then a standalone real-kernel-call test
(`test_sageattention_backend.py`) succeeded cleanly -- correct output
shape/dtype, no exception. Wired into `AttentionFunction.DEFAULT` in
`attention.py` as a new `SageAttention` class, tried before xformers.

**Confirmed on a real generation run: SLOWER than SDPA, not faster.**
Steady-state block total went from the ~1.74-1.78s SDPA baseline to
~1.84-1.88s with SageAttention active (confirmed active via the *absence*
of both the SDPA-fallback print and a SageAttention-failure print --
neither fired, meaning it ran successfully for the whole generation).
`denoise` alone rose from ~1.00-1.05s to ~1.08-1.10s, consistently across
every block -- not noise. Same lesson as FlashInfer (item -16): a fast
attention library actually running correctly on this GPU is not the same
as it being faster for this specific workload/shape. **Disabled by
default** (`ECHO_WM_SAGEATTENTION=1` to re-enable for further testing),
mirroring FlashInfer's `ECHO_WM_FLASHINFER` gate exactly. Doesn't change
item -16's overall conclusion: SDPA with explicit backend priority
remains the practical ceiling on this hardware.

## -0.5. Real-time target was overstated by 8x for most of this session (fixed)

Every "Nx too slow" figure discussed/printed earlier in this session used
`video_chunk_size / fps` as the real-time target -- treating
`video_chunk_size` (a *latent*-frame count, per `cache.py`'s own
docstring: "Video cache sizes in latent-frame units") as if it were
decoded output frames. The VAE has an **8x temporal compression**
(`VIDEO_SCALE_FACTORS.time`, `ltx_core/types.py`), so a
`video_chunk_size=3` block actually decodes to **24 output frames**, not
3. Real target at `fps=16` is **24/16 = 1.5s/block**, not `3/16 = 0.188s`.

Actual measured steady-state: ~1.74-1.87s/block. **That's ~1.15-1.25x
too slow, not ~10x.** The "not happening tonight, needs a smaller/distilled
model or multi-GPU parallelism" conclusion reached earlier in this session
was wrong, built on this bug in the debug print added at the same time
(`rollout.py`'s `[rollout] block ...` line) -- real-time is actually
within a plausible night's-work margin, not a different-model-scale
problem. Fixed in `rollout.py`: `target_s` now multiplies
`video_chunk_size` by `VIDEO_SCALE_FACTORS.time` before dividing by fps.

## 0. Recurring gotcha: "the fix didn't work" is very often just a stale sync, not a real failure

Hit repeatedly this session -- costly enough in wasted round-trips to call
out explicitly at the top of this doc. Symptom: a fix is applied and
committed, restart the server, test again, **same exact error/behavior as
before, often at the same line number**. Before concluding the fix is
wrong, check sync state first:

```bash
# On the box actually running the test:
git status                              # anything uncommitted/behind?
git log --oneline -3 -- <path/to/file>  # does the latest commit hash match what you expect?
git pull                                # if behind, pull, then restart the server fresh
```

Two concrete tells that it's a sync issue, not a real failure:
- The traceback/error references the **exact same line number** as the
  pre-fix version, even though the fix added/removed lines above that
  point (a real fix would have shifted the line numbers).
- `git log` on the specific file, checked on the *editing* side, shows
  the fix is already committed with clean `git status` -- meaning the gap
  is purely "hasn't been pulled on the test box yet," not "the fix is
  wrong."

This happened at least half a dozen times this session (`rope.py`'s
freq-grid fix, `attention.py`'s FlashAttention wiring, the diagnostic
sync-removal in `rollout.py`, others) -- each time costing a full
restart-and-retest round-trip before the real cause (stale pull) was
found. **Always verify sync state before re-investigating a "fix didn't
work" result.**

## -24. xformers source build: package installs but compiled CUDA extension missing (in progress)

Following up on item -3/-16 (prebuilt xformers has no kernel for compute
capability 10.3): attempted a from-source build tonight since GB300/sm_103
is new enough that prebuilt wheels may not cover it yet, but a from-source
build targeting the right arch might. First attempt (`pip install
git+https://github.com/facebookresearch/xformers.git`, `TORCH_CUDA_ARCH_LIST`
unset at first) **appeared to succeed** (`Successfully installed
xformers-0.0.35+029779d.d20260901`) but `from xformers.ops import
memory_efficient_attention` fails with `ImportError: cannot import name
'memory_efficient_attention'` -- `xformers.ops` silently omits it when the
compiled `xformers._C` extension isn't available, instead of erroring
loudly at install time. Root cause (not yet confirmed, but likely): `pip
install git+...` does a **shallow, non-recursive** clone, and xformers
pulls its actual CUDA kernels (flash-attention, cutlass) in via **git
submodules** -- a non-recursive clone silently produces a Python-only
package with no compiled kernels at all. Fix being tried: clone manually
with `--recursive` and build from that checkout instead of `pip install
git+`:
```bash
git clone --recursive https://github.com/facebookresearch/xformers.git /tmp/xformers-src
cd /tmp/xformers-src
export TORCH_CUDA_ARCH_LIST="10.3"
pip install -e .
```
Diagnostic to confirm the extension actually built: `python -c "from
xformers import _C; print(_C.__file__)"` -- if this raises ImportError,
the kernels didn't compile, regardless of what `pip show xformers` or a
bare `import xformers` says. **Outcome not yet known as of this note** --
even if this succeeds, xformers' flash/cutlass kernel *templates* need to
actually support sm_103 in their source for the compiled result to work at
call time (a from-source build can compile cleanly and still hit the same
`NotImplementedError: no operator found ... requires device with
capability <= (9,0)` at runtime if the CUTLASS templates simply don't cover
this SM version yet) -- that's the next thing to check once the extension
itself is confirmed to actually be present.

## -23. `update_kv_cache` fixed-chunk path: ported flashdreams' `BlockKVCache` algorithm for the steady-state case (implemented, correctness-verified on CPU, not yet benchmarked on GPU)

Item -20 found CUDA graph capture blocked on `update_kv_cache`'s
`torch.searchsorted(...).item()` calls and its `torch.cat`-based
variable-length intermediate tensors -- fundamentally incompatible with
graph capture's fixed-shape/no-CPU-sync requirement, not just a stray sync
to remove. Item -21 had scoped (not integrated) `flashdreams`'
`core/attention/kvcache.py::BlockKVCache` (Apache-2.0,
`C:\workspace\world\flashdream_public`) as a strong match: an explicitly
"CUDA-graph compatible" sink+FIFO-window cache using symbolic/plain-int
bounds and fixed-shape in-place slice writes instead of `.item()` or `cat`.

**Implemented tonight** in `ltx-core/.../attention.py`: `update_kv_cache`
now has two internal paths --
- `_update_kv_cache_general` (unchanged): the original searchsorted-based
  logic, used only for calls that aren't a plain "advance by one fixed
  chunk" or "redo the same range" -- in practice, just Echo-WM's one-off
  irregular first block (1 latent frame wide, vs. every later block's fixed
  `video_chunk_size=3`). This never repeats and is never part of the
  steady-state loop CUDA graph capture actually needs to be static, so its
  one `.item()` sync is harmless.
- `_update_kv_cache_fixed_chunk` (new): `BlockKVCache`'s roll-left +
  sink-prefix arithmetic, ported to plain Python ints (not
  `torch.sym_min`/`sym_max` -- those exist for `torch.compile`/dynamo
  symbolic tracing, which isn't in play for a raw `torch.cuda.graph()`
  capture; ordinary `min`/`max` on host-side ints that are already known
  synchronously -- `start` is passed in directly by every caller, `k.shape[1]`
  is always available without a sync -- is sufficient). Only in-place slice
  writes, no `torch.cat`, no `.item()`. Activates automatically once a
  cache dict has seen two consecutive calls that are either an exact
  chunk-width advance or an exact-range redo *and* `local_attn_size ==
  cache capacity` (always true for every real cache in this pipeline --
  confirmed via `model.py::init_av_kv_caches`'s `allocate()`).

All cache-write call sites (`video_self`, `audio_self`, `a2v`, `v2a`,
`video_ucpe` via `_ucpe_cache_attend`) funnel through this single function,
so no other file needed to change -- confirmed by tracing every call site
in `transformer.py`/`attention.py`.

**Correctness verified on CPU (no GPU needed)**: extended
`test_kv_cache_correctness.py` to compare the original boolean-mask
implementation against both the searchsorted rewrite *and* the real
current `update_kv_cache` (imported directly from `attention.py`, not a
copy) across 11 scenarios -- the original 7 (now with `capacity ==
local_attn_size`, matching real allocation, so the new fixed-chunk path
actually activates instead of being skipped by generous test headroom), 3
new "irregular first block" scenarios (width-1 then width-3, mirroring
Echo-WM's real `causal_video_blocks` layout), and the exact overlapping-start
call sequence from `tests/test_echo_wm_causal.py`'s own hardcoded-ground-truth
pytest (`test_sink_plus_fifo_cache_rollover_and_block_replacement`) --
including that test's own expected final positions
(`[0,1,6,7,8,9,10]`) and post-redo trailing values (`[99,99,99]`). All 11
scenarios: byte-identical match. Run: `python test_kv_cache_correctness.py`.

**Not yet done**: run on the real GB300 box (needs `git pull` + restart) --
first re-try `ECHO_WM_GRAPH_CAPTURE_TEST=1` to see if this actually
unblocks capture now, then a clean timing comparison against the
~1.82-1.87s/block baseline (expected to be a small win at best -- item -19
already showed CPU dispatch overhead is spread across the whole model, not
concentrated in this cache update, so the real value here is unblocking
CUDA graphs, not the cache update's own speed in isolation).

## -22. `ECHO_WM_COMPILE_ATTENTION=1`: scoped torch.compile on just the attention call -- confirmed net negative, default off

Narrower alternative to item -4's whole-model `torch.compile` (which
confirmed a recompilation-storm regression). Instead of compiling the
whole model, only `_pytorch_attention_core` -- the reshape -> SDPA ->
reshape sequence in `PytorchAttention.__call__` (`ltx-core/.../attention.py`)
-- is wrapped: `torch.compile(_pytorch_attention_core, dynamic=True)`.
Rationale: the whole-model recompilation storm came from a `self.idx`
guard living *outside* this function entirely (in `transformer.py`), so
scoping compilation to just this small, self-contained function may dodge
that guard altogether. `dynamic=True` is set upfront (not left to
dynamo's default guess-then-recompile-once-it-notices behavior) since
sequence length genuinely varies during KV-cache fill-up before
stabilizing at the windowed size.

**First test (contaminated, not trustworthy):** didn't hit a
recompilation storm (warmup completed normally, ~35s, similar to
uncompiled runs), but that run also had the 4-stale-GPU-process
contention issue (item -15/-18) active, so the ~2.9s/block result
couldn't be attributed to the compile flag.

**Second test, clean (GPU contention resolved, sync verified): confirmed
net negative.** ~2.0-2.25s/block vs. the verified-clean uncompiled
baseline (~1.82-1.87s/block) -- real, attributable regression, not
noise. Plausible explanation: `torch.compile`'s tracing/guard overhead on
a function that's already just one fused SDPA call plus a couple of
reshapes costs more than it saves -- there wasn't much dispatch overhead
concentrated in *this specific function* to remove (see item -19: the
828ms-CPU-vs-86ms-CUDA dispatch-overhead pattern is spread across
*thousands* of ops throughout the whole forward pass, not concentrated in
the attention wrapper alone).

**Default off, confirmed correct as the default.** Don't enable
`ECHO_WM_COMPILE_ATTENTION=1` for normal use -- it's real code left in
place for reference/future retesting if the underlying function ever
changes shape, not something to turn on.

## -20.5. Second graph-capture blocker: `_mask_is_effectively_none`'s GPU read, computed unconditionally for a value nothing on this box consumes (fixed)

After item -23's `BlockKVCache` rewrite fixed the RoPE-freq-grid and
KV-cache `.item()` blockers, re-running `ECHO_WM_GRAPH_CAPTURE_TEST=1` hit
a *different* capture failure at the identical class of bug, in code
written earlier this same session: `AttentionFunction.DEFAULT`
(`attention.py`) computed `mask_arg = None if
_mask_is_effectively_none(mask) else mask` **unconditionally**, before
even checking whether any library that consumes it (FlashAttention-2/3,
FlashInfer) is installed/enabled. `_mask_is_effectively_none` reads a
value back from the GPU (`bool(torch.all(mask == 0.0))`) -- forbidden
during `torch.cuda.graph()` capture (`cudaErrorStreamCaptureUnsupported`).
Traceback pointed at the text cross-attention path specifically
(`_apply_text_cross_attention` -> `context_mask`).

First fix attempted: cache `_mask_is_effectively_none`'s result by tensor
identity (mirrors item -20's RoPE freq-grid device-cache fix), with a
`weakref.finalize` callback to evict the entry the instant the real
tensor is freed (avoids a stale-`id()`-reuse correctness bug that a naive
`id()`-keyed dict would have). **Insufficient alone**: `context_mask` is
rebuilt as a brand-new tensor object on *every single block-forward call*
(`transformer_args.py::prepare()` -> `_prepare_attention_mask`), so
there's no repeat identity to ever hit the cache -- every call, including
the one inside capture, is genuinely "first time seen."

**Real fix**: on this box right now, `mask_arg`'s result is never actually
consumed at all -- xformers/flash-attn are uninstalled (item -24),
FlashInfer is off by default (item -16) -- and `PytorchAttention()` at the
very end of the fallback chain uses the original `mask`, not `mask_arg`.
Made the computation **lazy**: wrapped in a `_mask_arg()` closure, called
only from inside the FA3/FA2/FlashInfer branches, each already guarded by
an availability check. With none of those libraries active, `_mask_arg()`
is never invoked at all -- zero GPU reads on the hot path in the current
config, not just a cheaper one. The identity-cache fix stays in place too
(still correct, still useful if FlashInfer/FA2/FA3 are ever re-enabled
later, when `_mask_arg()` genuinely does get called every block).

**Re-tested on the box after this fix: CAPTURE SUCCEEDED, REPLAY SUCCEEDED.**
First successful graph capture all night --
```
[graph-test] CAPTURE SUCCEEDED. Attempting one replay...
[graph-test] REPLAY SUCCEEDED. This forward() call is graph-capturable -- real integration is worth pursuing.
```
This confirms item -23's `BlockKVCache` rewrite + this mask-laziness fix
together removed every CPU-GPU sync point in the steady-state cache-update
forward path. **This is a feasibility result, not a speedup yet** -- the
diagnostic still discards its capture and runs the normal eager path for
real generation (see item -20's warmup-call design). Real integration
(capture once during warmup, replay on every later block instead of
re-dispatching the model) is new work, not started -- see item -23's "Not
yet done" note for what that would involve. Expected upside if built: the
dispatch-overhead savings item -19 measured (828ms CPU vs 86ms CUDA per
cache-update forward), i.e. meaningful but not the 10x real-time needs.

## -21. `flashdreams` scoping: a mature, CUDA-graph-ready KV cache + RoPE implementation already exists for this exact pattern (not integrated, scoping only)

While investigating item -20's CUDA-graph blocker, checked
`C:\workspace\world\flashdream_public` (a sibling NVIDIA framework) for
prior art. Found `flashdreams/flashdreams/core/attention/` already
implements, in production-tested form, close to everything item -19/-20
were reasoning toward from scratch:

- **`BlockKVCache`** (`kvcache.py`) -- a bounded sink+window causal KV
  cache, explicitly documented as **"CUDA-graph compatible"**. Uses
  `torch.sym_min`/`torch.sym_max` (symbolic-integer ops, not `.item()`)
  for all its write-bounds math -- no CPU-GPU sync at all, unlike
  Echo-WM's `update_kv_cache` (fixed in item -19 to remove `nonzero()`,
  but the replacement `searchsorted(...).item()` still syncs). Pure
  slice-based writes (`self._k[dst_slice] = ...`), no boolean masking
  either. Explicit `before_update`/`update`/`after_update`/`cached_k`/
  `cached_v` lifecycle with distinct filling-phase vs. steady-state-phase
  code paths -- exactly the distinction item -14's "warmup doesn't
  transfer to real generation" investigation inferred empirically.
- **`KVCacheRelativeRotaryPositionEmbedding3D`** (`rope.py`) -- 3D RoPE
  specifically for bounded sink/window caches where tokens move through
  cache slots (vs. `RotaryPositionEmbedding3D` for unbounded monotonic
  positions) -- Echo-WM's exact pattern. All frequency tensors
  precomputed once in `__init__`, device-resident from construction (no
  per-call CPU-GPU copy at all -- the class of bug item -20 found and
  fixed in Echo-WM's own `rope.py`, but avoided here by construction, not
  a patch).
- **`apply_rope_freqs`** applies RoPE via a real fused Triton kernel
  (`rope_kernel.py`) rather than a sequence of separate elementwise ops.

**Not integrated -- scoping only.** These are strong architectural
matches (built for what looks like this exact problem shape), not just
generically similar. Real integration would mean rewiring Echo-WM's
`Attention`/`update_kv_cache` (`ltx-core/.../attention.py`) and
`CausalModelWrapper` (`ltx-causal/.../causal_wrapper.py`) to use these
instead of their own versions -- reconciling shape/dict conventions
between the two codebases, and Echo-WM's joint audio+video modeling
(unclear whether `flashdreams`' patterns, which look video-centric in
`flashdreams-integrations`'s own `SKILL.md`, have a direct analog for
that). Rough estimate: **1-3 days for a single-piece cherry-pick** (e.g.
just the RoPE class), **1-2+ weeks for the full coupled integration**
(`BlockKVCache` + the paired `CUDAGraphWrapper`, needed together to
reach the large -- theoretical, not measured -- speedup item -19's
profiler data suggested: up to ~80-90% off cache-update specifically if
its 828ms-CPU-vs-86ms-CUDA dispatch-overhead pattern is representative of
the whole model). Not started this session -- a real scoping decision for
next time, not incremental patching.

`ECHO_WM_PROFILE_CACHE=1` (`rollout.py`) profiles one real cache-update
`forward()` call (block 3) with `torch.profiler`. Real finding: **Self CUDA
time total: 86ms, Self CPU time total: 828ms** -- the GPU itself is doing
almost no work; nearly all wall-clock cost is Python/kernel-launch
dispatch overhead from a huge number of small individually-dispatched ops
(`aten::linear` x1756, `aten::cat` x2134, `aten::copy_` x4263,
`aten::mul` x3886, `aten::add` x3802). This directly explains why
attention-window/resolution changes never moved cache-update's flat
~0.5-0.6s (item -14/-16): those levers touch *compute*, but this cost is
*dispatch*, a different axis entirely.

**Concrete finding, fixed**: `aten::nonzero` -- 720 calls, paired 1:1 with
`aten::index` (also 720). `nonzero()` forces a CPU-GPU sync every call
(has to wait to learn the output size before allocating it) and is a
well-known PyTorch anti-pattern. Traced to `update_kv_cache`'s
boolean-mask indexing (`old_k[:, keep_old]` etc., `attention.py`) --
`positions` is always sorted ascending (tokens appended in increasing
order), so these masks are always contiguous prefix/suffix selections,
not scattered ones. Rewrote using `torch.searchsorted` + direct slicing
instead of boolean masks -- eliminates `nonzero()` entirely, needs only
one small `.item()` sync per rewritten selection instead of nonzero's
sync-and-gather per masked index.

**Correctness verified before trusting it**: `test_kv_cache_correctness.py`
(new, CPU-only, no GPU/model needed) compares the old boolean-mask
implementation against the new searchsorted one across 7 realistic
scenarios (every attention-window config used this session: 19/7, 10/4,
7/1, 4/1; with and without repeated/redone denoising-step overwrites of
the same block). **All 7 matched byte-identical.**

**Measured impact: not detectable at the `[rollout]` print's 0.1s
precision** (`cache=0.6s`, unchanged). Consistent with the honest
pre-estimate (nonzero+index's *directly reported* cost was only
~135ms CPU/~12ms CUDA out of the 828ms/86ms totals -- ~10-20% of
cache-update, i.e. maybe 50-100ms, below what the coarse block-timer can
show). The bulk of the 828ms CPU overhead is still the thousands of
*other* small ops this fix doesn't touch -- real further progress on this
axis needs the bigger CUDA-graph-capture approach (item -20), not more
targeted op-level fixes like this one. Kept the fix regardless -- it's a
genuine, verified-correct cleanup with no downside, even if its
measured impact is small.

## -20. CUDA graph capture: feasibility-only test added (not yet run)

Following from item -19's finding (CPU dispatch overhead, not GPU
compute, dominates) -- CUDA graphs are the architectural answer to
*that* specific problem (capture the whole op sequence once, replay with
near-zero per-op dispatch cost). Real engineering, not attempted as a
production integration tonight: requires fixed/pre-allocated buffers
(graphs freeze memory addresses, not just shapes), shapes that only
stabilize *after* the KV cache window fills (first few blocks differ
structurally from steady-state), and touches the same inference-critical
code just verified correct in item -19 -- a rushed version risks silently
wrong output, not just a crash.

**What was actually added**: `ECHO_WM_GRAPH_CAPTURE_TEST=1`
(`rollout.py`) -- a feasibility-only diagnostic. At block 5 (steady-state
windowing, not the variable-shape fill-up blocks), attempts a real
`torch.cuda.graph()` capture + replay of the actual cache-update
`forward()` call (warmup on a side stream first, per `torch.cuda.graph`'s
own recommendation, then capture, then one replay). Safe to run
repeatedly against the real production KV cache: `update_kv_cache` is
deliberately idempotent for repeated same-range writes (designed to
handle repeated denoising-step overwrites of the same block -- see its
docstring), so the extra warmup/capture/replay calls converge to the same
final cache state as the one real call already made this block, not
corrupt it. Doesn't change real generation output or speed either way --
purely diagnostic.

**Expected likely blocker, not yet confirmed**: item -19's own
`searchsorted(...).item()` fix is itself a CPU-GPU sync point, which
CUDA graph capture forbids entirely inside the captured region -- the
capture attempt may fail because of the very fix that was just added.
There's a deeper fix available for this specific problem (precompute all
cache-windowing math in pure Python ahead of time, since causal block
boundaries are fully known before the rollout loop starts -- no tensor
ops or GPU sync needed at all, faster than even the searchsorted version
and graph-capture-friendly), but implementing that is a bigger refactor
than fit in tonight's time -- noted for later, not attempted.

**Not yet run.** Whatever the actual capture error (or success) turns out
to be will tell us definitively whether pursuing real CUDA graph
integration is worth the multi-hour effort, instead of guessing.

## -17. "Radical config stack" for max speed -- tried, reverted ("slow and awful")

After fp8 (item -9) was confirmed a clean net loss (2.0-2.1s/block vs.
the 1.6-1.7s baseline -- upcast-per-matmul overhead with no offsetting
compute savings on this compute-bound workload) and the whole
attention-backend avenue closed out (item -16), tried stacking every
remaining aggressive lever at once for one final push:
- **1-step denoising** (`timesteps: [1000]`, down from the confirmed-
  acceptable 2-step `[1000, 500]`)
- **4/1 attention window** (`video_local_attn_size: 4, video_sink_size: 1`
  -- the minimum `CausalCacheConfig.validate()` allows), stacked on top of
  1-step this time (previously only tried alongside 2-step, where it was
  also reported bad -- see item -7)
- **320x192 resolution** (down from 512x288)

**Result: explicitly reported "slow and awful" -- reverted in full**, back
to 512x288 / 2-step / 10/4 window (all three, together, are the confirmed
floor of acceptable quality this session). Numerically the run wasn't
actually slower by fps (~10.74fps, arguably the best raw number of the
session) -- "slow and awful" reads as being about perceived motion
quality/coherence collapsing, not throughput, though this wasn't
separately confirmed.

**Not actually isolated -- worth knowing before retrying any piece of
this alone:** all three changes landed in the same untested run. Two real
open questions this leaves, deliberately not re-tried this session given
time pressure:
- Does **1-step alone** (at 10/4 window, 512x288 -- i.e. only cutting
  steps, nothing else) look acceptable? Never tested in isolation --
  1-step has only ever been tried already stacked with 4/1+320x192.
- Does **7/1 window** (the untested middle ground between confirmed-good
  10/4 and confirmed-bad 4/1) look acceptable? Set once this session but
  overwritten by the 4/1 stack before ever actually being run/reported on
  -- genuinely never tested.

Both would need to be tried **one change at a time**, not stacked, to get
an actual answer -- the stacked test only tells us the *combination* is
bad, not which piece(s) of it are the actual problem.

## -16. Attention-backend investigation, final result: FlashInfer works but is slower than SDPA (disabled by default)

Closing conclusion of the whole item -3/-10/-12/-13/-14 attention-backend
saga. After the torch/CUDA environment recovery (item -15) landed on
`torch==2.14`/cu132, `xformers`/`FlashAttention3`/`FlashAttention2` all
came back as **NOT INSTALLED** (`test_attention_backend.py`'s per-backend
check) -- the torch reinstall churn appears to have removed or broken
their compiled extensions along the way (not investigated further; given
the results below, not worth chasing back).

**FlashInfer is the only accelerated backend actually available now, and
it works correctly** -- `test_attention_backend.py` confirms
`FlashInferAttention` succeeds with the right output shape. But measured
through a real generation, it's **slower** than plain SDPA:
~1.9-2.0s/block (one spike to 3.3s) vs. SDPA's previously-confirmed
~1.6-1.7s/block. Plausible causes, not confirmed: `single_prefill_with_kv_cache`'s
`backend="auto"` selection may pick a suboptimal kernel for this
model's specific shape (1024 tokens, 32 heads, 128 dim_head), or the
batch-dimension indexing/unsqueeze wrapping added around the call (this
model's `[B, T, H*D]` convention vs. FlashInfer's per-request `[T, H, D]`
API) adds enough overhead to erase any kernel-level gain.

**Fixed**: `_flashinfer_enabled = False` added as an explicit local
override in `AttentionFunction.DEFAULT`, gating the FlashInfer fallback
stage off by default -- so `DEFAULT` now falls straight through to SDPA
again (the confirmed-fastest option), rather than FlashInfer winning by
default just because it happens to be the only accelerated library still
installed. `FlashInferAttention` itself is left fully implemented (not
deleted) in case a future shape/config change flips this tradeoff.

**Overall conclusion for this GPU (GB300, compute capability 10.3) and
this model's attention pattern**: none of xformers, FlashAttention (1-4),
SageAttention, or FlashInfer currently beat plain PyTorch SDPA with
explicit `EFFICIENT_ATTENTION`/`CUDNN_ATTENTION` backend priority (item
-3b). This is the practical ceiling for attention-kernel-level speed on
this hardware/model combination as of this session -- further speed work
should focus on the levers that already showed real gains (denoising step
count, attention *window* size -- a different axis from attention
*kernel* choice, resolution, encode pipelining), not more attention-library
swapping.

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

**Update, later same session: tried, hit a real dead end.**
`pip install flash-attn-4==4.0.0b28` succeeds and `pip show` confirms
real metadata (author Tri Dao, from the actual Dao-AILab/flash-attention
repo) -- but its declared top-level import module (checked via
`importlib.metadata.distribution('flash-attn-4').read_text('top_level.txt')`)
is **`flash_attn`** -- the exact same import name as mainline
FlashAttention-2 (`pip install flash-attn`), not a separate `flash_attn_4`
namespace as the standalone test (`test_flashattention4_backend.py`)
initially guessed. Calling `flash_attn_func` from this namespace hits the
**identical** crash as plain FA2 earlier tonight:
```
ImportError: .../flash_attn_2_cuda.cpython-312-aarch64-linux-gnu.so: undefined symbol: _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE
```
-- the traceback shows it loading `flash_attn_2_cuda` specifically, i.e.
genuinely FA2's compiled extension, not an FA4-specific kernel. Either
`flash-attn-4`'s package is a thin wrapper still depending on FA2's
extension, or this environment has stale FA2 files shadowing whatever FA4
actually ships -- not disentangled, and not worth the multi-round
environment-archaeology effort given tonight's earlier xformers
source-build saga (item -24) already cost significant time on the same
class of problem. **Closed for this session.** If revisited: would need a
clean venv (not this box's already-churned environment) to tell whether
FA4 itself works, or introspect `pip show -f flash-attn-4` to see exactly
which files it installs vs. what's already present from earlier FA2
attempts.

## -9. `ECHO_WM_FP8=1`: opt-in fp8 weight storage (implemented, BENCHMARKED: net negative, default off)

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

Default off (`ECHO_WM_FP8=0` in `run_gradio.sh`). **Benchmarked on real
hardware: net negative.** `ECHO_WM_FP8=1` real-generation run (10 blocks)
measured denoise ~1.13-1.2s (vs. ~1.02s baseline), cache ~0.59-0.615s (vs.
~0.53s baseline), total ~1.9-2.0s / 1.27-1.48x too slow (vs. ~1.76-1.8s /
1.17-1.2x baseline) -- worse across every segment, not just flat. Confirms
item -7's suspicion: the bottleneck is dispatch/op-count overhead (items
-21/-0.9/-1.0), not memory bandwidth, so the `UPCAST_DURING_INFERENCE`
upcast-before-every-matmul step (see above) adds an extra op per matmul
without removing anything -- pure overhead here, not a tradeoff. Leave
`ECHO_WM_FP8=0`.

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

**Update, later same session: the dynamo-hinted fix was attempted.**
`causal_ti2vid.py` now sets `torch._dynamo.config.allow_unspec_int_on_nn_module = True`
right before the `torch.compile()` call, still gated behind
`ECHO_WM_COMPILE=1` (default remains off). Not yet cleanly confirmed
whether it actually resolves the recompilation storm -- test by watching
warmup closely for a stuck/repeating heartbeat pattern (the original
failure mode) vs. a normal ~30-35s completion, then compare resulting
block timing against the ~1.82-1.87s clean baseline. If warmup completes
normally but timing doesn't improve over baseline, that confirms the
storm is fixed but compile still isn't a speed win here (plausible given
item -19's finding that dispatch overhead is spread across the whole
model, not concentrated where compile can easily help) -- still useful
information, not a wasted test either way.

**Final outcome: fatal inductor crash, whole-model compile abandoned.**
The `allow_unspec_int_on_nn_module` fix did resolve the recompilation
storm -- warmup progressed normally (clean ~2s heartbeat cadence,
`torch.compile()` wrap itself done in 0.6s) instead of the old stuck
pattern. But tracing hit two `Tensor.item()`/`bool()` graph breaks (one
in `update_kv_cache`'s `searchsorted(...).item()`, one in this session's
own `_mask_is_effectively_none()` -> `bool(torch.all(mask == 0.0))` in
`attention.py:411`) -- individually non-fatal, graph breaks just mean
more separately-traced/compiled subgraphs. But the resumed trace after
one of these breaks then crashed **inductor itself**, not dynamo:
```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Expected !size_bytes_is_heap_allocated_ to be true, but got false.
```
An internal inductor storage-tracking invariant violation, most likely
tripped by the in-place `addcmul_` calls on tensor views in
`apply_split_rotary_emb` (`rope.py:59-60`) combined with the
graph-broken partial trace -- not fixable from our side without
patching PyTorch/inductor. **Whole-model `torch.compile` is now
conclusively closed out for this session/hardware combo** (`torch==2.14+cu132`).
`ECHO_WM_COMPILE` stays default `0`. If revisited later: try
`TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` (dynamo's own suggestion, folds
the `.item()`/`bool()` calls into the graph instead of breaking on
them, which might avoid whichever specific resumed-trace shape is
crashing inductor) and/or a newer torch version with inductor fixes.

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

## -3.6. CUDA JIT cache tuning: size doesn't matter, but a warm cache is a real 2.8x restart-time win (measured)

Enlarged `CUDA_CACHE_MAXSIZE` (default 1GB -> 4GB) and set
`CUDA_CACHE_PATH` explicitly in `run_gradio.sh`, on the theory that the
model's many distinct kernel shapes were evicting the default-size cache,
forcing every server restart to pay full JIT compilation cost again.

**Clean, controlled A/B test tonight** (4 runs, each using a throwaway
`CUDA_CACHE_PATH` so cold/warm state was actually controlled, not
assumed), measuring `[warmup] Done in X.Xs`:

| | Cold | Warm |
|---|---|---|
| 1GB | 76.3s | 27.6s |
| 4GB | 76.5s | 27.5s |

**Cache size (1GB vs 4GB) genuinely makes no difference** -- confirms
the original theory (kernels evicting a too-small cache) was wrong.

**But cold-vs-warm is a real, large effect: ~2.8x faster restart
(76s -> ~27.5s)** when the JIT cache already has this process's kernels
compiled. Since the *default* `CUDA_CACHE_PATH` (`~/.nv/ComputeCache`)
persists across restarts on this box (unlike the throwaway `/tmp/...`
dirs used only for this A/B test), every restart *after the first one*
in a session should already be getting this ~2.8x warmup win for free --
no code change needed, it was already happening. Only the *first* server
start after a fresh checkout/clean cache pays the full ~76s. The larger
4GB `CUDA_CACHE_MAXSIZE` stays applied (harmless, doesn't hurt), but the
warm-cache win comes from cache *persistence*, not cache *size* -- don't
spend more time tuning `CUDA_CACHE_MAXSIZE` itself.

This only affects startup/warmup time, not per-block generation
throughput (confirmed no effect on the steady-state `[rollout]` block
timing, which stayed ~1.15-1.2x-too-slow across every run tonight
regardless of cache state) -- doesn't move the real-time gap at all.

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
