# Troubleshooting: `gradio_echo_wm.py` (Flash Preview / streaming UI)

## -4. `ECHO_WM_COMPILE=1`: opt-in torch.compile for the causal transformer (implemented, not yet benchmarked)

`ModelLedger.transformer()` (`utils/model_ledger.py:219`) builds a
brand-new model object from scratch on **every** `CausalTI2VidPipeline.__call__`
-- naively wrapping that in `torch.compile()` would mean paying full
graph-tracing cost on every single generation (same failure mode as the
CUDA-cache-size tuning in item -3.6 that turned out not to help). Fixed the
precondition for compiling to pay off at all: `causal_ti2vid.py` now caches
the built model per `(width, height)` (`self._model_cache`) -- built once,
reused across every later generation at that resolution -- and, only when
`ECHO_WM_COMPILE=1` is set (default off), wraps `x0_model.velocity_model`
in `torch.compile()` (default mode, deliberately **not**
`reduce-overhead`/CUDA-graphs -- the KV-cache argument is a `list[dict]`
with per-block-varying scalar start-frame ints, a much higher-risk
combination for CUDA-graph capture than for plain graph compilation).

**Not yet run on real hardware.** Expected real risk: the KV-cache
structure and changing start-frame ints are classic `torch.compile`
graph-break triggers -- if it breaks the graph every block (likely), most
of the theoretical speedup is lost and this may land close to a wash. The
*first* generation at a given resolution pays real compile time (the
startup `_warmup()` run absorbs this automatically as long as later real
generations use the same width/height as warmup); every later generation
at that resolution should be pure win if the graph holds, since nothing
rebuilds/recompiles after the first hit.

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
