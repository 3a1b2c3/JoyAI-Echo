# Troubleshooting: `gradio_echo_wm.py` (Flash Preview / streaming UI)

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

## -1. Visible stutter/gap between block updates in the live preview (tried a fix, reverted -- still open)

Even with item 0 below fixed (real incremental blocks now arrive), each
new block causes a visible pause: `stream_video` (`gr.Video`) does a full
`<video src=...>` swap per block, which means a full HTTP fetch + decode +
rebuffer in the browser every time, not a seamless append.

**Attempted and reverted:** flipped `stream_video` to
`gr.Video(streaming=True)` plus a `fragmented=True` option on
`encode_video()` (`movflags: frag_keyframe+empty_moov`), assuming
`streaming=True` does client-side MSE append of fragmented MP4 chunks.
**Wrong assumption, confirmed by testing:** this Gradio version's
`streaming=True` actually drives an **HLS** player (`hls.mjs`) that fetches
a `playlist.m3u8` from `/gradio_api/stream/.../playlist.m3u8` -- i.e.
Gradio expects to run its own server-side HLS segmentation, not receive
already-fragmented MP4 files directly. Result: `HLS error: levelEmptyError
-- No Segments found in Playlist`, fatal, video never played at all
(strictly worse than the stutter it was meant to fix). Both changes fully
reverted (`gradio_echo_wm.py`'s `stream_video`, and the `fragmented` param
on `encode_video()` in `media_io.py` -- removed entirely since nothing
uses it anymore, not left as unused dead code).

**Still an open problem.** Whatever the actual fix is, it needs to work
*with* Gradio's HLS-based streaming pipeline (or bypass Gradio's `Video`
component's file-serving path entirely via a custom component), not just
hand it a differently-encoded standalone file per block.

**Investigation in progress -- exact commands to resume with:**
```bash
cd ~/JoyAI-Echo/echo_wm && source .venv/bin/activate
GRADIO_DIR=$(python -c "import gradio, os; print(os.path.dirname(gradio.__file__))")
# routes.py has two `sse_stream` functions (found via `grep -n "def.*stream" routes.py`,
# neither matched the literal "gradio_api/stream" text -- route prefix is
# likely added via an included router elsewhere) -- read both:
sed -n '1420,1470p' "$GRADIO_DIR/routes.py"
sed -n '1615,1670p' "$GRADIO_DIR/routes.py"
# find the actual HLS/.m3u8 segmenter -- this is the real contract to learn:
grep -rn "playlist.m3u8\|\.m3u8" "$GRADIO_DIR" 2>/dev/null | grep -v ".pyc"
# separately, how the Video component's Python side treats streaming=True:
grep -n "streaming" "$GRADIO_DIR/components/video.py" | head -40
```
Not yet run/read this session -- resume here rather than guessing again.

**Alternative not requiring Gradio internals at all:** bypass `gr.Video`
entirely with a custom WebSocket + MediaSource Extensions player, the same
architecture `JoyAI-Video-Edit`'s UI already uses successfully (raw frames
over a live WebSocket into a `<video>` fed by `MediaSource`, no per-chunk
reload) -- more work (custom Gradio component or a raw HTML/JS panel via
`gr.Blocks(head=...)`), but a proven pattern already working elsewhere in
this environment, not a guess at Gradio's undocumented internals.

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
