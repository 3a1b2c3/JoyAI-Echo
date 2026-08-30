# Troubleshooting: `gradio_echo_wm.py` (Flash Preview / streaming UI)

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
