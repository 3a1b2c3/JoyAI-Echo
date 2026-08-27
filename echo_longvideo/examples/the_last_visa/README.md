# The Last Visa R2V example

This directory contains the production R2V requests used to create the selected
shots for *The Last Visa*. Each file under `requests/` is one request using the
same field names as the online `/api/r2v/generate` endpoint.

- `condition_img` is an optional first-frame condition and does not consume a
  memory slot.
- `memory_slots` is ordered and contains at most seven entries.
- Every local memory slot has an `image_url`; it either has an `audio_url` or
  explicitly sets `audio_mode` to `empty`.
- Paths are relative to the request JSON, so the example remains portable.
- The selected output videos are intentionally not copied into this source
  repository. `index.json` preserves their original sequence and provenance.

The six audio files are exact inputs from the online Last Visa case. They are
voice-filtered again during condition encoding, matching the online R2V path.

Prepare all text/image/audio conditions before loading the DiT:

```bash
python inference.py \
  --config configs/inference.bf16.yaml \
  --condition-encode \
  --conditioning-cache-dir conditioning_cache/the_last_visa
```

Run one request from that cache:

```bash
python inference.py \
  --config configs/inference.bf16.yaml \
  --request examples/the_last_visa/requests/009_01_shot_008_nathan_replies_to_elena_r2v.json \
  --conditioning-cache-dir conditioning_cache/the_last_visa
```
