# R2V request and conditioning boundary

The public local entrypoint uses the same request concepts as the online R2V
service. One JSON file describes one output shot. `memory_slots` is an ordered
array of zero to seven references; `condition_img` is an independent clean first
frame and never consumes a memory slot.

The canonical machine-readable contract is
[`schemas/r2v_request.schema.json`](../schemas/r2v_request.schema.json). Local
paths are resolved relative to the JSON file. HTTP(S) resources are downloaded
during condition encoding. For reproducible offline CLI inference, each memory
slot must use `image_url`. The local server additionally accepts a bare `shot_id`
and resolves it from its persisted successful artifacts for the same `work_id`.
It extracts a representative frame and the complete audio track before entering
this common conditioning boundary.

## Stage boundary

`python inference.py --condition-encode` deliberately stops immediately before
the DiT. It performs these operations in order:

1. Load and validate every request, preserve memory-slot order, and truncate the
   prompt to the online 1,500-character limit.
2. Deduplicate prompts/assets and batch compatible work. Text is processed by
   language-only Gemma and the Echo video/audio connectors. Images are resized
   to the request's exact output size and encoded by the video VAE. Memory audio
   is voice-filtered by MSST, normalized to stereo, and encoded by the audio VAE.
3. Assemble per-request CPU tensors. Empty audio slots receive zero latents only
   when another slot has real audio, so audio and video slot positions remain
   aligned.
4. Atomically write one `.safetensors` file per request.

The cache contains:

- `text.video_context`, optional `text.audio_context`, and `text.attention_mask`;
- optional `first_frame_latent` with shape `[1, 1, C, H, W]`;
- optional ordered `memory_video` with shape `[1, slots, C, H, W]`;
- optional `memory_audio` and `memory_audio_timestep`;
- audio segment lengths and slot-center RoPE parameters in metadata;
- request, checkpoint, Gemma, and input-content fingerprints.

Generation loads this bundle, moves only the needed tensors to the DiT device,
and applies the online DMD behavior. If a first-frame latent exists, frame zero
uses timestep zero and is restored after every prediction and re-noise step.
Memory RoPE uses `slot_center`, offset `500`, and stride `50`.

## Commands

Encode all Last Visa conditions, optionally sharded over GPUs:

```bash
torchrun --standalone --nproc-per-node=3 inference.py \
  --config configs/inference.fp8.yaml \
  --condition-encode \
  --conditioning-cache-dir conditioning_cache/the_last_visa
```

Run one request from the complete cache:

```bash
python inference.py \
  --config configs/inference.fp8.yaml \
  --request examples/the_last_visa/requests/009_01_shot_008_nathan_replies_to_elena_r2v.json \
  --conditioning-cache-dir conditioning_cache/the_last_visa
```

The old `--text-encode` spelling remains an alias, but it now executes the whole
conditioning stage rather than producing a text-only artifact.
