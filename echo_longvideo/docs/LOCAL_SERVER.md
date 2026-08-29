# Local R2V scheduler

The root `server.py` is the command-line entry point for a local inference
service, not a proxy. Its implementation is organized under `server/`, while
model execution stays in this repository through `InferenceEngine`.

## Runtime model

R2V and merge jobs are scheduled from process memory. Every state transition is
mirrored to a SQLite recovery journal, which is not polled as a work queue.
There is one worker thread and one `LocalModelRuntime` per configured logical
CUDA device; model objects are never shared between GPUs. Each free worker
atomically claims one FIFO job, so a worker cannot reserve a batch and leave
other GPUs idle.

```text
POST /r2v
    -> in-memory FIFO
    -> live GPU/RAM admission
    -> one job per free GPU -> conditioning-cache lookup
       -> cache miss: retain generator on GPU when VRAM permits
          -> otherwise retain on CPU when RAM permits, otherwise release
          -> encode text/image/audio -> save cache
       -> cache hit: reuse current weight tier
    -> load selected release checkpoint once when needed
    -> sequential DMD inference -> tiled VAE decode -> MP4 write
    -> artifact index -> optional callback
```

Server deployment and model inference use separate YAML files. Start the
service with a server YAML; its `inference.config` field selects the inference
YAML, and that inference YAML selects the checkpoint under `paths.checkpoint`.
A server process serves one checkpoint configuration; run separate processes
with separate ports and GPU sets when several checkpoint variants must be
available simultaneously.

Environment variables such as `ECHO_INFERENCE_CONFIG` and `ECHO_CHECKPOINT`
remain supported as explicit deployment overrides, but are not needed for the
normal YAML-driven startup path.

The server defaults to `auto` DiT residency. It compares live free VRAM
with the measured whole-device peak for the selected release checkpoint plus a
device-relative reserve. A sufficiently large GPU uses the normal fully
resident DiT and keeps it on GPU through tiled decode and subsequent cache-hit
requests. A smaller or occupied GPU enables layerwise swap and transfers only
the active/prefetched transformer blocks to CUDA. `resident` and `swap` force a
profile for explicit performance testing. The server YAML's
`runtime.dit_residency` overrides the inference YAML's offload flag; the
standalone `inference.py` CLI continues to obey its inference YAML exactly. It
never changes checkpoint precision.

`configs/server.consumer.yaml` selects the standalone FP4 inference profile and
forces layerwise swap. This is the recommended consumer-GPU deployment: on the
measured RTX 5090 workload it used 16.29 GiB peak VRAM and 26.51 GiB peak
process RAM. Tiled VAE decode remains enabled (`64` frames / `512` pixels with
the configured overlaps).

## Scheduling and weight lifecycle

A worker probes CUDA with `torch.cuda.mem_get_info()` and host memory with
`psutil.virtual_memory()` before claiming work. Headroom is derived from the
reported device/system totals (5% GPU and 10% RAM by default), rather than fixed
GiB thresholds. The ratios can be overridden with
`ECHO_GPU_HEADROOM_FRACTION` and `ECHO_RAM_HEADROOM_FRACTION`.
Cold-load admission additionally accounts for the selected checkpoint's actual
file size and precision format; a warm runtime uses its measured in-memory tensor
storage and current GPU/CPU location. The calculated requirements and mode are
reported under each worker's `/health` resource snapshot.
The resolved `resident` or `swap` choice and its observed/required GiB values
are included in `admission_plan.dit_residency`.

The generator and decode VAEs remain on GPU across cache hits whenever capacity
allows, including across the decode-to-next-denoise boundary. Cache-miss
conditioning is serialized across GPUs so several Gemma copies cannot consume
host RAM simultaneously. If live free VRAM can hold the conditioning stack,
generation weights stay on GPU; otherwise they move to CPU when RAM permits, or
are released as the last fallback. Decode applies the same GPU -> CPU -> release
tiering. OOM always releases all weights. The default idle timeout is `0`, which
disables idle eviction; set `ECHO_MODEL_IDLE_SECONDS` only when sharing GPUs.
MSST is bound to the worker's logical CUDA device and is released before a cold
generator load.

## Task state

The externally visible job status is `queued`, `running`, `succeeded`, or
`failed`. `stage` gives the finer transition:

```text
queued -> claimed -> validating -> conditioning_cache_lookup
       -> [keeping_generator_on_gpu | staging_generator_on_cpu
           | unloading_for_conditioning -> conditioning]
       -> loading_generator -> inferring -> decoding -> writing
       -> succeeded | failed
```

Queued jobs survive process restart. A job interrupted while running is restored
to the FIFO with stage `recovered`; completed/failed status and pending callback
retries are restored without rerunning the job. Agent `job_id` is a durable
idempotency key: retrying the same request returns the original `version_id`,
while reusing it for a different payload returns HTTP 409. `GET /health` exposes
each GPU worker's task, weight location, latest GPU/RAM snapshot, and the
`scheduler` counters consumed by Echo Director admission. `GET /version/{id}`
continues to work after restart.

## Input boundary

The local endpoint accepts up to seven ordered memory slots. Resources may be
base64 `data:` URLs, absolute HTTP(S) URLs, `file://` URLs, or absolute local
paths visible to the server. Inline data is written to a content-addressed local
asset before queueing. A bare memory `shot_id` resolves to the newest successful
local R2V artifact with the same `work_id`; the server extracts a representative
frame and its complete audio track, then submits those assets through the same
conditioning path as explicit `image_url` and `audio_url` inputs. An unavailable
shot reference returns HTTP 409.

## HTTP API

The examples below assume the default base URL, `http://127.0.0.1:8221`.
Interactive OpenAPI documentation is available at `/docs` while the server is
running.

### `GET /health`

Returns service readiness, R2V and merge queue counts, GPU-worker occupancy,
the active inference configuration, memory-residency policy, and FFmpeg
availability.

```bash
curl http://127.0.0.1:8221/health
```

Important response fields:

- `status`: `ok` after both services have started.
- `inference.enabled`: whether GPU inference workers are enabled.
- `inference.workers`: per-GPU state, current task, model location, and latest
  resource snapshot.
- `queues.r2v` and `queues.merge`: counts grouped by `queued`, `running`,
  `succeeded`, and `failed`.
- `scheduler`: aggregate worker and pending-task counters used by Echo Director.
- `ffmpeg_available`: whether the configured FFmpeg executable can be found.

### `POST /r2v`

Queues one reference-to-video generation job.

```bash
curl -X POST http://127.0.0.1:8221/r2v \
  -H 'Content-Type: application/json' \
  -d '{
    "work_id": "demo",
    "shot_id": "shot-002",
    "job_id": "demo-shot-002-v1",
    "prompt": "A cinematic tracking shot through a rainy city street.",
    "condition_img": "/absolute/path/to/first-frame.png",
    "memory_slots": [
      {"shot_id": "shot-001"}
    ],
    "num_frames": 241,
    "width": 1280,
    "height": 736,
    "seed": 42
  }'
```

Request fields:

| Field | Required | Description |
| --- | --- | --- |
| `work_id` | yes | Logical project or sequence identifier. |
| `shot_id` | yes | Identifier for the shot being generated. |
| `prompt` | yes | Non-empty generation prompt. |
| `memory_slots` | yes | Ordered list of up to seven memory references; it may be empty. |
| `job_id` | no | Durable idempotency key supplied by the caller. |
| `condition_img` | no | First-frame image as an HTTP(S) URL, `file://` URL, absolute path, or base64 data URL. |
| `callback_url` | no | HTTP(S) endpoint notified when the job succeeds or fails. |
| `callback_context` | no | Object copied into supported callback context fields. |
| `num_frames` | no | Positive frame count; otherwise the inference YAML default is used. |
| `duration_sec` | no | Positive target duration metadata. |
| `width`, `height` | no | Positive output dimensions; otherwise configuration defaults are used. |
| `seed` | no | Integer generation seed. |

Each `memory_slots` entry must contain exactly one of:

- `shot_id`: resolve the newest successful local R2V artifact with the same
  `work_id`; or
- `image_url`: use an explicit image resource, optionally paired with
  `audio_url` or `audio_mode: "empty"`.

`image_mode` is currently unsupported. A slot using `shot_id` cannot also
provide audio fields. `X-Nanobot-Director-Callback-Url` may be supplied as a
request header and takes precedence over `callback_url`.

The endpoint returns the accepted task immediately. `task_id`, `version_id`,
and `remote_task_id` identify the same R2V job. Poll `status_url` until `status`
is `succeeded` or `failed`.

```json
{
  "accepted": true,
  "kind": "r2v",
  "task_id": "8d1d...",
  "version_id": "8d1d...",
  "remote_task_id": "8d1d...",
  "work_id": "demo",
  "job_id": "demo-shot-002-v1",
  "shot_id": "shot-002",
  "status": "queued",
  "stage": "queued",
  "queue_position": 1,
  "status_url": "http://127.0.0.1:8221/version/8d1d...",
  "artifacts": []
}
```

Reusing a `job_id` with the same request returns the original `version_id`.
Reusing it with a different request returns HTTP 409.

### `POST /merge`

Queues an ordered MP4 concatenation job. Each shot must provide either a local
R2V `version_id` or an absolute HTTP(S) `video_url`. When both are present,
`version_id` is used.

```bash
curl -X POST http://127.0.0.1:8221/merge \
  -H 'Content-Type: application/json' \
  -d '{
    "work_id": "demo",
    "job_id": "demo-merge-v1",
    "shots": [
      {"version_id": "8d1d..."},
      {"video_url": "https://example.com/shot-003.mp4"}
    ]
  }'
```

The response uses the same asynchronous pattern as `/r2v`, with `kind` set to
`merge`. On success, `video_url` and the `artifacts` list identify the merged
output. The endpoint also accepts the Echo Director envelope fields `payload`,
`job.job_id`, and `callback`; a callback URL may instead be supplied through
`X-Nanobot-Director-Callback-Url`.

### `GET /version/{version_id}`

Returns the current R2V or merge job record. R2V responses include `stage`,
`queue_position`, worker/resource information, generated video fields, and
artifact metadata. Merge responses include the merged `video_url`. Unknown IDs
return HTTP 404.

```bash
curl http://127.0.0.1:8221/version/8d1d...
```

### `GET /artifact/{artifact_id}`

Returns metadata for one generated artifact, including its `version_id`,
`work_id`, optional `shot_id`, `kind`, `role`, public URL, byte size, SHA-256,
timestamps, and generation metadata. Local filesystem paths are not exposed.
Unknown IDs return HTTP 404.

```bash
curl http://127.0.0.1:8221/artifact/2b57...
```

### `GET /media/{asset_path}`

Streams an MP4 stored below the configured `media_root`. Paths are constrained
to that directory. Missing files and traversal attempts return HTTP 404.

```bash
curl -O http://127.0.0.1:8221/media/r2v/demo/shot-002/8d1d.../output.mp4
```

### Callbacks and errors

When a callback URL is present, the server sends one JSON `POST` after success
or failure. R2V callbacks contain `work_id`, `job_id`, `shot_id`,
`remote_task_id`, `status`, and either result URLs or `error`. Merge callbacks
contain the corresponding job identifiers and a `result` object with the
merged URL. Failed callbacks are retried according to
`server.callback_max_attempts`; `0` means unlimited retries.

Common HTTP errors are:

| Status | Meaning |
| --- | --- |
| `404` | Task, artifact, or media file was not found. |
| `409` | Missing referenced shot or conflicting reuse of an idempotency key. |
| `422` | Invalid request fields, resource URL/path, or unsupported media input. |
| `503` | Service is not ready or the selected queue is full. |

## Start the server

Run the checked-in server configuration directly:

```bash
uv run python server.py --config configs/server.consumer.yaml
```

The server YAML owns the bind address, port and scheduler/runtime settings. It
must use one Uvicorn process because GPU workers live in process memory. GPU IDs
are logical IDs after `CUDA_VISIBLE_DEVICES` is applied; listing several IDs
explicitly opts those devices into the service.
