# Director Merge Protocol

This document defines the Echo Director contract for final video merge jobs. Echo Server owns the
FFmpeg operation, the resulting video, and its artifact metadata.

## Request

Nanobot submits merge jobs with:

- Method: `POST`
- Path: `/merge`
- Content type: `application/json`
- Protocol version: `director-http-v1`

Request body:

```json
{
  "protocol_version": "director-http-v1",
  "operation": "merge_shot",
  "job": {
    "job_id": "merge-work-20260429-demo-final-20260429T140000Z",
    "work_id": "work-20260429-demo",
    "target": "final",
    "created_at": "2026-04-29T14:00:00Z"
  },
  "callback": {
    "event_type": "director_remote_result",
    "protocol_version": "director-http-v1",
    "operation": "merge_shot",
    "work_id": "work-20260429-demo",
    "job_id": "merge-work-20260429-demo-final-20260429T140000Z",
    "target": "final",
    "channel": "websocket",
    "chat_id": "cafc067e-c8e3-4fb2-af11-62bc5c04e8d2",
    "session_key": "websocket:cafc067e-c8e3-4fb2-af11-62bc5c04e8d2",
    "inject_back_to_agent": true,
    "url": "http://127.0.0.1:18791/api/director/merge-shot/callback"
  },
  "payload": {
    "work_id": "work-20260429-demo",
    "shot_ids": [1, 2],
    "shots": [
      {
        "shot_id": 1,
        "version_id": "r2v-version-id-for-shot-001"
      }
    ]
  }
}
```

The backend acknowledgement should be JSON:

```json
{
  "accepted": true,
  "task_id": "remote-merge-task-id",
  "version_id": "remote-merge-task-id",
  "status": "queued",
  "status_url": "https://backend.example/version/remote-merge-task-id"
}
```

`remote_task_id` may also be returned as `task_id`.

## Callback

When the merge completes, the backend calls the callback URL from the request:

- Method: `POST`
- Path: `/api/director/merge-shot/callback`
- Content type: `application/json`

Successful callback body:

```json
{
  "work_id": "work-20260429-demo",
  "job_id": "merge-work-20260429-demo-final-20260429T140000Z",
  "status": "completed",
  "session_key": "websocket:cafc067e-c8e3-4fb2-af11-62bc5c04e8d2",
  "channel": "websocket",
  "chat_id": "cafc067e-c8e3-4fb2-af11-62bc5c04e8d2",
  "remote_task_id": "remote-merge-task-id",
  "result": {
    "artifact_url": "http://127.0.0.1:8221/media/merges/remote-merge-task-id.mp4",
    "result_url": "http://127.0.0.1:8221/media/merges/remote-merge-task-id.mp4"
  },
  "completed_at": "2026-04-29T14:02:00Z"
}
```

The Agent treats the returned URL as an opaque media location; filesystem and object-storage details
remain inside Echo Server.

Failed callback body:

```json
{
  "work_id": "work-20260429-demo",
  "job_id": "merge-work-20260429-demo-final-20260429T140000Z",
  "status": "failed",
  "session_key": "websocket:cafc067e-c8e3-4fb2-af11-62bc5c04e8d2",
  "channel": "websocket",
  "chat_id": "cafc067e-c8e3-4fb2-af11-62bc5c04e8d2",
  "remote_task_id": "remote-merge-task-id",
  "error": "merge failed"
}
```

Nanobot applies completed callbacks to `state.final_output_url` or `state.final_output_path`, clears the pending merge job, publishes a workplace update, and exposes the final video in the workplace timeline after the shot list.
