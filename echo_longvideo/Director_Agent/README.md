# 🎬 Echo Director Agent

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/WebUI-TypeScript-3178C6?logo=typescript&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E)
![Storage](https://img.shields.io/badge/Storage-Local--first-8B5CF6)

Echo Director Agent turns a story idea into a structured video workflow. It combines a conversational
agent, a visual production workspace, character-memory review, and a multi-shot generation pipeline.
The agent calls an independently deployed Echo 1.5 video service over HTTP.

This repository contains the agent runtime, WebUI, workflow prompts, and long-video orchestration. It
does **not** bundle model weights or a hosted video-generation service.

## ✨ Highlights

- 🎭 **Director workflow** — plan stories, edit shots, review results, regenerate, and merge a final cut.
- 🧠 **Visual memory** — select character references manually or using your favorite vlm.
- 🎞️ **One-click long video** — expand one idea into a planned, reviewed, multi-shot production.
- 🔌 **Service-based generation** — use an Echo-compatible HTTP service or local debug mode for development.
- 🗂️ **Local-first assets** — keep working files on disk and expose them through the local gateway.
- ☁️ **Optional S3 mapping** — publish only the files that an external service must access.
- 🖥️ **Integrated WebUI** — chat, story editing, shot control, reference selection, progress, and playback.
## 🧭 Architecture

```text
┌──────────────────────┐
│ Browser / Echo WebUI │
└──────────┬───────────┘
           │ HTTP + WebSocket
           ▼
┌──────────────────────┐       ┌────────────────────────┐
│ nanobot AgentLoop    │──────▶│ Configured LLM / VLM   │
│ + Director tools     │       │ provider               │
└──────────┬───────────┘       └────────────────────────┘
           │
           ├──────── HTTP ─────▶ Echo-compatible service
           │                         │
           │◀────── callback ────────┘
           ▼
┌──────────────────────┐
│ Local workspace      │
│ stories · shots      │
│ memory · final media │
└──────────────────────┘
```

The storage layer sits below the workflow. Director and memory code operate on logical asset URLs;
they do not contain vendor-specific bucket or endpoint logic.

## 🚀 Quick start

### Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
- Node.js 20.19+ and npm
- `ffmpeg`
- An OpenAI-compatible LLM/VLM endpoint
- An Echo-compatible video service, unless you enable local debug mode

### Install

```bash
bash setup_local.sh
```

The setup script installs Python and WebUI dependencies and creates `.config.local.json` from the
public example when needed.

### Configure

Copy the local environment template and add your model API key:

```bash
cp .env.example .env
```

`start_local.sh` loads `.env` automatically. Environment variables supplied by the caller take
precedence, so CI and one-off runs can override local defaults without editing files.

Then edit `.config.local.json` and set at least:

```json
{
  "agents": {
    "defaults": {
      "provider": "custom",
      "model": "your-model-name",
      "maxConcurrentRequests": 3
    }
  },
  "providers": {
    "custom": {
      "apiKey": "${NANOBOT_MODEL_API_KEY}",
      "apiBase": "https://llm.example.com/v1"
    }
  },
  "tools": {
    "echoGenerator": {
      "baseUrl": "http://127.0.0.1:8221",
      "callbackBaseUrl": "http://127.0.0.1:18791"
    },
    "memoryReview": {
      "provider": "custom",
      "model": "your-multimodal-model"
    }
  }
}
```

The complete, secret-free template is in [`.config.local.example.json`](.config.local.example.json).
`${VARIABLE}` references are resolved at startup and fail clearly when the variable is missing.

### Run

macOS, Linux, or Git Bash:

```bash
bash start_local.sh
```

Windows Command Prompt:

```bat
start_local.cmd
```

To override a value for one launch:

```bash
NANOBOT_MODEL_API_KEY="temporary-key" bash start_local.sh
```

```bat
set NANOBOT_MODEL_API_KEY=temporary-key && start_local.cmd
```

Open [http://127.0.0.1:5187](http://127.0.0.1:5187). Runtime logs are written to:

```text
.local-runtime/gateway.log
.local-runtime/webui.log
```

To run only the gateway:

```bash
uv run --extra api nanobot gateway \
  --config .config.local.json \
  --workspace .local-workspace \
  --debug
```

## ⚙️ Configuration

### Echo Director runtime

Generic agent limits live under `agents.defaults`. Echo service connectivity lives under
`tools.echoGenerator`:

```json
{
  "agents": {
    "defaults": {
      "maxConcurrentRequests": 3
    }
  },
  "tools": {
    "echoGenerator": {
      "baseUrl": "http://127.0.0.1:8221",
      "callbackBaseUrl": "http://127.0.0.1:18791"
    }
  }
}
```

`maxConcurrentRequests` limits simultaneous agent turns (`0` means unlimited). Generation and merge
jobs are submitted to the configured Echo service and complete asynchronously through callbacks.

This release enables the WebUI WebSocket channel and the Director callback channel in its public
configuration. Other channels inherited from the underlying nanobot framework are not configured or
used by default.

### Model providers

All model credentials live under `providers`. Agent features reference a provider by name instead of
copying keys and endpoints into feature-specific sections.

```json
{
  "providers": {
    "custom": {
      "apiKey": "${NANOBOT_MODEL_API_KEY}",
      "apiBase": "https://llm.example.com/v1",
      "extraHeaders": null
    }
  }
}
```

`custom` supports OpenAI-compatible services. The underlying nanobot runtime also supports providers
such as OpenAI, Anthropic, OpenRouter, Gemini, Ollama, and other entries defined in
`nanobot/providers/registry.py`.

### Echo generation service

```json
{
  "tools": {
    "echoGenerator": {
      "baseUrl": "http://127.0.0.1:8221",
      "callbackBaseUrl": "http://127.0.0.1:18791",
      "httpTimeoutSec": 30
    }
  }
}
```

| Field | Purpose |
| --- | --- |
| `baseUrl` | Root URL of the configured video-generation service. |
| `callbackBaseUrl` | Local Agent callback origin used by Echo Server for R2V and merge completion. |
| `httpTimeoutSec` | Per-request HTTP timeout. |

For the standard local setup, `callbackBaseUrl` must point to the same host and port as
`channels.director_callback` (`http://127.0.0.1:18791` in the example config). The Agent does not
poll job status; Echo Server calls the operation-specific callback when work reaches a terminal state.

The personal release does not send an `Authorization` header to the video service and exposes no
video-service token setting.

### Memory review

```json
{
  "tools": {
    "memoryReview": {
      "enabled": true,
      "autoApprove": false,
      "candidateCount": 24,
      "provider": "custom",
      "model": "your-multimodal-model"
    }
  }
}
```

The VLM route reuses `providers.<name>`. With `autoApprove: false`, the workflow pauses before the
next shot so the user can review the proposed Memory slots in the WebUI.

Memory Workspace is a local asset workbench, not an automatic prompt attachment list:

- Generated-shot candidates and local uploads share an editable text profile and provenance.
- Image uploads receive a short VLM profile when a VLM route is configured. Without one, the asset
  remains available for manual use but is hidden from Agent recommendations until a profile is added.
- Audio can be uploaded separately and paired with an image in Build Memory. Audio profiles are
  manual unless an ASR/audio-capable profiler is integrated; a visual VLM does not invent audio content.
- The Agent reads profiles and asset IDs only and writes `recommended_memory_slot_refs`.
- The user can reorder, add, remove, and pair assets. Applying the draft creates
  `approved_memory_slots`, which is the complete Memory payload sent to R2V.
- `reference_shot_ids` remains narrative context and is never appended to approved Memory slots.

### Local-first file storage

```json
{
  "tools": {
    "fileStorage": {
      "local": {
        "directory": "director/assets",
        "baseUrl": "http://127.0.0.1:8765",
        "routePrefix": "/api/assets"
      },
      "outbound": {
        "backend": "inline"
      }
    }
  }
}
```

Assets are stored under the configured workspace by default. `inline` converts local assets to data
URIs only when they must be sent to an external service.

For services that cannot accept inline media, configure an explicit S3-compatible mapping:

```json
{
  "tools": {
    "fileStorage": {
      "outbound": {
        "backend": "s3",
        "s3": {
          "endpointUrl": "https://s3.example.com",
          "publicBaseUrl": "https://cdn.example.com/echo-assets",
          "bucket": "echo-assets",
          "region": "region-1",
          "keyPrefix": "agent-assets",
          "addressingStyle": "auto",
          "accessKeyId": "${FILE_STORAGE_ACCESS_KEY_ID}",
          "secretAccessKey": "${FILE_STORAGE_SECRET_ACCESS_KEY}",
          "sessionToken": ""
        }
      }
    }
  }
}
```

There are no built-in endpoints, buckets, credentials, or cloud-vendor preferences. Local originals
remain the source of truth.

## 🎥 Workflows

### Interactive Director

```text
idea → story → shot prompts → generation → asset extraction/profile → shot acceptance
     → Agent Memory recommendation → Build Memory approval → next shot → merge
```

The Director tools manage durable state in the workspace. Video jobs complete through the HTTP callback
channel, so long-running generation does not block the model conversation.

### Quick Film

Quick Film is a mode of the same Director workflow, not a separate agent or model runtime. It marks
the session for automatic production and runs:

```text
idea → story → shot prompts → Echo generation → VLM memory review → approval → merge
```

It reuses the Director workspace, provider configuration, memory pipeline, Echo HTTP client, callbacks,
and file storage. The WebUI only changes how much human approval the workflow requests.

## 🗃️ Workspace layout

```text
<workspace>/
├── director/
│   ├── assets/
│   └── works/<work_id>/
│       ├── state.json
│       ├── story.md
│       ├── story_profile.json
│       ├── shots/
│       ├── jobs/
│       ├── memory/
│       └── outputs/
└── sessions/
```

## 🛠️ Development

Install development dependencies:

```bash
uv sync --extra api --extra dev
npm --prefix webui ci
```

Run the checks:

```bash
uv run ruff check nanobot
npm --prefix webui run build
```

The WebUI production build is copied to `nanobot/web/dist/` and can be served by the gateway at
`/webui/`.

Python package builds run this WebUI build automatically. Building an sdist or wheel from source
therefore requires Node.js and npm; installing a prebuilt wheel does not.

## 🧱 Project layout

```text
.
├── nanobot/                 # Agent runtime, channels, tools, config, and memory
├── pe/                      # Prompt-engineering profiles and skills
├── webui/                   # React/Vite production workspace
├── .config.local.example.json
├── setup_local.sh
└── start_local.sh
```

## 🔐 Security

- Keep `.config.local.json`, `.env`, workspace files, and runtime logs out of Git.
- Bind services to `127.0.0.1` unless remote access is intentional and protected.
- Configure a callback secret before exposing the callback channel.
- Keep `tools.exec.enable` off when shell access is unnecessary.
- Add external download domains explicitly; the default allow-list is empty.
- Use scoped, short-lived credentials for optional S3-compatible publishing.

See [SECURITY.md](SECURITY.md) for reporting and deployment guidance.

## 📜 License & acknowledgements

Echo Director Agent is released under the [MIT License](LICENSE).

- We sincerely thank the [nanobot](https://github.com/HKUDS/nanobot) team for their excellent work, which provides the foundation for the general agent runtime.
- We thank [alexwang58](https://github.com/alexwang58), [oasis-cloud](https://github.com/oasis-cloud), and Weijie Wang for their efforts on this work.
- Echo video generation is provided by a separately deployed service and is not bundled here.
- Third-party notices are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
