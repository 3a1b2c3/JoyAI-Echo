# Framework Reference

Echo Director Agent is built on nanobot. This directory contains general reference material for the
underlying framework. For Echo Director installation, video workflows, model routing, and file-storage
configuration, start with the root [`README.md`](../README.md) and
[`.config.local.example.json`](../.config.local.example.json). Upstream framework documentation is
available at [nanobot.wiki](https://nanobot.wiki/docs/latest/getting-started/nanobot-overview).

## Core Docs

Start here for setup and everyday usage.

| Topic | Repo docs | What it covers |
|---|---|---|
| Configuration | [`configuration.md`](./configuration.md) | Providers, tools, channels, MCP, and runtime settings |
| Multiple instances | [`multiple-instances.md`](./multiple-instances.md) | Run isolated bots with separate configs and workspaces |
| CLI reference | [`cli-reference.md`](./cli-reference.md) | Core CLI commands and common entrypoints |
| In-chat commands | [`chat-commands.md`](./chat-commands.md) | Slash commands and periodic task behavior |
| OpenAI-compatible API | [`openai-api.md`](./openai-api.md) | Local API endpoints, request format, and file uploads |

## Advanced Docs

Use these when you want deeper customization, integration, or extension details.

| Topic | Repo docs | What it covers |
|---|---|---|
| Memory | [`memory.md`](./memory.md) | How nanobot stores, consolidates, and restores memory |
| Python SDK | [`python-sdk.md`](./python-sdk.md) | Use nanobot programmatically from Python |
| Channel plugin guide | [`channel-plugin-guide.md`](./channel-plugin-guide.md) | Build and test custom chat channel plugins |
| WebSocket channel | [`websocket.md`](./websocket.md) | Real-time WebSocket access and protocol details |
| Custom tools | [`my-tool.md`](./my-tool.md) | Inspect and tune runtime state with the `my` tool |
