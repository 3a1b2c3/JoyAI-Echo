#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "[Echo Server] uv was not found. Install uv and reopen the shell." >&2
  exit 1
fi

if [[ ! -f configs/server.consumer.yaml ]]; then
  echo "[Echo Server] Missing configs/server.consumer.yaml" >&2
  exit 1
fi

echo "[Echo Server] Starting on the configured host and port..."
exec uv run python server.py --config configs/server.consumer.yaml "$@"
