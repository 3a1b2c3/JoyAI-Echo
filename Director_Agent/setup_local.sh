#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js and npm are required." >&2
  exit 1
fi

cd "${ROOT}"
uv sync --extra api
npm --prefix webui ci

if [[ ! -f "${ROOT}/.config.local.json" ]]; then
  cp "${ROOT}/.config.local.example.json" "${ROOT}/.config.local.json"
  echo "Created .config.local.json."
fi
if [[ ! -f "${ROOT}/.env" ]]; then
  cp "${ROOT}/.env.example" "${ROOT}/.env"
  echo "Created .env. Add your model API key before starting."
fi

echo "Local dependencies are ready. Edit .env, then run bash start_local.sh."
