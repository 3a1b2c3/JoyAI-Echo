#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${ROOT}/.config.local.json"
WORKSPACE="${ROOT}/.local-workspace"
RUNTIME="${ROOT}/.local-runtime"

load_env_defaults() {
  local env_file="$1" line key value
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    [[ "${line}" =~ ^[[:space:]]*(#|$) ]] && continue
    [[ "${line}" == *=* ]] || continue

    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if printenv "${key}" >/dev/null 2>&1; then
      continue
    fi

    if [[ ${#value} -ge 2 && ${value:0:1} == '"' && ${value: -1} == '"' ]]; then
      value="${value:1:${#value}-2}"
    elif [[ ${#value} -ge 2 && ${value:0:1} == "'" && ${value: -1} == "'" ]]; then
      value="${value:1:${#value}-2}"
    fi
    printf -v "${key}" '%s' "${value}"
    export "${key}"
  done <"${env_file}"
}

# Local defaults come from .env. Values already supplied by the caller win.
if [[ -f "${ROOT}/.env" ]]; then
  load_env_defaults "${ROOT}/.env"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi
if [[ ! -d "${ROOT}/.venv" ]]; then
  echo "Run bash ${ROOT}/setup_local.sh first." >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing ${CONFIG}." >&2
  exit 1
fi

mkdir -p "${WORKSPACE}" "${RUNTIME}"
cd "${ROOT}"
uv run --extra api nanobot gateway \
  --config "${CONFIG}" \
  --workspace "${WORKSPACE}" \
  --debug >"${RUNTIME}/gateway.log" 2>&1 &
GATEWAY_PID=$!

cleanup() {
  kill "${GATEWAY_PID}" "${WEBUI_PID:-}" 2>/dev/null || true
  wait "${GATEWAY_PID}" "${WEBUI_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

npm --prefix webui run dev -- --host 127.0.0.1 --port 5187 >"${RUNTIME}/webui.log" 2>&1 &
WEBUI_PID=$!

echo "Echo Director: http://127.0.0.1:5187/"
echo "Gateway log:   ${RUNTIME}/gateway.log"
echo "WebUI log:     ${RUNTIME}/webui.log"
wait "${GATEWAY_PID}" "${WEBUI_PID}"
