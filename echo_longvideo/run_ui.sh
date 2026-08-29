#!/bin/bash
# JoyAI-Echo 1.5: Launch the Echo video service + Echo Director Agent WebUI
#
# Two processes:
#   1. Echo 1.5 HTTP video service (server.py) -> http://127.0.0.1:8221
#   2. Director Agent gateway + WebUI (nanobot)  -> http://127.0.0.1:5187
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

SERVER_CONFIG="${SERVER_CONFIG:-configs/server.fp8.yaml}"
DIRECTOR_DIR="$REPO_DIR/Director_Agent"

echo ""
echo "================================================================================"
echo "JoyAI-Echo 1.5 - Launch UI (Echo server + Director Agent)"
echo "================================================================================"
echo ""

# --- Preflight ---------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run setup_and_run.sh first"
    exit 1
fi
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it: https://docs.astral.sh/uv/"
    exit 1
fi
if ! command -v npm &>/dev/null; then
    echo "ERROR: Node.js/npm not found. Director Agent's WebUI needs Node.js 20.19+."
    exit 1
fi
if [ ! -f "$SERVER_CONFIG" ]; then
    echo "ERROR: $SERVER_CONFIG not found"
    exit 1
fi
if ! compgen -G "checkpoints/echo15_*/*" > /dev/null; then
    echo "ERROR: Models not downloaded. Run download_models.sh first"
    exit 1
fi

mkdir -p logs

# --- 1. Start the Echo video service in the background -----------------------
echo "[1/2] Starting Echo 1.5 video service ($SERVER_CONFIG)..."
source .venv/bin/activate
python server.py --config "$SERVER_CONFIG" > logs/server.log 2>&1 &
SERVER_PID=$!

cleanup() {
    echo ""
    echo "Stopping Echo video service (pid $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "  Waiting for it to come up on 127.0.0.1:8221..."
READY=0
for _ in $(seq 1 60); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Echo video service exited early — see logs/server.log"
        exit 1
    fi
    if (exec 3<>/dev/tcp/127.0.0.1/8221) 2>/dev/null; then
        exec 3>&-
        READY=1
        break
    fi
    sleep 1
done
if [ "$READY" -ne 1 ]; then
    echo "ERROR: Echo video service did not come up within 60s — see logs/server.log"
    exit 1
fi
echo "  ✓ Echo video service is up (pid $SERVER_PID, log: logs/server.log)"

# --- 2. Set up and start the Director Agent -----------------------------------
echo ""
echo "[2/2] Starting Echo Director Agent..."
cd "$DIRECTOR_DIR"

if [ ! -f ".config.local.json" ] || [ ! -d ".venv" ] || [ ! -d "webui/node_modules" ]; then
    echo "  First run — installing Director Agent dependencies (uv sync + npm ci)..."
    bash setup_local.sh
fi

if [ ! -f ".env" ] || ! grep -q "^NANOBOT_MODEL_API_KEY=." ".env" || grep -q "^NANOBOT_MODEL_API_KEY=your-model-api-key$" ".env"; then
    echo ""
    echo "WARNING: Director_Agent/.env has no real NANOBOT_MODEL_API_KEY set."
    echo "  The chat agent needs its own LLM/VLM provider key (separate from the Echo video model)."
    echo "  Edit Director_Agent/.env and Director_Agent/.config.local.json, then re-run this script."
    echo ""
fi

echo "  Echo Director: http://127.0.0.1:5187/"
echo "  Gateway log:   .local-runtime/gateway.log"
echo "  WebUI log:     .local-runtime/webui.log"
echo ""
bash start_local.sh
