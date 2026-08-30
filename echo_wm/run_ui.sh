#!/bin/bash
# Echo-WM: Launch the Gradio web demo
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run setup_and_run.sh first"
    exit 1
fi

source .venv/bin/activate

# Prefer the causal/Flash checkpoint (supports streaming live preview) if
# present; fall back to Base otherwise.
CHECKPOINT=""
ENGINE=""
CONFIG=""
for entry in "echo-wm-flash:causal:configs/inference_wm_causal.yaml" "echo-wm-base:base:configs/inference_wm.yaml"; do
    v="${entry%%:*}"
    rest="${entry#*:}"
    e="${rest%%:*}"
    c="${rest#*:}"
    if [ -f "checkpoints/$v.safetensors" ]; then
        CHECKPOINT="checkpoints/$v.safetensors"
        ENGINE="$e"
        CONFIG="$c"
        break
    fi
done
if [ -z "$CHECKPOINT" ] || [ ! -d "checkpoints/gemma-3" ]; then
    echo "ERROR: Models not downloaded. Run download_models.sh first"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Echo-WM - Launch Web Demo (engine: $ENGINE)"
echo "================================================================================"
echo ""

CHECKPOINT="$CHECKPOINT" \
GEMMA_PATH="checkpoints/gemma-3" \
CONFIG="$CONFIG" \
PORT="${PORT:-7860}" \
bash run_gradio.sh --engine "$ENGINE"
