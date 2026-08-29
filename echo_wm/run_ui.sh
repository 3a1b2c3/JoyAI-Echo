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

CHECKPOINT=""
for v in echo-wm-base echo-wm-flash; do
    if [ -f "checkpoints/$v.safetensors" ]; then
        CHECKPOINT="checkpoints/$v.safetensors"
        break
    fi
done
if [ -z "$CHECKPOINT" ] || [ ! -d "checkpoints/gemma-3" ]; then
    echo "ERROR: Models not downloaded. Run download_models.sh first"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Echo-WM - Launch Web Demo"
echo "================================================================================"
echo ""

CHECKPOINT="$CHECKPOINT" \
GEMMA_PATH="checkpoints/gemma-3" \
PORT="${PORT:-7860}" \
bash run_gradio.sh
