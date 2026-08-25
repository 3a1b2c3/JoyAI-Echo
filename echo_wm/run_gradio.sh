#!/bin/bash
# Launch Echo-WM Gradio interface

set -e

# Default paths (can be overridden by environment variables)
CHECKPOINT="${CHECKPOINT:-checkpoints/echo-wm-base.safetensors}"
GEMMA_PATH="${GEMMA_PATH:-checkpoints/gemma-3}"
CONFIG="${CONFIG:-configs/inference_wm.yaml}"
PORT="${PORT:-7860}"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Please download the checkpoint first:"
    echo "  huggingface-cli download Echo-Team/Echo-WM --local-dir checkpoints/"
    exit 1
fi

# Check if gemma exists
if [ ! -d "$GEMMA_PATH" ]; then
    echo "Error: Gemma model not found at $GEMMA_PATH"
    echo "Please download the model first."
    exit 1
fi

# Launch
echo "Starting Echo-WM Gradio interface..."
echo "  Checkpoint: $CHECKPOINT"
echo "  Gemma: $GEMMA_PATH"
echo "  Config: $CONFIG"
echo "  Port: $PORT"
echo ""
echo "Opening http://0.0.0.0:$PORT"
echo ""

python gradio_echo_wm.py \
    --checkpoint "$CHECKPOINT" \
    --gemma-path "$GEMMA_PATH" \
    --config "$CONFIG" \
    --port "$PORT" \
    "$@"
