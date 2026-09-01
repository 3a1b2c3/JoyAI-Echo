#!/bin/bash
# Launch Echo-WM Gradio interface -- BASE model (full multi-step diffusion,
# no live streaming preview). This is a copy of run_gradio.sh preserving its
# original defaults, kept alongside it since run_gradio.sh itself now
# defaults to the flash/causal model instead.

set -eo pipefail

# Default paths (can be overridden by environment variables)
CHECKPOINT="${CHECKPOINT:-checkpoints/echo-wm-base.safetensors}"
GEMMA_PATH="${GEMMA_PATH:-checkpoints/gemma-3}"
CONFIG="${CONFIG:-configs/inference_wm.yaml}"
PORT="${PORT:-7860}"

# CUDA's own PTX->SASS JIT cache persists to disk across process restarts,
# but the driver's default max size (1GB) is easily evicted by a model with
# this many distinct kernel shapes -- leaving warmup paying the same one-time
# JIT cost on every single startup instead of only the first. Enlarge it
# (still overridable) so repeated startups actually get faster.
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$HOME/.nv/ComputeCache}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"

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
LOG_FILE="gradio_debug.log"

echo "Starting Echo-WM Gradio interface (base model)..."
echo "  Checkpoint: $CHECKPOINT"
echo "  Gemma: $GEMMA_PATH"
echo "  Config: $CONFIG"
echo "  Port: $PORT"
echo "  Log: $LOG_FILE (full output, including [DEBUG ...] lines, also mirrored here)"
echo ""
echo "The server is ready once the '[server] Serving on ...' line appears below."
echo ""

python gradio_echo_wm.py \
    --checkpoint "$CHECKPOINT" \
    --gemma-path "$GEMMA_PATH" \
    --config "$CONFIG" \
    --port "$PORT" \
    "$@" 2>&1 | tee "$LOG_FILE"
