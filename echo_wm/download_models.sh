#!/bin/bash
# Echo-WM: Download checkpoint + Gemma text encoder
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Which checkpoint to fetch. Override with:
#   VARIANT=echo-wm-flash ./download_models.sh   (Flash Preview, autoregressive/causal)
VARIANT="${VARIANT:-echo-wm-base}"

echo ""
echo "================================================================================"
echo "Echo-WM - Download Models (variant: $VARIANT)"
echo "================================================================================"
echo ""

CKPT_DIR="$REPO_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

echo "Checkpoints: $CKPT_DIR"
echo ""
echo "Files to download:"
echo "  1. $VARIANT.safetensors                              ~47.8 GB"
echo "  2. gemma-3-12b-it-qat-q4_0-unquantized (text encoder) (bfloat16 weights)"
echo ""

if ! python -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "Installing huggingface_hub..."
    python -m pip install --quiet -U huggingface_hub
fi

# ============================================================================
# 1. Echo-WM checkpoint variant from Echo-Team/Echo-WM
# ============================================================================
echo "[1/2] Echo-WM checkpoint ($VARIANT)"
echo ""

CKPT_FILE="$CKPT_DIR/$VARIANT.safetensors"

if [ -f "$CKPT_FILE" ]; then
    SIZE=$(du -h "$CKPT_FILE" | cut -f1)
    echo "✓ Already exists ($SIZE)"
else
    echo "Downloading $VARIANT.safetensors (~47.8 GB, this will take a while)..."
    if VARIANT="$VARIANT" CKPT_DIR="$CKPT_DIR" python -c "
import os
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Echo-Team/Echo-WM',
    filename=f\"{os.environ['VARIANT']}.safetensors\",
    local_dir=os.environ['CKPT_DIR'],
)
"; then
        SIZE=$(du -h "$CKPT_FILE" | cut -f1)
        echo "✓ Downloaded ($SIZE)"
    else
        echo "✗ Download failed"
        echo "  URL: https://huggingface.co/Echo-Team/Echo-WM"
        exit 1
    fi
fi

# ============================================================================
# 2. Gemma 3 text encoder (QAT, unquantized weights — NOT the same repo as
#    echo_longvideo's gemma-3-12b; Echo-WM needs the bfloat16 QAT variant)
# ============================================================================
echo ""
echo "[2/2] Text Encoder (Gemma 3 12B, QAT unquantized)"
echo ""

GEMMA_DIR="$CKPT_DIR/gemma-3"

if [ -d "$GEMMA_DIR" ] && [ -n "$(ls -A "$GEMMA_DIR" 2>/dev/null)" ]; then
    SIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
    echo "✓ Already exists ($SIZE)"
else
    echo "Downloading (this will take a while)..."
    if GEMMA_DIR="$GEMMA_DIR" python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/gemma-3-12b-it-qat-q4_0-unquantized',
    local_dir=os.environ['GEMMA_DIR'],
)
"; then
        SIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
        echo "✓ Downloaded ($SIZE)"
    else
        echo "✗ Download failed"
        echo "  Gemma is gated: accept the license at https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
        echo "  then log in with: hf auth login (or huggingface-cli login)"
        exit 1
    fi
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "Download Complete"
echo "================================================================================"
echo ""

echo "Checkpoints:"
du -sh "$CKPT_DIR"/* 2>/dev/null | sed 's/^/  /'
echo ""
echo "Total: $(du -sh "$CKPT_DIR" | cut -f1)"
echo ""
echo "Next: bash run_examples.sh"
echo ""
