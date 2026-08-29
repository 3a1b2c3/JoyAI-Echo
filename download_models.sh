#!/bin/bash
# JoyAI-Echo: Download models (Echo 1.5 checkpoints)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Which Echo 1.5 checkpoint variant to fetch. Override with:
#   VARIANT=echo15_full_dmd ./download_models.sh   (BF16, 46.14 GB, highest quality)
#   VARIANT=echo15_fp8      ./download_models.sh   (FP8, 27.62 GB, default)
#   VARIANT=echo15_fp4      ./download_models.sh   (FP4, 22.81 GB, needs NVIDIA Model Optimizer support in inference code)
VARIANT="${VARIANT:-echo15_fp8}"

echo ""
echo "================================================================================"
echo "JoyAI-Echo - Download Models (variant: $VARIANT)"
echo "================================================================================"
echo ""

CKPT_DIR="$REPO_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

echo "Checkpoints: $CKPT_DIR"
echo ""
echo "Files to download:"
echo "  1. $VARIANT (Echo 1.5 checkpoint)"
echo "  2. gemma-3-12b-it (text encoder)       ~24 GB"
echo ""

if ! python -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "Installing huggingface_hub..."
    python -m pip install --quiet -U huggingface_hub
fi

# ============================================================================
# 1. Echo 1.5 checkpoint variant from jdopensource/JoyAI-Echo
# ============================================================================
echo "[1/2] Echo Model ($VARIANT)"
echo ""

VARIANT_DIR="$CKPT_DIR/$VARIANT"

if [ -d "$VARIANT_DIR" ] && [ -n "$(ls -A "$VARIANT_DIR" 2>/dev/null)" ]; then
    SIZE=$(du -sh "$VARIANT_DIR" | cut -f1)
    echo "✓ Already exists ($SIZE)"
else
    echo "Downloading $VARIANT (this will take a while)..."
    if VARIANT="$VARIANT" CKPT_DIR="$CKPT_DIR" python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='jdopensource/JoyAI-Echo',
    allow_patterns=[f\"{os.environ['VARIANT']}/*\"],
    local_dir=os.environ['CKPT_DIR'],
)
"; then
        SIZE=$(du -sh "$VARIANT_DIR" | cut -f1)
        echo "✓ Downloaded ($SIZE)"
    else
        echo "✗ Download failed"
        echo "  URL: https://huggingface.co/jdopensource/JoyAI-Echo/tree/main/$VARIANT"
        exit 1
    fi
fi

# ============================================================================
# 2. Gemma text encoder (required — inference.py fails to start without it)
# ============================================================================
echo ""
echo "[2/2] Text Encoder (Gemma 3 12B Instruct)"
echo ""

GEMMA_DIR="$CKPT_DIR/gemma-3-12b"

if [ -d "$GEMMA_DIR" ] && [ -n "$(ls -A "$GEMMA_DIR" 2>/dev/null)" ]; then
    SIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
    echo "✓ Already exists ($SIZE)"
else
    echo "Downloading (~24 GB, this will take a while)..."
    if GEMMA_DIR="$GEMMA_DIR" python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/gemma-3-12b-it',
    local_dir=os.environ['GEMMA_DIR'],
)
"; then
        SIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
        echo "✓ Downloaded ($SIZE)"
    else
        echo "✗ Download failed"
        echo "  Gemma is gated: accept the license at https://huggingface.co/google/gemma-3-12b-it"
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
echo "NOTE: this checkout's inference.py predates the Echo 1.5 checkpoint format"
echo "(checkpoint.json + $VARIANT/*.safetensors). It will not load these files"
echo "until the code is updated to match jd-opensource/JoyAI-Echo's echo_longvideo/ branch."
echo ""
