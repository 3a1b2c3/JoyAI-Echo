#!/bin/bash
# JoyAI-Echo 1.5: Download models (checkpoint variant + Gemma text encoder + MSST voice filter)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Which Echo 1.5 checkpoint variant to fetch. Override with:
#   VARIANT=echo15_full_dmd ./download_models.sh   (BF16, 46.14 GB, highest quality)
#   VARIANT=echo15_fp8      ./download_models.sh   (FP8, 27.62 GB, default)
#   VARIANT=echo15_fp4      ./download_models.sh   (FP4, 22.81 GB, needs requirements-fp4.txt / NVIDIA ModelOpt)
VARIANT="${VARIANT:-echo15_fp8}"

echo ""
echo "================================================================================"
echo "JoyAI-Echo 1.5 - Download Models (variant: $VARIANT)"
echo "================================================================================"
echo ""

CKPT_DIR="$REPO_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

echo "Checkpoints: $CKPT_DIR"
echo ""
echo "Files to download:"
echo "  1. $VARIANT (Echo 1.5 checkpoint)"
echo "  2. gemma-3-12b-it (text encoder)       ~24 GB"
echo "  3. MSST-WebUI source + Bandit checkpoint (voice filter)"
echo ""

if ! python -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "Installing huggingface_hub..."
    python -m pip install --quiet -U huggingface_hub
fi

# ============================================================================
# 1. Echo 1.5 checkpoint variant from jdopensource/JoyAI-Echo
# ============================================================================
echo "[1/3] Echo Model ($VARIANT)"
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
echo "[2/3] Text Encoder (Gemma 3 12B Instruct)"
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
# 3. MSST voice filter (source + Bandit checkpoint, pinned by scripts/setup_msst.py)
# ============================================================================
echo ""
echo "[3/3] MSST voice filter (third_party/MSST-WebUI + checkpoints/msst)"
echo ""

if [ -d "$CKPT_DIR/msst" ] && [ -n "$(ls -A "$CKPT_DIR/msst" 2>/dev/null)" ]; then
    echo "✓ Already installed"
else
    if ! python -c "import librosa, soundfile, ml_collections, omegaconf" >/dev/null 2>&1; then
        echo "Installing MSST inference dependencies (requirements-msst.txt)..."
        python -m pip install --quiet -r requirements-msst.txt
    fi
    echo "Running scripts/setup_msst.py..."
    if python scripts/setup_msst.py; then
        echo "✓ MSST installed"
    else
        echo "✗ MSST setup failed — voice_filter will need memory.voice_filter.enabled: false in the config"
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
