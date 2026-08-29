#!/bin/bash
# JoyAI-Echo: Download models
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "JoyAI-Echo - Download Models"
echo "================================================================================"
echo ""

CKPT_DIR="$REPO_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

echo "Checkpoints: $CKPT_DIR"
echo ""
echo "Files to download:"
echo "  1. echo-longvideo-release.safetensors  ~46 GB"
echo "  2. gemma-3-12b-it (text encoder)       ~24 GB"
echo "  ────────────────────────────────────────────"
echo "  Total:                                 ~70 GB"
echo ""

if ! python -m pip show huggingface_hub >/dev/null 2>&1; then
    echo "Installing huggingface_hub..."
    python -m pip install --quiet -U "huggingface_hub[cli]"
fi

# ============================================================================
# 1. Echo model from jdopensource/JoyAI-Echo
# ============================================================================
echo "[1/2] Echo Model"
echo ""

MODEL_FILE="$CKPT_DIR/echo-longvideo-release.safetensors"

if [ -f "$MODEL_FILE" ]; then
    SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "✓ Already exists ($SIZE)"
else
    echo "Downloading (~46 GB, this will take a while)..."

    if hf download jdopensource/JoyAI-Echo \
        echo-longvideo-release.safetensors \
        --local-dir "$CKPT_DIR"; then

        SIZE=$(du -h "$MODEL_FILE" | cut -f1)
        echo "✓ Downloaded ($SIZE)"
    else
        echo "✗ Download failed"
        echo "  URL: https://huggingface.co/jdopensource/JoyAI-Echo"
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

    if hf download google/gemma-3-12b-it \
        --local-dir "$GEMMA_DIR"; then

        SIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
        echo "✓ Downloaded ($SIZE)"
    else
        echo "✗ Download failed"
        echo "  Gemma is gated: accept the license at https://huggingface.co/google/gemma-3-12b-it"
        echo "  then log in with: hf auth login"
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

if [ -f "$MODEL_FILE" ]; then
    echo "✓ Ready to run examples"
    echo ""
    echo "Next: bash run_examples.sh"
else
    echo "✗ Main model missing"
    exit 1
fi

echo ""
