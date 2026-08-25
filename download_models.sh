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
echo "  1. echo-longvideo-release.safetensors  2.5 GB"
echo "  2. gemma-3-12b (text encoder)          8.0 GB"
echo "  ────────────────────────────────────────────"
echo "  Total:                                 10.5 GB"
echo ""

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
    echo "Downloading (2.5 GB, ~5-10 min)..."

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
# 2. Gemma text encoder
# ============================================================================
echo ""
echo "[2/2] Text Encoder (Gemma)"
echo ""

GEMMA_DIR="$CKPT_DIR/gemma-3-12b"

if [ -d "$GEMMA_DIR" ] && [ -n "$(ls -A "$GEMMA_DIR" 2>/dev/null)" ]; then
    SIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
    echo "✓ Already exists ($SIZE)"
else
    echo "Downloading (8.0 GB, ~10-15 min)..."

    if hf download google/gemma-2-12b \
        --local-dir "$GEMMA_DIR"; then

        SIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
        echo "✓ Downloaded ($SIZE)"
    else
        echo "⚠ Warning: Gemma download failed (optional)"
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
