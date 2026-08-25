#!/bin/bash
# JoyAI-Echo: Download models (robust version)
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "JoyAI-Echo - Download Models"
echo "================================================================================"
echo ""

# Check HF CLI
if ! command -v huggingface-cli &>/dev/null; then
    echo "Installing huggingface-hub..."
    pip install huggingface-hub
fi

CKPT_DIR="$REPO_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

echo "Checkpoints directory: $CKPT_DIR"
echo ""
echo "Files to download:"
echo "  1. echo-longvideo-release.safetensors  2.5 GB"
echo "  2. gemma-3-12b/                        8.0 GB"
echo "  ────────────────────────────────────────────"
echo "  Total:                                 10.5 GB"
echo ""

# ============================================================================
# 1. Echo model
# ============================================================================
echo "[1/2] Downloading echo-longvideo-release.safetensors"
echo ""

MODEL_FILE="$CKPT_DIR/echo-longvideo-release.safetensors"

if [ -f "$MODEL_FILE" ]; then
    SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "✓ Already exists ($SIZE)"
else
    echo "Downloading (2.5 GB, ~5-10 min)..."
    echo ""

    if huggingface-cli download \
        jdopensource/JoyAI-Echo \
        echo-longvideo-release.safetensors \
        --local-dir "$CKPT_DIR"; then

        SIZE=$(du -h "$MODEL_FILE" | cut -f1)
        echo ""
        echo "✓ Downloaded ($SIZE)"
    else
        echo ""
        echo "✗ Download failed"
        exit 1
    fi
fi

# ============================================================================
# 2. Gemma text encoder
# ============================================================================
echo ""
echo "[2/2] Downloading gemma-3-12b (text encoder)"
echo ""

GEMMA_DIR="$CKPT_DIR/gemma-3-12b"

if [ -d "$GEMMA_DIR" ] && [ -n "$(ls -A "$GEMMA_DIR" 2>/dev/null)" ]; then
    SIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
    echo "✓ Already exists ($SIZE)"
else
    echo "Downloading (8.0 GB, ~10-15 min)..."
    echo ""

    if huggingface-cli download \
        google/gemma-2-12b \
        --local-dir "$GEMMA_DIR"; then

        SIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
        echo ""
        echo "✓ Downloaded ($SIZE)"
    else
        echo ""
        echo "✗ Download failed (will attempt to use fallback)"
    fi
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "Download Summary"
echo "================================================================================"
echo ""

echo "Checkpoint directory contents:"
du -sh "$CKPT_DIR"/* 2>/dev/null | sed 's/^/  /'
echo ""
echo "Total: $(du -sh "$CKPT_DIR" | cut -f1)"
echo ""

if [ -f "$MODEL_FILE" ]; then
    echo "✓ Main model ready"
    echo ""
    echo "Next step: bash run_examples.sh"
else
    echo "✗ Main model missing - download failed"
    exit 1
fi

echo ""
