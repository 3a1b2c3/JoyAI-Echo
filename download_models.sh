#!/bin/bash
# JoyAI-Echo: Download models (with fallback)
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "JoyAI-Echo - Download Models"
echo "================================================================================"
echo ""

CKPT_DIR="$REPO_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

echo "Checkpoints directory: $CKPT_DIR"
echo ""
echo "Files needed:"
echo "  1. echo-longvideo-release.safetensors  2.5 GB"
echo "  2. gemma-3-12b/                        8.0 GB"
echo "  ────────────────────────────────────────────"
echo "  Total:                                 10.5 GB"
echo ""

# ============================================================================
# Install HF tools
# ============================================================================
if ! command -v huggingface-cli &>/dev/null; then
    echo "Installing huggingface-hub..."
    pip install -U huggingface-hub
fi

# ============================================================================
# 1. Echo model
# ============================================================================
echo "[1/2] Echo Model"
echo ""

MODEL_FILE="$CKPT_DIR/echo-longvideo-release.safetensors"

if [ -f "$MODEL_FILE" ]; then
    SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "✓ Already exists ($SIZE)"
else
    echo "Trying to download (2.5 GB)..."
    echo ""

    # Try different possible paths
    DOWNLOADED=0

    # Try 1: Direct repo
    if huggingface-cli download \
        jdopensource/JoyAI-Echo \
        echo-longvideo-release.safetensors \
        --local-dir "$CKPT_DIR" 2>/dev/null; then
        DOWNLOADED=1
        echo "✓ Downloaded from jdopensource/JoyAI-Echo"
    fi

    # Try 2: LTX repo (alternative location)
    if [ $DOWNLOADED -eq 0 ]; then
        if huggingface-cli download \
            SII-YuanyangYin/Evoke \
            echo-longvideo-release.safetensors \
            --local-dir "$CKPT_DIR" 2>/dev/null; then
            DOWNLOADED=1
            echo "✓ Downloaded from SII-YuanyangYin/Evoke"
        fi
    fi

    # Try 3: Manual URL (if HF has direct download)
    if [ $DOWNLOADED -eq 0 ]; then
        echo "⚠ Automated download failed. Manual download required:"
        echo ""
        echo "  1. Visit: https://huggingface.co/jdopensource/JoyAI-Echo"
        echo "  2. Download: echo-longvideo-release.safetensors"
        echo "  3. Place in: $CKPT_DIR/"
        echo ""
        read -p "Press Enter once file is in place, or Ctrl+C to cancel: "

        if [ -f "$MODEL_FILE" ]; then
            SIZE=$(du -h "$MODEL_FILE" | cut -f1)
            echo "✓ Found ($SIZE)"
        else
            echo "✗ File not found"
            exit 1
        fi
    else
        if [ -f "$MODEL_FILE" ]; then
            SIZE=$(du -h "$MODEL_FILE" | cut -f1)
            echo "  Size: $SIZE"
        fi
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
    echo "Downloading (8.0 GB)..."
    echo ""

    if huggingface-cli download \
        google/gemma-2-12b \
        --local-dir "$GEMMA_DIR"; then

        SIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
        echo "✓ Downloaded ($SIZE)"
    else
        echo "⚠ Warning: Gemma download failed"
        echo "  This is optional - inference may use CPU fallback"
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

echo "Checkpoint directory:"
du -sh "$CKPT_DIR" | sed 's/^/  /'
echo ""

if [ -f "$MODEL_FILE" ]; then
    echo "✓ Ready to run!"
    echo ""
    echo "Next: bash run_examples.sh"
else
    echo "✗ Main model still missing"
    exit 1
fi

echo ""
