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

CKPT_DIR="$REPO_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

echo "Checkpoints: $CKPT_DIR"
echo ""

# ============================================================================
# 1. Echo model (try multiple sources)
# ============================================================================
echo "[1/2] Echo Main Model"
echo ""

MODEL_FILE="$CKPT_DIR/echo-longvideo-release.safetensors"

if [ -f "$MODEL_FILE" ]; then
    SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "✓ Already exists ($SIZE)"
else
    echo "Model not found on public HuggingFace repos."
    echo ""
    echo "Options:"
    echo "  1. Skip (test mode - will fail at inference)"
    echo "  2. Manual download (provide local file)"
    echo "  3. Use alternative (Evoke/JoyAI-Video-Edit)"
    echo ""
    read -p "Choose (1-3, default 1): " choice

    case "${choice:-1}" in
        1)
            echo "Skipping model download (testing setup only)"
            # Create empty placeholder
            touch "$MODEL_FILE"
            ;;
        2)
            read -p "Enter local file path: " filepath
            if [ -f "$filepath" ]; then
                cp "$filepath" "$MODEL_FILE"
                SIZE=$(du -h "$MODEL_FILE" | cut -f1)
                echo "✓ Copied ($SIZE)"
            else
                echo "✗ File not found: $filepath"
                exit 1
            fi
            ;;
        3)
            echo ""
            echo "Use JoyAI-Video-Edit instead:"
            echo "  cd ../JoyAI-Video-Edit"
            echo "  bash download_models.sh"
            echo "  bash run_server_best.sh"
            exit 0
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
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
    if python -m pip list | grep -q huggingface-hub; then
        if python << 'PYEOF'
from huggingface_hub import snapshot_download
import os
snapshot_download(
    "google/gemma-2-12b",
    local_dir=os.environ['GEMMA_DIR'],
    local_dir_use_symlinks=False,
)
print("✓ Downloaded")
PYEOF
        then
            SIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
            echo "  Size: $SIZE"
        else
            echo "⚠ Download failed (optional)"
        fi
    else
        echo "⚠ huggingface-hub not installed"
        echo "  Run: pip install huggingface-hub"
    fi
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "Setup Status"
echo "================================================================================"
echo ""

if [ -f "$MODEL_FILE" ]; then
    if [ -s "$MODEL_FILE" ]; then
        echo "✓ Main model: $(du -h "$MODEL_FILE" | cut -f1)"
    else
        echo "⚠ Main model: placeholder only (testing mode)"
        echo "  → Inference will fail without real model"
    fi
else
    echo "✗ Main model: missing"
fi

if [ -d "$GEMMA_DIR" ] && [ -n "$(ls -A "$GEMMA_DIR" 2>/dev/null)" ]; then
    echo "✓ Text encoder: $(du -sh "$GEMMA_DIR" | cut -f1)"
else
    echo "⚠ Text encoder: not downloaded"
fi

echo ""
echo "Next:"
echo "  bash run_examples.sh  (will fail without real model)"
echo "  or"
echo "  cd ../JoyAI-Video-Edit && bash download_models.sh  (working alternative)"
echo ""
