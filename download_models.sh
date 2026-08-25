#!/bin/bash
# JoyAI-Echo: Download models only (standalone)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "JoyAI-Echo - Download Models"
echo "================================================================================"
echo ""

# Check Python
if ! command -v python &>/dev/null; then
    echo "ERROR: Python not found. Install Python 3.11+"
    exit 1
fi

# Create checkpoints directory
CKPT_DIR="$REPO_DIR/checkpoints"
mkdir -p "$CKPT_DIR"
echo "Checkpoints directory: $CKPT_DIR"
echo ""

# File size estimates
echo "Files to download:"
echo "  1. echo-longvideo-release.safetensors  2.5 GB"
echo "  2. gemma-3-12b/                        8.0 GB"
echo "  ────────────────────────────────────────────"
echo "  Total:                                 10.5 GB"
echo ""

TOTAL_SIZE=0
DOWNLOADED=0

# ============================================================================
# 1. Echo model (main diffusion model)
# ============================================================================
echo "[1/2] echo-longvideo-release.safetensors"
echo ""

MODEL_FILE="$CKPT_DIR/echo-longvideo-release.safetensors"

if [ -f "$MODEL_FILE" ]; then
    SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "  ✓ Already downloaded ($SIZE)"
    DOWNLOADED=$((DOWNLOADED + 1))
else
    echo "  Downloading from jdopensource/JoyAI-Echo..."
    echo "  Size: 2.5 GB | Time: ~5-10 min (depends on connection)"
    echo ""

    if python << 'PYEOF'
from huggingface_hub import hf_hub_download
import os

try:
    path = hf_hub_download(
        repo_id='jdopensource/JoyAI-Echo',
        filename='echo-longvideo-release.safetensors',
        local_dir=os.environ['CKPT_DIR']
    )
    size = os.path.getsize(path) / (1024**3)
    print(f"\n  ✓ Downloaded ({size:.2f} GB)")
except Exception as e:
    print(f"\n  ✗ ERROR: {e}")
    exit(1)
PYEOF
    then
        DOWNLOADED=$((DOWNLOADED + 1))
    else
        echo "  ✗ Download failed"
        exit 1
    fi
fi

# ============================================================================
# 2. Gemma-3 text encoder
# ============================================================================
echo ""
echo "[2/2] gemma-3-12b (text encoder)"
echo ""

GEMMA_DIR="$CKPT_DIR/gemma-3-12b"

if [ -d "$GEMMA_DIR" ] && [ -n "$(ls -A "$GEMMA_DIR" 2>/dev/null)" ]; then
    SIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
    echo "  ✓ Already downloaded ($SIZE)"
    DOWNLOADED=$((DOWNLOADED + 1))
else
    echo "  Downloading from google/gemma-2-12b..."
    echo "  Size: 8.0 GB | Time: ~15-20 min (depends on connection)"
    echo ""

    if python << 'PYEOF'
from huggingface_hub import snapshot_download
import os

try:
    path = snapshot_download(
        repo_id='google/gemma-2-12b',
        local_dir=os.environ['GEMMA_DIR'],
        local_dir_use_symlinks=False
    )
    import subprocess
    result = subprocess.run(['du', '-sh', path], capture_output=True, text=True)
    size = result.stdout.split()[0]
    print(f"\n  ✓ Downloaded ({size})")
except Exception as e:
    print(f"\n  ✗ ERROR: {e}")
    exit(1)
PYEOF
    then
        DOWNLOADED=$((DOWNLOADED + 1))
    else
        echo "  ✗ Download failed"
        echo "  WARNING: Will attempt to use fallback or CPU encoding"
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

echo "Total downloaded: $DOWNLOADED / 2"
echo ""

# Show final sizes
echo "Checkpoint directory contents:"
if [ -d "$CKPT_DIR" ]; then
    du -sh "$CKPT_DIR"/*  2>/dev/null | sed 's/^/  /'
    echo ""
    echo "Total: $(du -sh "$CKPT_DIR" | cut -f1)"
else
    echo "  (empty)"
fi

echo ""
if [ "$DOWNLOADED" -eq 2 ]; then
    echo "✓ All models downloaded successfully!"
    echo ""
    echo "Next step: bash setup_and_run.sh"
else
    echo "⚠ WARNING: Some models failed to download"
    echo "  Try running this script again or check your internet connection"
    exit 1
fi

echo ""
