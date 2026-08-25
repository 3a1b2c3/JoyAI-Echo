#!/bin/bash
# JoyAI-Echo: Download models, setup environment, run examples
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "JoyAI-Echo - Setup & Run"
echo "================================================================================"
echo ""

# 1. Check Python
echo "[1/5] Checking Python..."

# Try python, then python3, then python3.11
PYTHON=""
for py in python python3 python3.11; do
    if command -v "$py" &>/dev/null; then
        PYTHON="$py"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found. Install Python 3.11+:"
    echo "  Ubuntu/WSL: sudo apt install python3.11 python3.11-venv python3.11-dev"
    echo "  macOS:      brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$("$PYTHON" --version 2>&1 | awk '{print $2}')
echo "  Python: $PYTHON ($PYTHON_VERSION)"

# 2. Create venv
echo ""
echo "[2/5] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python -m venv .venv
    echo "  ✓ Created .venv"
else
    echo "  ✓ .venv already exists"
fi

source .venv/bin/activate
echo "  ✓ Activated .venv"

# 3. Install dependencies
echo ""
echo "[3/5] Installing dependencies (CUDA 12.8)..."
"$PYTHON" -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install PyTorch from official index
echo "  Installing PyTorch..."
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 > /dev/null 2>&1

# Install requirements
echo "  Installing requirements..."
"$PYTHON" -m pip install -r requirements.txt > /dev/null 2>&1

# Install local packages
for pkg_dir in ltx-core ltx-pipelines ltx-distillation; do
    if [ -d "$pkg_dir" ]; then
        echo "  Installing $pkg_dir..."
        "$PYTHON" -m pip install -e "$pkg_dir" > /dev/null 2>&1
    fi
done

echo "  ✓ Dependencies installed"

# 4. Download models
echo ""
echo "[4/5] Downloading models (first run only)..."
echo ""
echo "  File size estimates:"
echo "  • echo-longvideo-release.safetensors: 2.5 GB"
echo "  • gemma-3-12b (text encoder):         8.0 GB"
echo "  • Total:                               10.5 GB"
echo ""

CKPT_DIR="$REPO_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

# Check main model
MODEL_FILE="$CKPT_DIR/echo-longvideo-release.safetensors"
if [ ! -f "$MODEL_FILE" ]; then
    echo "  Downloading echo-longvideo-release.safetensors (2.5 GB)..."
    if python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='jdopensource/JoyAI-Echo',
    filename='echo-longvideo-release.safetensors',
    local_dir='$CKPT_DIR'
)" 2>&1 | grep -E "Downloading|Downloaded"; then
        SIZE=$(du -h "$MODEL_FILE" 2>/dev/null | cut -f1)
        echo "    ✓ Downloaded ($SIZE)"
    else
        echo "    ERROR: Failed to download model"
        exit 1
    fi
else
    SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "  ✓ Main model ready ($SIZE)"
fi

# Check text encoder
GEMMA_DIR="$CKPT_DIR/gemma-3-12b"
if [ ! -d "$GEMMA_DIR" ] || [ -z "$(ls -A "$GEMMA_DIR" 2>/dev/null)" ]; then
    echo "  Downloading gemma-3-12b (8.0 GB)..."
    if python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/gemma-2-12b',
    local_dir='$GEMMA_DIR',
    local_dir_use_symlinks=False
)" 2>&1 | grep -E "Downloading|Downloaded"; then
        SIZE=$(du -sh "$GEMMA_DIR" 2>/dev/null | cut -f1)
        echo "    ✓ Downloaded ($SIZE)"
    else
        echo "    WARNING: Failed to download Gemma (will use fallback or CPU)"
    fi
else
    SIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
    echo "  ✓ Text encoder ready ($SIZE)"
fi

echo ""
echo "  Total checkpoint size:"
du -sh "$CKPT_DIR" 2>/dev/null | sed 's/^/    /'
echo ""

# 5. Run examples
echo ""
echo "[5/5] Ready to run examples!"
echo ""
echo "================================================================================"
echo "QUICKSTART"
echo "================================================================================"
echo ""
echo "Single shot (9.6 seconds):"
echo "  python inference.py"
echo ""
echo "Multi-shot (5 minutes, from JSON):"
echo "  python inference.py --prompts_glob 'prompts/example_multi_shot.json'"
echo ""
echo "Custom resolution:"
echo "  python inference.py --width 1024 --height 576"
echo ""
echo "================================================================================"
echo ""

# Optional: run default example
read -p "Run default example now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running inference (this will take 1-2 minutes on A100/H100)..."
    echo "Expected output size: ~500 MB - 1 GB per video"
    echo ""

    START_TIME=$(date +%s)
    if python inference.py; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        echo ""
        echo "✓ Complete! (${DURATION}s)"
        echo ""
        echo "Outputs:"
        find inference_result/dmd -type f -name "*.mp4" -o -name "*.wav" 2>/dev/null | while read f; do
            SIZE=$(du -h "$f" | cut -f1)
            echo "  $f ($SIZE)"
        done || echo "  (no outputs yet)"
    else
        echo "ERROR: Inference failed"
        exit 1
    fi
else
    echo ""
    echo "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Run single shot: python inference.py"
    echo "  2. Run multi-shot:  python inference.py --prompts_glob 'prompts/example_multi_shot.json'"
    echo "  3. Custom config:   python inference.py --width 1024 --height 576"
    echo ""
    echo "Expected sizes:"
    echo "  • Single shot (9.6s):    ~100-150 MB"
    echo "  • Multi-shot (5 min):    ~2-3 GB"
    echo "  • Audio + video + logs:  +200-300 MB"
fi

echo ""
