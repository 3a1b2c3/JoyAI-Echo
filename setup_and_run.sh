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
if ! command -v python &>/dev/null; then
    echo "ERROR: Python not found. Install Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "  Python: $PYTHON_VERSION"

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
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install PyTorch from official index
echo "  Installing PyTorch..."
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 > /dev/null 2>&1

# Install requirements
echo "  Installing requirements..."
pip install -r requirements.txt > /dev/null 2>&1

# Install local packages
for pkg_dir in ltx-core ltx-pipelines ltx-distillation; do
    if [ -d "$pkg_dir" ]; then
        echo "  Installing $pkg_dir..."
        pip install -e "$pkg_dir" > /dev/null 2>&1
    fi
done

echo "  ✓ Dependencies installed"

# 4. Download models
echo ""
echo "[4/5] Downloading models (first run only)..."
CKPT_DIR="$REPO_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

# Check main model
if [ ! -f "$CKPT_DIR/echo-longvideo-release.safetensors" ]; then
    echo "  Downloading echo-longvideo-release.safetensors (2.5 GB)..."
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='jdopensource/JoyAI-Echo',
    filename='echo-longvideo-release.safetensors',
    local_dir='$CKPT_DIR'
)
" || echo "  ERROR: Failed to download model"
else
    echo "  ✓ Main model already downloaded"
fi

# Check text encoder
if [ ! -d "$CKPT_DIR/gemma-3-12b" ]; then
    echo "  Downloading gemma-3-12b (8 GB)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/gemma-2-12b',
    local_dir='$CKPT_DIR/gemma-3-12b',
    local_dir_use_symlinks=False
)
" || echo "  WARNING: Failed to download Gemma. May use fallback."
else
    echo "  ✓ Text encoder already downloaded"
fi

echo "  ✓ Models ready"

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
    echo "Running inference (this will take 1-2 minutes)..."
    python inference.py

    echo ""
    echo "✓ Complete! Check 'inference_result/dmd' for outputs"
    ls -lh inference_result/dmd/*/inference_*/video* 2>/dev/null || echo "(no outputs yet)"
else
    echo ""
    echo "Setup complete! Run 'python inference.py' to generate videos."
fi

echo ""
