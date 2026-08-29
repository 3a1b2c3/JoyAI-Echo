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
    "$PYTHON" -m venv .venv
    echo "  ✓ Created .venv"
else
    echo "  ✓ .venv already exists"
fi

source .venv/bin/activate
echo "  ✓ Activated .venv"

# 3. Install dependencies
echo ""
echo "[3/5] Installing dependencies (CUDA 13.0)..."
"$PYTHON" -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install PyTorch from official index
echo "  Installing PyTorch..."
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 > /dev/null 2>&1

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
bash "$REPO_DIR/download_models.sh"

# 5. Run examples
echo ""
echo "[5/5] Ready to run examples!"
echo ""
echo "================================================================================"
echo "QUICKSTART"
echo "================================================================================"
echo ""
echo "Run all prompt files in prompts/ (each is a multi-shot story):"
echo "  python inference.py"
echo ""
echo "Run a single prompt file:"
echo "  python inference.py --prompts-glob 'test_001.json'"
echo ""
echo "Custom resolution:"
echo "  python inference.py --video-width 1024 --video-height 576"
echo ""
echo "================================================================================"
echo ""

# Optional: run default example
read -p "Run one example prompt file now (test_001.json, ~15 shots, ~50-60 min on A100/H100)? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running inference on prompts/test_001.json..."
    echo "Expected output size: ~2-3 GB"
    echo ""

    START_TIME=$(date +%s)
    if python inference.py --prompts-glob "test_001.json"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        echo ""
        echo "✓ Complete! (${DURATION}s)"
        echo ""
        echo "Outputs:"
        find inference_result -name "combined_shots.mp4" 2>/dev/null | while read -r f; do
            SIZE=$(du -h "$f" | cut -f1)
            echo "  $f ($SIZE)"
        done
    else
        echo "ERROR: Inference failed"
        exit 1
    fi
else
    echo ""
    echo "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Run all prompt files: python inference.py"
    echo "  2. Run one prompt file:  python inference.py --prompts-glob 'test_001.json'"
    echo "  3. Custom resolution:    python inference.py --video-width 1024 --video-height 576"
    echo "  Or use the interactive picker: bash run_examples.sh"
fi

echo ""
