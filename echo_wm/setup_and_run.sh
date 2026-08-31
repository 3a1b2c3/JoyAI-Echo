#!/bin/bash
# Echo-WM: Set up the uv environment, download models, run a bundled example case
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "Echo-WM - Setup & Run"
echo "================================================================================"
echo ""

# 1. Check python3.11
echo "[1/5] Checking python3.11..."
if ! command -v python3.11 &>/dev/null; then
    echo "ERROR: python3.11 not found. Install Python 3.11 first."
    exit 1
fi
echo "  python3.11: $(python3.11 --version)"

# 2. Create venv (separate from echo_longvideo — the two projects don't share an env)
echo ""
echo "[2/5] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python3.11 -m venv .venv
    echo "  ✓ Created .venv"
else
    echo "  ✓ .venv already exists"
fi

source .venv/bin/activate
echo "  ✓ Activated .venv"
pip install --upgrade pip

# 3. Install dependencies
echo ""
echo "[3/5] Installing dependencies (CUDA 13.2)..."

echo "  Installing PyTorch..."
pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

ARCH="$(uname -m)"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "  Detected $ARCH (e.g. Grace/GB300) — xformers has no aarch64 wheel,"
    echo "  installing everything except it. Attention falls back to PyTorch SDPA."
    grep -v '^xformers' requirements.txt > /tmp/echo_wm_requirements.$$.txt
    pip install -r /tmp/echo_wm_requirements.$$.txt
    rm -f /tmp/echo_wm_requirements.$$.txt
else
    echo "  Installing requirements..."
    pip install -r requirements.txt
fi

echo "  ✓ Dependencies installed"

# 4. Download models
echo ""
echo "[4/5] Downloading models (first run only)..."
echo ""
bash "$REPO_DIR/download_models.sh"

# 5. Run example
echo ""
echo "[5/5] Ready to run examples!"
echo ""
echo "================================================================================"
echo "QUICKSTART"
echo "================================================================================"
echo ""
echo "Run a bundled case:"
echo "  python scripts/run_wm_case.py --case examples/wm_cases/0010 \\"
echo "    --checkpoint checkpoints/echo-wm-base.safetensors --gemma-path checkpoints/gemma-3"
echo ""
echo "List all bundled cases:"
echo "  python scripts/run_wm_case.py --list"
echo ""
echo "Launch the Gradio web demo:"
echo "  bash run_ui.sh"
echo ""
echo "================================================================================"
echo ""

read -p "Run the bundled 0010 example case now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running scripts/run_wm_case.py --case examples/wm_cases/0010..."
    echo ""

    START_TIME=$(date +%s)
    if python scripts/run_wm_case.py --case examples/wm_cases/0010 \
        --checkpoint checkpoints/echo-wm-base.safetensors --gemma-path checkpoints/gemma-3; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        echo ""
        echo "✓ Complete! (${DURATION}s)"
        echo ""
        echo "Outputs:"
        find outputs/wm_cases -name "*.mp4" 2>/dev/null | while read -r f; do
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
    echo "  1. Run a case:          bash run_examples.sh"
    echo "  2. Launch the web demo: bash run_ui.sh"
fi

echo ""
