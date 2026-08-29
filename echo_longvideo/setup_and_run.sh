#!/bin/bash
# JoyAI-Echo 1.5: Set up the uv environment, download models, run an example
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

VARIANT="${VARIANT:-echo15_fp8}"

echo ""
echo "================================================================================"
echo "JoyAI-Echo 1.5 - Setup & Run (variant: $VARIANT)"
echo "================================================================================"
echo ""

# 1. Check uv
echo "[1/5] Checking uv..."
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it: https://docs.astral.sh/uv/"
    exit 1
fi
echo "  uv: $(uv --version)"

# 2. Create venv
echo ""
echo "[2/5] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv --python 3.11 .venv
    echo "  ✓ Created .venv"
else
    echo "  ✓ .venv already exists"
fi

source .venv/bin/activate
echo "  ✓ Activated .venv"

# 3. Install dependencies
echo ""
echo "[3/5] Installing dependencies (CUDA 13.2)..."

echo "  Installing PyTorch..."
uv pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

echo "  Installing requirements..."
uv pip install -r requirements.txt

if [ "$VARIANT" = "echo15_fp4" ]; then
    echo "  Installing requirements-fp4.txt (NVIDIA ModelOpt)..."
    uv pip install -r requirements-fp4.txt
fi

# Install local packages
for pkg_dir in ltx-core ltx-pipelines ltx-distillation; do
    if [ -d "$pkg_dir" ]; then
        echo "  Installing $pkg_dir..."
        uv pip install -e "$pkg_dir"
    fi
done

echo "  ✓ Dependencies installed"

# 4. Download models
echo ""
echo "[4/5] Downloading models (first run only)..."
echo ""
VARIANT="$VARIANT" bash "$REPO_DIR/download_models.sh"

# 5. Run examples
echo ""
echo "[5/5] Ready to run examples!"
echo ""
echo "================================================================================"
echo "QUICKSTART"
echo "================================================================================"
echo ""
echo "Run the bundled example (examples/the_last_visa/requests/):"
echo "  python inference.py --config configs/inference.fp8.yaml"
echo ""
echo "Use a different precision:"
echo "  python inference.py --config configs/inference.bf16.yaml"
echo "  python inference.py --config configs/inference.fp4.yaml"
echo ""
echo "Custom resolution:"
echo "  python inference.py --config configs/inference.fp8.yaml --video-width 1024 --video-height 576"
echo ""
echo "================================================================================"
echo ""

case "$VARIANT" in
    echo15_full_dmd) INFERENCE_CONFIG="configs/inference.bf16.yaml" ;;
    echo15_fp4)      INFERENCE_CONFIG="configs/inference.fp4.yaml" ;;
    *)               INFERENCE_CONFIG="configs/inference.fp8.yaml" ;;
esac

read -p "Run the bundled example now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running inference.py --config $INFERENCE_CONFIG..."
    echo ""

    START_TIME=$(date +%s)
    if python inference.py --config "$INFERENCE_CONFIG"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        echo ""
        echo "✓ Complete! (${DURATION}s)"
        echo ""
        echo "Outputs:"
        find inference_result -name "*.mp4" 2>/dev/null | while read -r f; do
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
    echo "  1. Run the bundled example: python inference.py --config configs/inference.fp8.yaml"
    echo "  2. Or use the interactive picker: bash run_examples.sh"
    echo "  3. Or launch the Director Agent UI: bash run_ui.sh"
fi

echo ""
