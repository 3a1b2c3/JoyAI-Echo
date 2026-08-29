#!/bin/bash
# JoyAI-Echo 1.5: Interactive example runner
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "JoyAI-Echo 1.5 - Run Examples"
echo "================================================================================"
echo ""

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run setup_and_run.sh first"
    exit 1
fi

source .venv/bin/activate
echo "✓ Activated .venv"
echo ""

if ! compgen -G "checkpoints/echo15_*/*" > /dev/null; then
    echo "ERROR: Models not downloaded. Run download_models.sh first"
    exit 1
fi

# Which checkpoint variant is actually on disk selects the default config.
VARIANT=""
for v in echo15_fp8 echo15_full_dmd echo15_fp4; do
    if [ -d "checkpoints/$v" ] && [ -n "$(ls -A "checkpoints/$v" 2>/dev/null)" ]; then
        VARIANT="$v"
        break
    fi
done
case "$VARIANT" in
    echo15_full_dmd) DEFAULT_CONFIG="configs/inference.bf16.yaml" ;;
    echo15_fp4)      DEFAULT_CONFIG="configs/inference.fp4.yaml" ;;
    *)               DEFAULT_CONFIG="configs/inference.fp8.yaml" ;;
esac

mapfile -t REQUEST_FILES < <(find examples -maxdepth 3 -path "*/requests/*.json" | sort)

echo "Detected checkpoint: ${VARIANT:-none} -> $DEFAULT_CONFIG"
echo ""
echo "Available examples:"
echo ""
echo "  1. Run all bundled R2V requests (${#REQUEST_FILES[@]} found)"
echo "  2. Run a single request file"
echo "  3. Custom resolution (e.g. 1024x576)"
echo "  4. Custom FPS (e.g. 30)"
echo "  5. Use a different precision config (bf16 / fp8 / fp4)"
echo ""

read -p "Choose example (1-5): " choice

case "$choice" in
    1)
        echo ""
        echo "Running all bundled R2V requests..."
        python inference.py --config "$DEFAULT_CONFIG"
        ;;
    2)
        echo ""
        echo "Request files:"
        for i in "${!REQUEST_FILES[@]}"; do
            echo "  $((i + 1)). ${REQUEST_FILES[$i]}"
        done
        read -p "Choose file (1-${#REQUEST_FILES[@]}): " fchoice
        SELECTED="${REQUEST_FILES[$((fchoice - 1))]}"
        echo ""
        echo "Running $(basename "$SELECTED")..."
        python inference.py --config "$DEFAULT_CONFIG" \
            --requests-dir "$(dirname "$SELECTED")" \
            --requests-glob "$(basename "$SELECTED")"
        ;;
    3)
        echo ""
        read -p "Width: " width
        read -p "Height: " height
        echo "Running with custom resolution (${width}x${height})..."
        python inference.py --config "$DEFAULT_CONFIG" --video-width "$width" --video-height "$height"
        ;;
    4)
        echo ""
        read -p "FPS: " fps
        echo "Running with custom FPS ($fps)..."
        python inference.py --config "$DEFAULT_CONFIG" --video-fps "$fps"
        ;;
    5)
        echo ""
        echo "  1. bf16 (configs/inference.bf16.yaml, needs checkpoints/echo15_full_dmd)"
        echo "  2. fp8  (configs/inference.fp8.yaml, needs checkpoints/echo15_fp8)"
        echo "  3. fp4  (configs/inference.fp4.yaml, needs checkpoints/echo15_fp4 + requirements-fp4.txt)"
        read -p "Choose precision (1-3): " pchoice
        case "$pchoice" in
            1) CFG="configs/inference.bf16.yaml" ;;
            2) CFG="configs/inference.fp8.yaml" ;;
            3) CFG="configs/inference.fp4.yaml" ;;
            *) echo "Invalid choice"; exit 1 ;;
        esac
        echo "Running with $CFG..."
        python inference.py --config "$CFG"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "Complete!"
echo "================================================================================"
echo ""

echo "Results:"
find inference_result -name "*.mp4" 2>/dev/null | while read -r f; do
    SIZE=$(du -h "$f" | cut -f1)
    echo "  $f ($SIZE)"
done

echo ""
echo "View a video with: mpv <path-from-above>"
echo ""
