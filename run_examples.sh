#!/bin/bash
# JoyAI-Echo: Interactive example runner
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "JoyAI-Echo - Run Examples"
echo "================================================================================"
echo ""

# Activate venv
if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run setup_and_run.sh first"
    exit 1
fi

source .venv/bin/activate
echo "✓ Activated .venv"
echo ""

# Check models (any downloaded Echo 1.5 variant: echo15_fp8, echo15_full_dmd, echo15_fp4)
if ! compgen -G "checkpoints/echo15_*/*" > /dev/null; then
    echo "ERROR: Models not downloaded. Run download_models.sh first"
    exit 1
fi

mapfile -t PROMPT_FILES < <(find prompts -maxdepth 1 -name "*.json" | sort)

echo "Available examples:"
echo ""
echo "  1. Run all prompt files in prompts/ (${#PROMPT_FILES[@]} found, each is a multi-shot story)"
echo "  2. Run a single prompt file"
echo "  3. Custom resolution (e.g. 1024x576)"
echo "  4. Custom FPS (e.g. 30)"
echo ""

read -p "Choose example (1-4): " choice

case "$choice" in
    1)
        echo ""
        echo "Running all prompt files..."
        python inference.py
        ;;
    2)
        echo ""
        echo "Prompt files:"
        for i in "${!PROMPT_FILES[@]}"; do
            echo "  $((i + 1)). $(basename "${PROMPT_FILES[$i]}")"
        done
        read -p "Choose file (1-${#PROMPT_FILES[@]}): " fchoice
        SELECTED="${PROMPT_FILES[$((fchoice - 1))]}"
        echo ""
        echo "Running $(basename "$SELECTED")..."
        python inference.py --prompts-glob "$(basename "$SELECTED")"
        ;;
    3)
        echo ""
        read -p "Width: " width
        read -p "Height: " height
        echo "Running with custom resolution (${width}x${height})..."
        python inference.py --video-width "$width" --video-height "$height"
        ;;
    4)
        echo ""
        read -p "FPS: " fps
        echo "Running with custom FPS ($fps)..."
        python inference.py --video-fps "$fps"
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
find inference_result -name "combined_shots.mp4" 2>/dev/null | while read -r f; do
    SIZE=$(du -h "$f" | cut -f1)
    echo "  $f ($SIZE)"
done

echo ""
echo "View a video with: mpv <path-from-above>"
echo ""
