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

# Check models
if [ ! -f "checkpoints/echo-longvideo-release.safetensors" ]; then
    echo "ERROR: Models not downloaded. Run download_models.sh first"
    exit 1
fi

echo "Available examples:"
echo ""
echo "  1. Single shot (9.6 sec, ~1-2 min)"
echo "  2. Multi-shot (5 min, ~50-60 min)"
echo "  3. Custom prompt"
echo "  4. Custom resolution (1024x576)"
echo "  5. Custom FPS (30 fps)"
echo ""

read -p "Choose example (1-5): " choice

case "$choice" in
    1)
        echo ""
        echo "Running single shot example..."
        echo "Time: ~1-2 min (A100/H100) | Output: ~100-150 MB"
        echo ""
        python inference.py
        ;;
    2)
        echo ""
        echo "Running multi-shot example (5 minutes)..."
        echo "Time: ~50-60 min (A100/H100) | Output: ~2-3 GB"
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python inference.py --prompts_glob 'prompts/example_multi_shot.json'
        else
            echo "Cancelled"
        fi
        ;;
    3)
        echo ""
        read -p "Enter custom prompt: " prompt
        echo ""
        echo "Generating video for prompt: $prompt"
        python inference.py --prompt "$prompt"
        ;;
    4)
        echo ""
        echo "Running with custom resolution (1024x576)..."
        python inference.py --width 1024 --height 576
        ;;
    5)
        echo ""
        echo "Running with custom FPS (30)..."
        python inference.py --fps 30
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

# Show results
echo "Results saved to:"
RESULTS_DIR="inference_result/dmd"
if [ -d "$RESULTS_DIR" ]; then
    find "$RESULTS_DIR" -name "*.mp4" -o -name "*.wav" 2>/dev/null | while read f; do
        SIZE=$(du -h "$f" | cut -f1)
        echo "  $f ($SIZE)"
    done || echo "  (no outputs)"
else
    echo "  $RESULTS_DIR (not found)"
fi

echo ""
echo "View videos:"
echo "  mpv inference_result/dmd/*/inference_*/video/*.mp4"
echo ""
