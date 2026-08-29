#!/bin/bash
# Echo-WM: Interactive example runner
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "Echo-WM - Run Examples"
echo "================================================================================"
echo ""

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run setup_and_run.sh first"
    exit 1
fi

source .venv/bin/activate
echo "✓ Activated .venv"
echo ""

# Which checkpoint variant is actually on disk
CHECKPOINT=""
for v in echo-wm-base echo-wm-flash; do
    if [ -f "checkpoints/$v.safetensors" ]; then
        CHECKPOINT="checkpoints/$v.safetensors"
        break
    fi
done
if [ -z "$CHECKPOINT" ] || [ ! -d "checkpoints/gemma-3" ]; then
    echo "ERROR: Models not downloaded. Run download_models.sh first"
    exit 1
fi

mapfile -t CASES < <(find examples/wm_cases -mindepth 1 -maxdepth 1 -type d | sort)

echo "Detected checkpoint: $CHECKPOINT"
echo ""
echo "Available examples:"
echo ""
for i in "${!CASES[@]}"; do
    echo "  $((i + 1)). $(basename "${CASES[$i]}")"
done
echo "  $(( ${#CASES[@]} + 1 )). Custom image + prompt + action string"
echo ""

read -p "Choose example (1-$(( ${#CASES[@]} + 1 ))): " choice

if [ "$choice" -ge 1 ] 2>/dev/null && [ "$choice" -le "${#CASES[@]}" ] 2>/dev/null; then
    SELECTED="${CASES[$((choice - 1))]}"
    echo ""
    echo "Running $(basename "$SELECTED")..."
    python scripts/run_wm_case.py --case "$SELECTED" --checkpoint "$CHECKPOINT" --gemma-path checkpoints/gemma-3
elif [ "$choice" = "$(( ${#CASES[@]} + 1 ))" ]; then
    echo ""
    read -p "Image path: " image
    read -p "Prompt (use Environment/Character/Style/Perspective/Sounds/Speech fields — see PROMPT_SKILL.md): " prompt
    read -p "Action string (e.g. w-60,a-60,w-60,d-60): " action
    echo ""
    echo "Running inference_wm.py..."
    python inference_wm.py \
        --image "$image" \
        --prompt "$prompt" \
        --action-str "$action" \
        --checkpoint "$CHECKPOINT" \
        --gemma-path checkpoints/gemma-3 \
        --output outputs/custom_result.mp4
else
    echo "Invalid choice"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Complete!"
echo "================================================================================"
echo ""

echo "Results:"
find outputs -name "*.mp4" 2>/dev/null | while read -r f; do
    SIZE=$(du -h "$f" | cut -f1)
    echo "  $f ($SIZE)"
done

echo ""
echo "View a video with: mpv <path-from-above>"
echo ""
