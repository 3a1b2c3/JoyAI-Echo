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

OUTPUT_MP4=""
START_TIME=$(date +%s.%N)

if [ "$choice" -ge 1 ] 2>/dev/null && [ "$choice" -le "${#CASES[@]}" ] 2>/dev/null; then
    SELECTED="${CASES[$((choice - 1))]}"
    CASE_NAME="$(basename "$SELECTED")"
    echo ""
    echo "Running $CASE_NAME..."
    python scripts/run_wm_case.py --case "$SELECTED" --checkpoint "$CHECKPOINT" --gemma-path checkpoints/gemma-3
    OUTPUT_MP4="outputs/wm_cases/$CASE_NAME/result.mp4"
elif [ "$choice" = "$(( ${#CASES[@]} + 1 ))" ]; then
    echo ""
    read -p "Image path: " image
    read -p "Prompt (use Environment/Character/Style/Perspective/Sounds/Speech fields — see PROMPT_SKILL.md): " prompt
    read -p "Action string (e.g. w-60,a-60,w-60,d-60): " action
    echo ""
    echo "Running inference_wm.py..."
    OUTPUT_MP4="outputs/custom_result.mp4"
    python inference_wm.py \
        --image "$image" \
        --prompt "$prompt" \
        --action-str "$action" \
        --checkpoint "$CHECKPOINT" \
        --gemma-path checkpoints/gemma-3 \
        --output "$OUTPUT_MP4"
else
    echo "Invalid choice"
    exit 1
fi

END_TIME=$(date +%s.%N)
ELAPSED=$(awk -v a="$START_TIME" -v b="$END_TIME" 'BEGIN { printf "%.2f", b - a }')

echo ""
echo "================================================================================"
echo "Complete!"
echo "================================================================================"
echo ""

echo "Wall-clock time: ${ELAPSED}s"

if [ -n "$OUTPUT_MP4" ] && [ -f "$OUTPUT_MP4" ] && command -v ffprobe &>/dev/null; then
    VIDEO_DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUTPUT_MP4" 2>/dev/null)
    if [ -n "$VIDEO_DURATION" ]; then
        REALTIME_FACTOR=$(awk -v v="$VIDEO_DURATION" -v e="$ELAPSED" 'BEGIN { if (e > 0) printf "%.3f", v / e; else print "n/a" }')
        echo "Video duration:   ${VIDEO_DURATION}s"
        echo "Real-time factor: ${REALTIME_FACTOR}x"
        echo "  (1.0x = generation kept pace with video length; <1.0x = slower than real-time)"
    fi
fi

echo ""
echo "Results:"
find outputs -name "*.mp4" 2>/dev/null | while read -r f; do
    SIZE=$(du -h "$f" | cut -f1)
    echo "  $f ($SIZE)"
done

echo ""
echo "View a video with: mpv <path-from-above>"
echo ""
