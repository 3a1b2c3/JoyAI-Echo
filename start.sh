#!/bin/bash
# JoyAI-Echo: pick a UI to launch (Echo-LongVideo Director Agent or Echo-WM Gradio)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "JoyAI-Echo - Start"
echo "================================================================================"
echo ""
echo "  1. Echo-LongVideo Director Agent  (multi-shot story planning, http://127.0.0.1:5187)"
echo "  2. Echo-WM Gradio demo            (interactive navigation, http://127.0.0.1:7860)"
echo ""

read -p "Choose UI (1-2): " choice

case "$choice" in
    1)
        PROJECT_DIR="$REPO_DIR/echo_longvideo"
        ;;
    2)
        PROJECT_DIR="$REPO_DIR/echo_wm"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo ""
    echo "ERROR: $PROJECT_DIR/.venv not found."
    echo "Run setup first: cd $(basename "$PROJECT_DIR") && bash setup_and_run.sh"
    exit 1
fi

echo ""
cd "$PROJECT_DIR"
exec bash run_ui.sh
