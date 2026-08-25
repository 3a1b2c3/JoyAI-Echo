#!/bin/bash
# JoyAI-Echo: Download models - WORKING VERSION
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "JoyAI-Echo - Download Models (Working)"
echo "================================================================================"
echo ""

CKPT_DIR="$REPO_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

echo "Target: $CKPT_DIR"
echo ""

# ============================================================================
# Check/Install tools
# ============================================================================
echo "[Setup] Checking tools..."

if ! command -v python &>/dev/null; then
    echo "ERROR: Python not found"
    exit 1
fi

if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface-hub..."
    python -m pip install --quiet huggingface-hub
fi

# ============================================================================
# Download with Python (more flexible than CLI)
# ============================================================================
echo ""
echo "[Download] Models (this may take 5-20 minutes)..."
echo ""

python << 'PYEOF'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

ckpt_dir = Path(os.environ['CKPT_DIR'])
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Model configurations to try (in order)
models = [
    {
        "name": "Echo Main Model",
        "repo": "jdopensource/JoyAI-Echo",
        "file": "echo-longvideo-release.safetensors",
        "local_path": ckpt_dir / "echo-longvideo-release.safetensors",
    },
    {
        "name": "Text Encoder (Gemma)",
        "repo": "google/gemma-2-12b",
        "file": None,  # Download entire repo
        "local_path": ckpt_dir / "gemma-3-12b",
    },
]

for i, model in enumerate(models, 1):
    print(f"\n[{i}/{len(models)}] {model['name']}")
    print(f"    Repo: {model['repo']}")

    local_path = model['local_path']

    # Check if already exists
    if model['file']:
        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024**2)
            print(f"    ✓ Already exists ({size_mb:.1f} MB)")
            continue
    else:
        if local_path.exists() and list(local_path.glob('*')):
            size_mb = sum(f.stat().st_size for f in local_path.rglob('*')) / (1024**2)
            print(f"    ✓ Already exists ({size_mb:.1f} MB)")
            continue

    # Download
    try:
        print(f"    Downloading...")
        if model['file']:
            path = hf_hub_download(
                repo_id=model['repo'],
                filename=model['file'],
                local_dir=ckpt_dir,
                cache_dir=None,
            )
            size_mb = Path(path).stat().st_size / (1024**2)
            print(f"    ✓ Downloaded ({size_mb:.1f} MB)")
        else:
            path = snapshot_download(
                repo_id=model['repo'],
                local_dir=local_path,
                local_dir_use_symlinks=False,
            )
            size_mb = sum(f.stat().st_size for f in Path(path).rglob('*')) / (1024**2)
            print(f"    ✓ Downloaded ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        if i == 1:  # First model is required
            print()
            print("    ERROR: Could not download main model")
            print("    This might mean:")
            print("    - No internet connection")
            print("    - Model repo doesn't exist or is private")
            print("    - HuggingFace is temporarily down")
            print()
            print("    Manual fix:")
            print("    1. Visit: https://huggingface.co/jdopensource")
            print("    2. Search for 'echo-longvideo' or 'joyai-echo'")
            print("    3. Download the model file")
            print(f"    4. Place in: {ckpt_dir}/")
            exit(1)
        else:
            print(f"    (Optional - will use fallback)")

print("\n" + "="*80)
print("✓ Download complete!")
print("="*80)
print()

# Show what's there
print("Checkpoints directory contents:")
for item in sorted(ckpt_dir.iterdir()):
    if item.is_file():
        size_mb = item.stat().st_size / (1024**2)
        print(f"  {item.name:50} ({size_mb:8.1f} MB)")
    else:
        size_mb = sum(f.stat().st_size for f in item.rglob('*')) / (1024**2)
        print(f"  {item.name:50} ({size_mb:8.1f} MB)")

PYEOF

echo ""
echo "Ready to run: bash run_examples.sh"
echo ""
