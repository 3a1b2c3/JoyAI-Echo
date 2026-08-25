#!/usr/bin/env python3
"""Run a checked-in single-image WM case through inference_wm.py."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "examples" / "wm_cases"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "echo-wm-base.safetensors"


def _case_dirs() -> list[Path]:
    return sorted(p for p in CASES.iterdir() if p.is_dir() and (p / "case.json").is_file())


def _load_case(case_dir: Path) -> dict:
    data = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    required = ("prompt", "action")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{case_dir}/case.json missing fields: {', '.join(missing)}")
    data["name"] = case_dir.name
    data["image"] = data.get("image", "input.png")
    data["action_str"] = data["action"]
    data["fov_deg"] = data.get("fov_deg", 70.0)
    image = case_dir / data["image"]
    if not image.is_file():
        raise FileNotFoundError(f"Case image not found: {image}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gemma-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "wm_cases")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--action-overlay", action="store_true")
    args = parser.parse_args()
    if args.list:
        for case_dir in _case_dirs():
            data = _load_case(case_dir)
            print(f"{case_dir.name}\t{data.get('description', 'single-image I2V case')}")
        return
    if args.case is None:
        parser.error("--case is required unless --list is used")
    case_dir = args.case if args.case.is_absolute() else ROOT / args.case
    case_dir = case_dir.resolve()
    data = _load_case(case_dir)
    output = args.output_dir / data["name"] / "result.mp4"
    command = [
        sys.executable, str(ROOT / "inference_wm.py"),
        "--image", str(case_dir / data["image"]),
        "--prompt", data["prompt"], "--action-str", data["action_str"],
        "--fov-deg", str(data["fov_deg"]), "--output", str(output),
    ]
    if args.checkpoint:
        command += ["--checkpoint", str(args.checkpoint)]
    if args.gemma_path:
        command += ["--gemma-path", str(args.gemma_path)]
    if args.action_overlay:
        command += ["--action-overlay"]
    if "seed" in data:
        command += ["--seed", str(data["seed"])]
    for flag, value in (("--num-frames", args.num_frames), ("--width", args.width),
                        ("--height", args.height), ("--steps", args.steps)):
        if value is not None:
            command += [flag, str(value)]
    print(" ".join(shlex.quote(part) for part in command))
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
