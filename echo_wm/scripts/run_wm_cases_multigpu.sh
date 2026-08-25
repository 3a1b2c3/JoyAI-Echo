#!/usr/bin/env bash
# Run WM cases across the GPUs you have.
#
# Cases are split round-robin across GPU_LIST, and each GPU works through its own
# share one at a time. The GPU count does not have to match the case count: with
# 3 cases on 2 GPUs, GPU 0 runs cases 1 and 3 in sequence while GPU 1 runs case 2.
# Only one inference process per GPU is ever live, which also keeps host memory use
# bounded (loading a checkpoint is memory-hungry).
#
# Environment:
#   GPU_LIST        comma-separated GPU indices (default: 0,1,2)
#   CASES           space/comma-separated case names (default: every case dir found)
#   PYTHON_BIN      interpreter to use (default: python3)
#   ACTION_OVERLAY  set to any value to also write the HUD copies
#
# Positional: [checkpoint] [gemma_path] [output_root]
set -euo pipefail

wm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint="${1:-$wm_root/checkpoints/echo-wm-base.safetensors}"
gemma_path="${2:-$wm_root/checkpoints/gemma-3}"
output_root="${3:-$wm_root/outputs/wm_cases_multigpu}"
gpu_list="${GPU_LIST:-0,1,2}"
python_bin="${PYTHON_BIN:-python3}"
overlay_flag="${ACTION_OVERLAY:+--action-overlay}"

IFS=',' read -r -a gpus <<< "$gpu_list"

# Default to every checked-in case; override with CASES="0004 0009".
if [[ -n "${CASES:-}" ]]; then
  IFS=', ' read -r -a cases <<< "$CASES"
else
  cases=()
  for case_dir in "$wm_root"/examples/wm_cases/*/; do
    [[ -f "$case_dir/case.json" ]] && cases+=("$(basename "$case_dir")")
  done
fi
if (( ${#cases[@]} == 0 )); then
  echo "No cases found under $wm_root/examples/wm_cases" >&2
  exit 2
fi

echo "Cases:  ${cases[*]}"
echo "GPUs:   ${gpus[*]}"
echo "Output: $output_root"
mkdir -p "$output_root"

# Walk this GPU's share of the case list, one case at a time.
run_share() {
  local gpu="$1" start="$2" stride="$3" rc=0
  for (( i = start; i < ${#cases[@]}; i += stride )); do
    local case_name="${cases[$i]}"
    local case_output="$output_root/$case_name"
    mkdir -p "$case_output"
    echo "[$case_name] -> GPU $gpu"
    if (
      cd "$wm_root"
      CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" scripts/run_wm_case.py \
        --case "examples/wm_cases/$case_name" \
        --checkpoint "$checkpoint" \
        --gemma-path "$gemma_path" \
        $overlay_flag \
        --output-dir "$output_root"
    ) >"$case_output/run_gpu${gpu}.log" 2>&1; then
      echo "[$case_name] completed (GPU $gpu)"
    else
      echo "[$case_name] failed; see $case_output/run_gpu${gpu}.log" >&2
      rc=1
    fi
  done
  return "$rc"
}

pids=()
for idx in "${!gpus[@]}"; do
  run_share "${gpus[$idx]}" "$idx" "${#gpus[@]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

exit "$status"
