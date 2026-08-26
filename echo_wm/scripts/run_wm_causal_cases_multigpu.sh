#!/usr/bin/env bash
# Run checked-in causal WM cases across the available GPUs.
#
# Cases are assigned round-robin. Each GPU processes its assigned cases
# sequentially, so at most one inference process is active per GPU.
#
# Environment:
#   GPU_LIST        comma-separated GPU indices (default: 0,1,2)
#   CASES           space/comma-separated case names (default: every case dir)
#   PYTHON_BIN      interpreter to use (default: python from the active environment)
#   ACTION_OVERLAY  write the HUD copies (default: 1; set to 0 to skip them)
#
# Positional: [checkpoint] [gemma_path] [output_root]
set -euo pipefail

wm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint="${1:-$wm_root/checkpoints/echo-wm-flash.safetensors}"
gemma_path="${2:-$wm_root/checkpoints/gemma-3}"
output_root="${3:-$wm_root/outputs/wm_causal_cases_multigpu}"
gpu_list="${GPU_LIST:-0,1,2}"
python_bin="${PYTHON_BIN:-python}"
case "${ACTION_OVERLAY-1}" in
  0|false|no|off|"") overlay_flag="--no-action-overlay" ;;
  *)                 overlay_flag="--action-overlay" ;;
esac

IFS=',' read -r -a gpus <<< "$gpu_list"

if [[ -n "${CASES:-}" ]]; then
  IFS=', ' read -r -a cases <<< "$CASES"
else
  cases=()
  for case_dir in "$wm_root"/examples/wm_causal_cases/*/; do
    [[ -f "$case_dir/case.json" ]] && cases+=("$(basename "$case_dir")")
  done
fi
if (( ${#cases[@]} == 0 )); then
  echo "No cases found under $wm_root/examples/wm_causal_cases" >&2
  exit 2
fi

echo "Cases:  ${cases[*]}"
echo "GPUs:   ${gpus[*]}"
echo "Output: $output_root"
mkdir -p "$output_root"

run_share() {
  local gpu="$1" start="$2" stride="$3" rc=0
  for (( i = start; i < ${#cases[@]}; i += stride )); do
    local case_name="${cases[$i]}"
    local case_output="$output_root/$case_name"
    mkdir -p "$case_output"
    echo "[$case_name] -> GPU $gpu"
    if (
      cd "$wm_root"
      CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" scripts/run_wm_case_causal.py \
        --case "examples/wm_causal_cases/$case_name" \
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
