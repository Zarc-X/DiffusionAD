#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="${ENV_NAME:-difflow}"
LOG_DIR="${LOG_DIR:-logs}"
CLASS_SET="${CLASS_SET:-screw,transistor,leather,carpet,wood}"

mkdir -p "$LOG_DIR"

run_one() {
  local cfg="$1"
  local tag="$2"
  local log_path="$LOG_DIR/${tag}.log"

  echo "[$(date '+%F %T')] START ${tag} (${cfg})"
  echo "[$(date '+%F %T')] class_set=${CLASS_SET}" | tee -a "$log_path"

  conda run -n "$ENV_NAME" python train_mamba_recon.py \
    --config "$cfg" \
    --class-set "$CLASS_SET" 2>&1 | tee -a "$log_path"

  echo "[$(date '+%F %T')] END ${tag}" | tee -a "$log_path"
}

run_one "args/args_mamba_low_unidir_singlepath_tschedule_hard_w1.json" "tschedule_hard_w1"
run_one "args/args_mamba_low_unidir_singlepath_tschedule_hard_w2.json" "tschedule_hard_w2"
run_one "args/args_mamba_low_unidir_singlepath_tschedule_hard_w3.json" "tschedule_hard_w3"

echo "All hard-weight tschedule runs finished."
