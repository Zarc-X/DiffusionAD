#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

run_one() {
  local cfg="$1"
  local tag="$2"
  local log_path="logs/${tag}.log"

  echo "[$(date '+%F %T')] START ${tag} (${cfg})" | tee -a "$log_path"
  conda run -n difflow python train_mamba_recon.py --config "$cfg" 2>&1 | tee -a "$log_path"
  echo "[$(date '+%F %T')] END ${tag}" | tee -a "$log_path"
}

run_one "args/args_mamba_low_ds16_8.json" "mamba_low_ds16_8"
run_one "args/args_mamba_low_dstate24.json" "mamba_low_dstate24"
run_one "args/args_mamba_low_unidir.json" "mamba_low_unidir"

echo "All three ablations finished."
