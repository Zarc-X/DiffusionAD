#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/diffusionad/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

BASE_CFG="${BASE_CFG:-args/args_recon_teacher_student_20260727.json}"
NUM_GPUS="${NUM_GPUS:-4}"
EXECUTE="${EXECUTE:-0}"
SEEDS="${SEEDS:-42,123,3407}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"
EVAL_INTERVAL_OVERRIDE="${EVAL_INTERVAL_OVERRIDE:-}"
CLASS_SET="${CLASS_SET:-}"

# both | branches | losses
VALIDATION_MODE="${VALIDATION_MODE:-both}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_recon_two_stage_validation}"
CFG_ROOT="${CFG_ROOT:-args/two_stage_validation}"
ARG_PREFIX_BASE="${ARG_PREFIX_BASE:-recon_v2}"

BRANCH_KEYS="${BRANCH_KEYS:-residual_branch,structure_branch,seg_branch}"
LOSS_KEYS="${LOSS_KEYS:-kd_recon_w,kd_res_w,kd_edge_w,feature_kd_w,anomaly_kd_w,anomaly_ms_kd_w,normal_suppress_w,hard_negative_w,student_recon_w,student_structure_w,seg_supervise_w,seg_kd_w}"

if [[ ! -f "$BASE_CFG" ]]; then
  echo "[ERROR] base config not found: $BASE_CFG"
  exit 1
fi

if [[ ! -x "$SCRIPT_DIR/run_recon_loss_ablation.sh" ]]; then
  chmod +x "$SCRIPT_DIR/run_recon_loss_ablation.sh"
fi

mkdir -p "$OUTPUT_ROOT" "$CFG_ROOT"

run_task() {
  local task_name="$1"
  local keys="$2"
  local task_output="$OUTPUT_ROOT/$task_name"
  local task_cfg_dir="$CFG_ROOT/$task_name"
  local task_arg_prefix="${ARG_PREFIX_BASE}_${task_name}"

  echo "============================================================"
  echo "[TASK] $task_name"
  echo "[TASK] keys=$keys"
  echo "[TASK] output=$task_output"

  if [[ -n "${TORCHRUN_BIN:-}" ]]; then
    BASE_CFG="$BASE_CFG" \
    NUM_GPUS="$NUM_GPUS" \
    EXECUTE="$EXECUTE" \
    SEEDS="$SEEDS" \
    EPOCHS_OVERRIDE="$EPOCHS_OVERRIDE" \
    EVAL_INTERVAL_OVERRIDE="$EVAL_INTERVAL_OVERRIDE" \
    CLASS_SET="$CLASS_SET" \
    ARG_PREFIX="$task_arg_prefix" \
    ABLATE_KEYS="$keys" \
    CFG_DIR="$task_cfg_dir" \
    OUTPUT_ROOT="$task_output" \
    PYTHON_BIN="$PYTHON_BIN" \
    TORCHRUN_BIN="$TORCHRUN_BIN" \
    "$SCRIPT_DIR/run_recon_loss_ablation.sh"
  else
    BASE_CFG="$BASE_CFG" \
    NUM_GPUS="$NUM_GPUS" \
    EXECUTE="$EXECUTE" \
    SEEDS="$SEEDS" \
    EPOCHS_OVERRIDE="$EPOCHS_OVERRIDE" \
    EVAL_INTERVAL_OVERRIDE="$EVAL_INTERVAL_OVERRIDE" \
    CLASS_SET="$CLASS_SET" \
    ARG_PREFIX="$task_arg_prefix" \
    ABLATE_KEYS="$keys" \
    CFG_DIR="$task_cfg_dir" \
    OUTPUT_ROOT="$task_output" \
    PYTHON_BIN="$PYTHON_BIN" \
    "$SCRIPT_DIR/run_recon_loss_ablation.sh"
  fi
}

case "$VALIDATION_MODE" in
  both)
    run_task "branch_effectiveness" "$BRANCH_KEYS"
    run_task "loss_effectiveness" "$LOSS_KEYS"
    ;;
  branches)
    run_task "branch_effectiveness" "$BRANCH_KEYS"
    ;;
  losses)
    run_task "loss_effectiveness" "$LOSS_KEYS"
    ;;
  *)
    echo "[ERROR] invalid VALIDATION_MODE: $VALIDATION_MODE (use both|branches|losses)"
    exit 1
    ;;
esac

echo "============================================================"
echo "[INFO] building combined summary..."

"$PYTHON_BIN" - <<PY
import pandas as pd
from pathlib import Path

output_root = Path("$OUTPUT_ROOT")
combined_rows = []

for task_name in ["branch_effectiveness", "loss_effectiveness"]:
    summary_csv = output_root / task_name / "ablation_summary.csv"
    if not summary_csv.exists():
        continue

    df = pd.read_csv(summary_csv)
    if len(df) == 0:
        continue

    df.insert(0, "task", task_name)
    combined_rows.append(df)

combined_path = output_root / "combined_ablation_summary.csv"
if len(combined_rows) == 0:
    pd.DataFrame(columns=["task", "ablated_key", "planned_runs", "finished_runs"]).to_csv(combined_path, index=False)
    print(f"Saved: {combined_path} (empty)")
else:
    combined = pd.concat(combined_rows, axis=0, ignore_index=True)

    rank_cols = [
        "delta_Pixel-AP_mean",
        "delta_Pixel-AUROC_mean",
        "delta_Image-AP_mean",
        "delta_Image-AUROC_mean",
    ]

    sort_col = None
    for col in rank_cols:
        if col in combined.columns:
            sort_col = col
            break

    if sort_col is not None:
        combined = combined.sort_values(["task", sort_col], ascending=[True, True])
    else:
        combined = combined.sort_values(["task", "ablated_key"], ascending=[True, True])

    combined.to_csv(combined_path, index=False)
    print("\n[COMBINED SUMMARY]")
    print(combined.to_string(index=False))
    print(f"\nSaved: {combined_path}")
PY

echo "[INFO] done"
