#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/diffusionad/bin/python}"
BASE_CFG="${BASE_CFG:-args/args_dualpath_baseline.json}"
NUM_GPUS="${NUM_GPUS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_dualpath_weight_grid}"
CFG_DIR="${CFG_DIR:-args/weight_grid}"
LOG_DIR="$OUTPUT_ROOT/logs"
DRY_RUN="${DRY_RUN:-0}"

# Optional overrides
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"
EVAL_INTERVAL_OVERRIDE="${EVAL_INTERVAL_OVERRIDE:-}"
CLASS_SET="${CLASS_SET:-}"

if [[ ! -f "$BASE_CFG" ]]; then
  echo "[ERROR] base config not found: $BASE_CFG"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi
# Keep launcher tied to the same Python env to avoid torch version mismatches.
if [[ -n "${TORCHRUN_BIN:-}" ]]; then
  LAUNCH_CMD=("$TORCHRUN_BIN")
else
  LAUNCH_CMD=("$PYTHON_BIN" -m torch.distributed.run)
fi

mkdir -p "$CFG_DIR" "$LOG_DIR"

# 4-run weight grid: keep test mode fixed to main_head, only tune loss weights.
declare -a RUN_IDS=("w1" "w2" "w3" "w4")
declare -a AUX_W=("1.0" "0.6" "0.4" "0.3")
declare -a MAIN_W=("1.0" "1.5" "2.0" "2.0")
declare -a KD_W=("0.2" "0.6" "1.0" "1.2")

echo "[INFO] base config: $BASE_CFG"
echo "[INFO] num gpus: $NUM_GPUS"
echo "[INFO] output root: $OUTPUT_ROOT"

generate_cfg() {
  local run_id="$1"
  local aux="$2"
  local main="$3"
  local kd="$4"
  local cfg_path="$CFG_DIR/dualpath_${run_id}.json"
  local arg_num="dualpath_${run_id}"
  local out_path="$OUTPUT_ROOT/$arg_num"

  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

base_cfg = Path("$BASE_CFG")
out_cfg = Path("$cfg_path")

d = json.loads(base_cfg.read_text())
d["arg_num"] = "$arg_num"
d["output_path"] = "$out_path"
d["lambda_aux"] = float("$aux")
d["lambda_main"] = float("$main")
d["lambda_kd"] = float("$kd")
d["test_anomaly_map_mode"] = "main_head"

if "$EPOCHS_OVERRIDE":
    d["EPOCHS"] = int("$EPOCHS_OVERRIDE")
if "$EVAL_INTERVAL_OVERRIDE":
    d["eval_interval"] = int("$EVAL_INTERVAL_OVERRIDE")
if "$CLASS_SET":
    d["selected_classes"] = [x.strip() for x in "$CLASS_SET".split(",") if x.strip()]

out_cfg.parent.mkdir(parents=True, exist_ok=True)
out_cfg.write_text(json.dumps(d, indent=4))
PY

  echo "$cfg_path"
}

for i in "${!RUN_IDS[@]}"; do
  run_id="${RUN_IDS[$i]}"
  aux="${AUX_W[$i]}"
  main="${MAIN_W[$i]}"
  kd="${KD_W[$i]}"

  echo "============================================================"
  echo "[RUN] $run_id | lambda_aux=$aux lambda_main=$main lambda_kd=$kd"
  cfg_file="$(generate_cfg "$run_id" "$aux" "$main" "$kd")"
  echo "[CFG] wrote $cfg_file"
  log_file="$LOG_DIR/dualpath_${run_id}.log"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] ${LAUNCH_CMD[*]} --nproc_per_node=$NUM_GPUS train_dualpath_baseline.py --config $cfg_file"
  else
    "${LAUNCH_CMD[@]}" --nproc_per_node="$NUM_GPUS" train_dualpath_baseline.py --config "$cfg_file" 2>&1 | tee "$log_file"
  fi

done

echo "============================================================"
echo "[INFO] all runs finished, building summary..."

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
import numpy as np
import pandas as pd

cfg_dir = Path("$CFG_DIR")
out_root = Path("$OUTPUT_ROOT")

rows = []
for cfg_path in sorted(cfg_dir.glob("dualpath_w*.json")):
    cfg = json.loads(cfg_path.read_text())
    arg_num = cfg["arg_num"]
    summary_csv = Path(cfg["output_path"]) / f"metrics/ARGS={arg_num}/200_400t_1_MVTec_image_pixel_auroc_train_dualpath.csv"

    row = {
        "run": arg_num,
        "lambda_aux": cfg.get("lambda_aux"),
        "lambda_main": cfg.get("lambda_main"),
        "lambda_kd": cfg.get("lambda_kd"),
        "status": "ok" if summary_csv.exists() else "missing_summary",
    }

    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        if "classname" in df.columns and not df.empty:
            last = df.drop_duplicates("classname", keep="last")
            for m in ["Image-AUROC", "Pixel-AUROC", "Image-AP", "Pixel-AP", "Image-F1", "Pixel-F1", "Eval-FPS"]:
                row[m] = pd.to_numeric(last[m], errors="coerce").mean(skipna=True) if m in last.columns else np.nan
        else:
            row["status"] = "bad_summary_format"

    rows.append(row)

summary = pd.DataFrame(rows)
summary_path = out_root / "grid_summary.csv"
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(summary_path, index=False)

print("\n[GRID SUMMARY]")
if not summary.empty:
    sort_key = "Image-AUROC" if "Image-AUROC" in summary.columns else "run"
    print(summary.sort_values(sort_key, ascending=False).to_string(index=False))
print(f"\nSaved: {summary_path}")
PY
