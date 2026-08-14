#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/diffusionad/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

if [[ -n "${TORCHRUN_BIN:-}" ]]; then
  LAUNCH_CMD=("$TORCHRUN_BIN")
else
  LAUNCH_CMD=("$PYTHON_BIN" -m torch.distributed.run)
fi

BASE_CFG="${BASE_CFG:-args/args_recon_teacher_student_20260727.json}"
NUM_GPUS="${NUM_GPUS:-4}"
CFG_DIR="${CFG_DIR:-args/loss_ablation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_recon_loss_ablation}"
LOG_DIR="$OUTPUT_ROOT/logs"
MANIFEST="$OUTPUT_ROOT/ablation_manifest.csv"
RAW_SUMMARY_CSV="$OUTPUT_ROOT/ablation_summary_raw.csv"
GROUP_SUMMARY_CSV="$OUTPUT_ROOT/ablation_summary.csv"
SUMMARY_CSV="$GROUP_SUMMARY_CSV"

# Safe default: generate configs and summary skeleton only.
# Set EXECUTE=1 to actually run all ablations.
EXECUTE="${EXECUTE:-0}"

# Multi-seed validation (comma-separated), e.g. "42,123,3407"
SEEDS="${SEEDS:-42}"

# Optional fast-debug overrides
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"
EVAL_INTERVAL_OVERRIDE="${EVAL_INTERVAL_OVERRIDE:-}"
CLASS_SET="${CLASS_SET:-}"
ARG_PREFIX="${ARG_PREFIX:-recon_ablation}"

# Comma-separated list of leave-one-out ablations.
# You may add pseudo-keys: residual_branch, seg_branch, structure_branch
ABLATE_KEYS="${ABLATE_KEYS:-kd_recon_w,kd_res_w,kd_edge_w,feature_kd_w,anomaly_kd_w,anomaly_ms_kd_w,normal_suppress_w,hard_negative_w,student_recon_w,student_structure_w,seg_supervise_w,seg_kd_w,residual_branch,seg_branch,structure_branch}"

if [[ ! -f "$BASE_CFG" ]]; then
  echo "[ERROR] base config not found: $BASE_CFG"
  exit 1
fi

mkdir -p "$CFG_DIR" "$LOG_DIR" "$OUTPUT_ROOT"

echo "run_id,seed,ablated_key,arg_num,config_path,output_path" > "$MANIFEST"

generate_cfg() {
  local run_id="$1"
  local seed_value="$2"
  local ablated_key="$3"
  local cfg_path="$CFG_DIR/${ARG_PREFIX}_${run_id}.json"
  local arg_num="${ARG_PREFIX}_${run_id}"
  local out_path="$OUTPUT_ROOT/$arg_num"

  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

base_cfg = Path("$BASE_CFG")
out_cfg = Path("$cfg_path")

cfg = json.loads(base_cfg.read_text())
cfg["arg_num"] = "$arg_num"
cfg["output_path"] = "$out_path"
cfg["seed"] = int("$seed_value")

if "$EPOCHS_OVERRIDE":
    cfg["EPOCHS"] = int("$EPOCHS_OVERRIDE")
    cfg["student_epochs"] = int("$EPOCHS_OVERRIDE")
if "$EVAL_INTERVAL_OVERRIDE":
    cfg["eval_interval"] = int("$EVAL_INTERVAL_OVERRIDE")
if "$CLASS_SET":
    cfg["selected_classes"] = [x.strip() for x in "$CLASS_SET".split(",") if x.strip()]

ablated_key = "$ablated_key"
if ablated_key and ablated_key != "none":
    if ablated_key == "residual_branch":
        cfg["anomaly_fusion_residual_w"] = 0.0
    elif ablated_key == "seg_branch":
        cfg["student_use_segmentation_head"] = False
        cfg["use_segmentation_head"] = False
        cfg["seg_supervise_w"] = 0.0
        cfg["seg_kd_w"] = 0.0
        cfg["anomaly_fusion_seg_w"] = 0.0
    elif ablated_key == "structure_branch":
        cfg["use_structure_branch"] = False
        cfg["kd_edge_w"] = 0.0
        cfg["student_structure_w"] = 0.0
        cfg["anomaly_fusion_structure_w"] = 0.0
    else:
        cfg[ablated_key] = 0.0

out_cfg.parent.mkdir(parents=True, exist_ok=True)
out_cfg.write_text(json.dumps(cfg, indent=4))
PY

  echo "$run_id,$seed_value,$ablated_key,$arg_num,$cfg_path,$out_path" >> "$MANIFEST"
}

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=',' read -r -a LOSS_KEYS_ARRAY <<< "$ABLATE_KEYS"

for seed_item in "${SEED_ARRAY[@]}"; do
  seed_trimmed="$(echo "$seed_item" | xargs)"
  [[ -z "$seed_trimmed" ]] && continue

  generate_cfg "full_s${seed_trimmed}" "$seed_trimmed" "none"

  for key in "${LOSS_KEYS_ARRAY[@]}"; do
    key_trimmed="$(echo "$key" | xargs)"
    [[ -z "$key_trimmed" ]] && continue
    run_id="no_${key_trimmed}_s${seed_trimmed}"
    generate_cfg "$run_id" "$seed_trimmed" "$key_trimmed"
  done
done

echo "[INFO] manifest saved: $MANIFEST"

if [[ "$EXECUTE" == "1" ]]; then
  echo "[INFO] EXECUTE=1, start running ablations..."
  while IFS=',' read -r run_id seed_value ablated_key arg_num config_path out_path; do
    [[ "$run_id" == "run_id" ]] && continue
    echo "============================================================"
    echo "[RUN] $run_id | seed=$seed_value | ablated_key=$ablated_key"
    log_file="$LOG_DIR/${run_id}.log"
    "${LAUNCH_CMD[@]}" --nproc_per_node="$NUM_GPUS" train_recon_teacher_student.py --config "$config_path" 2>&1 | tee "$log_file"
  done < "$MANIFEST"
else
  echo "[INFO] EXECUTE=0, only generated configs. Set EXECUTE=1 to run training."
fi

echo "[INFO] building ablation summary..."

"$PYTHON_BIN" - <<PY
import json
import pandas as pd
import numpy as np
from pathlib import Path

manifest = Path("$MANIFEST")
raw_out = Path("$RAW_SUMMARY_CSV")
group_out = Path("$GROUP_SUMMARY_CSV")

m = pd.read_csv(manifest)
metrics = [
    "Image-AUROC",
    "Pixel-AUROC",
    "Image-AP",
    "Pixel-AP",
    "Image-F1",
    "Pixel-F1",
    "Eval-FPS",
    "Eval-ms-per-img",
]

rows = []
for _, r in m.iterrows():
    run_id = r["run_id"]
    seed_value = r["seed"]
    arg_num = r["arg_num"]
    out_path = Path(r["output_path"])
    cfg_path = Path(r["config_path"])
    summary_csv = out_path / f"metrics/ARGS={arg_num}/200_400t_1_MVTec_image_pixel_auroc_train_recon_teacher_student.csv"

    expected_classes = []
    if cfg_path.exists():
        try:
            cfg_obj = json.loads(cfg_path.read_text())
            maybe_classes = cfg_obj.get("selected_classes", [])
            if isinstance(maybe_classes, list):
                expected_classes = [str(x) for x in maybe_classes]
        except Exception:
            expected_classes = []

    row = {
        "run_id": run_id,
        "seed": seed_value,
        "ablated_key": r["ablated_key"],
        "arg_num": arg_num,
        "config_path": str(cfg_path),
        "expected_class_count": int(len(expected_classes)),
        "summary_csv": str(summary_csv),
        "status": "ok" if summary_csv.exists() else "missing_summary",
    }

    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        if len(df) > 0 and "classname" in df.columns:
            present_classes = sorted(df["classname"].dropna().astype(str).unique().tolist())
            row["present_class_count"] = int(len(present_classes))
            row["present_classes"] = "|".join(present_classes)

            if len(expected_classes) > 0:
                missing = [c for c in expected_classes if c not in present_classes]
                row["missing_class_count"] = int(len(missing))
                row["missing_classes"] = "|".join(missing)
                if len(missing) > 0:
                    row["status"] = "partial_summary"

            last = df.drop_duplicates("classname", keep="last")
            for metric in metrics:
                row[metric] = pd.to_numeric(last[metric], errors="coerce").mean(skipna=True) if metric in last.columns else np.nan
        else:
            row["status"] = "bad_summary"

    rows.append(row)

s = pd.DataFrame(rows)
available_metrics = [metric for metric in metrics if metric in s.columns]

if len(s) > 0 and len(available_metrics) > 0:
  ref = s[(s["ablated_key"] == "none") & (s["status"] == "ok")][["seed"] + available_metrics].drop_duplicates("seed", keep="last")
  if len(ref) > 0:
    ref = ref.set_index("seed")
    for metric in available_metrics:
      deltas = []
      for _, rr in s.iterrows():
        seed_key = rr.get("seed")
        if seed_key in ref.index and pd.notna(rr.get(metric)) and pd.notna(ref.at[seed_key, metric]):
          deltas.append(float(rr[metric]) - float(ref.at[seed_key, metric]))
        else:
          deltas.append(np.nan)
      s[f"delta_{metric}"] = deltas

raw_out.parent.mkdir(parents=True, exist_ok=True)
s.to_csv(raw_out, index=False)

planned = m.groupby("ablated_key", dropna=False).size().rename("planned_runs").reset_index()
finished = s[s["status"] == "ok"].groupby("ablated_key", dropna=False).size().rename("finished_runs").reset_index()
grouped = planned.merge(finished, on="ablated_key", how="left")
grouped["finished_runs"] = grouped["finished_runs"].fillna(0).astype(int)

ok = s[s["status"] == "ok"].copy()
if len(ok) > 0:
    agg_rows = []
    for ablated_key, g in ok.groupby("ablated_key", dropna=False):
        row = {"ablated_key": ablated_key}
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce") if metric in g.columns else pd.Series(dtype=float)
            row[f"{metric}_mean"] = vals.mean(skipna=True)
            row[f"{metric}_std"] = vals.std(skipna=True, ddof=1)

            dcol = f"delta_{metric}"
            if dcol in g.columns:
                dvals = pd.to_numeric(g[dcol], errors="coerce")
                row[f"{dcol}_mean"] = dvals.mean(skipna=True)
                row[f"{dcol}_std"] = dvals.std(skipna=True, ddof=1)

        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows)
    grouped = grouped.merge(agg, on="ablated_key", how="left")

    if "delta_Pixel-AP_mean" in grouped.columns:
        grouped = grouped.sort_values("delta_Pixel-AP_mean", ascending=True)
    elif "delta_Pixel-AUROC_mean" in grouped.columns:
        grouped = grouped.sort_values("delta_Pixel-AUROC_mean", ascending=True)
    else:
        grouped = grouped.sort_values("ablated_key")
else:
    grouped = grouped.sort_values("ablated_key")

group_out.parent.mkdir(parents=True, exist_ok=True)
grouped.to_csv(group_out, index=False)

print("\n[ABLATION SUMMARY]")
if len(grouped) > 0:
    print(grouped.to_string(index=False))
print(f"\nSaved raw: {raw_out}")
print(f"Saved grouped: {group_out}")
PY

echo "[INFO] done"
