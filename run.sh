#!/usr/bin/env bash
set -euo pipefail

# libgomp does not accept 0 or negative values; use a safe default.
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
	export OMP_NUM_THREADS=4
fi

# Prefer difflow env python for reproducibility.
DEFAULT_PYTHON="/root/miniconda3/envs/difflow/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
	PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
	PYTHON_BIN="${PYTHON_BIN:-python}"
fi

echo "Using Python: ${PYTHON_BIN}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

# Help reduce CUDA memory fragmentation for large UNet+Mamba training.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" train_mamba_recon.py --config args/args_mamba_low.json
"${PYTHON_BIN}" train_mamba_recon.py --config args/args_mamba_medium.json