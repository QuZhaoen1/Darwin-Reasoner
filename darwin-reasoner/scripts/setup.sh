#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ "${DARWIN_GPU_INSTALL:-0}" == "1" ]]; then
  if [[ "${DARWIN_WANDB_INSTALL:-0}" == "1" ]]; then
    python -m pip install -e '.[gpu,tracking,dev]'
  else
    python -m pip install -e '.[gpu,dev]'
  fi
else
  python -m pip install -e '.[dev]'
fi

echo "Environment ready. Activate with: source ${VENV_DIR}/bin/activate"
echo "W&B is optional; default experiments log locally under runs/."
