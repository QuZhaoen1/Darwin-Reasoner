#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
# Install PyTorch from the server's CUDA-compatible index/module first if your cluster requires it.
# If torch is already present in the environment, pip will keep a compatible installed version.
pip install -e '.[gpu,train]'
python -m pip check
printf '\nEnvironment installed. Next: source .venv/bin/activate && bash scripts/check_8xa100.sh\n'
