#!/usr/bin/env bash
set -euo pipefail

echo "=== DarwinReasoner server preflight ==="
echo "Host: $(hostname)"
echo "PWD : $(pwd)"
echo "Git : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Python: $(python --version 2>&1 || true)"

echo
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
else
  echo "WARNING: nvidia-smi not found. CPU/mock tests can still run."
fi

echo
python - <<'PY'
import importlib.util
mods = ['torch', 'vllm', 'yaml', 'networkx', 'sklearn']
for m in mods:
    print(f"{m:12s}: {'OK' if importlib.util.find_spec(m) else 'MISSING'}")
try:
    import torch
    print('torch cuda available:', torch.cuda.is_available())
    print('torch gpu count     :', torch.cuda.device_count())
except Exception as exc:
    print('torch inspection failed:', exc)
PY
