#!/usr/bin/env bash
set -euo pipefail

# Zero-key bootstrap for a fresh GPU server.
# Public Qwen + public MATH-500; W&B disabled.

export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export WANDB_MODE=disabled

if [[ ! -d .venv ]]; then
  DARWIN_GPU_INSTALL=1 DARWIN_WANDB_INSTALL=0 bash scripts/setup.sh
fi
source .venv/bin/activate

python scripts/bootstrap_public_data.py \
  --preset math500 \
  --limit 200 \
  --output data/math500_pilot.jsonl

echo
echo "Zero-key assets are ready."
echo "Next: bash scripts/smoke.sh"
echo "Then: python -m darwin_reasoner.cli collect --config configs/pilot_8b.yaml --run-dir runs/pilot_counterfactual"
