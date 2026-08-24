#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export WANDB_MODE=disabled

echo "[1/4] Python"
python --version

echo "[2/4] GPU"
nvidia-smi || true

echo "[3/4] Dataset"
test -f data/math500_pilot.jsonl && wc -l data/math500_pilot.jsonl || { echo "Dataset missing. Run bash scripts/bootstrap_no_key.sh"; exit 2; }

echo "[4/4] No-key policy"
if [[ -n "${HF_TOKEN:-}" ]]; then echo "HF_TOKEN is set, but it is not required for the default public assets."; else echo "HF_TOKEN: not set (OK)"; fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then echo "WANDB_API_KEY is set, but cloud tracking is disabled by default."; else echo "WANDB_API_KEY: not set (OK)"; fi

echo "Preflight PASS"
