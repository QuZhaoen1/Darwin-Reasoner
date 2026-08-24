#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

bash scripts/check_8xa100.sh

if [[ ! -f data/math500_pilot.jsonl ]]; then
  python scripts/bootstrap_public_data.py --preset math500 --limit 200 --output data/math500_pilot.jsonl
fi

python scripts/collect_counterfactuals_8gpu.py \
  --config configs/pilot_8b.yaml \
  --gpus 8 \
  --run-root runs/pilot_counterfactual_8gpu

python scripts/build_policy_dataset.py \
  --input runs/pilot_counterfactual_8gpu/raw_results.jsonl \
  --train data/policy_train.jsonl \
  --valid data/policy_valid.jsonl

bash scripts/train_policy_8xa100.sh configs/training/policy_qwen3_8b_a100.yaml
