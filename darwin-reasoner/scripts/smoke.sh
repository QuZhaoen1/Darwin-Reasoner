#!/usr/bin/env bash
set -euo pipefail

# Allow smoke testing directly from a fresh clone, even before editable install.
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

RUN_ROOT="${RUN_ROOT:-runs/smoke_e2e}"
rm -rf "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"

python -m darwin_reasoner.cli baselines \
  --config configs/e0_smoke.yaml \
  --run-dir "${RUN_ROOT}/baselines"

python -m darwin_reasoner.cli collect \
  --config configs/e0_smoke.yaml \
  --run-dir "${RUN_ROOT}/counterfactual"

python -m darwin_reasoner.cli train-world-model \
  --config configs/e0_smoke.yaml \
  --data "${RUN_ROOT}/counterfactual/raw_results.jsonl" \
  --output "${RUN_ROOT}/world_model.pt"

python -m darwin_reasoner.cli mine-macros \
  --config configs/e0_smoke.yaml \
  --data "${RUN_ROOT}/counterfactual/raw_results.jsonl" \
  --output "${RUN_ROOT}/macro_candidates.json"

python -m darwin_reasoner.cli validate-macros \
  --config configs/e0_smoke.yaml \
  --candidates "${RUN_ROOT}/macro_candidates.json" \
  --output "${RUN_ROOT}/admitted_macros.json" \
  --run-dir "${RUN_ROOT}/macro_validation"

python -m darwin_reasoner.cli run \
  --config configs/e0_smoke.yaml \
  --world-model "${RUN_ROOT}/world_model.pt" \
  --macros "${RUN_ROOT}/admitted_macros.json" \
  --run-dir "${RUN_ROOT}/darwin"

python scripts/build_paper_results.py --runs "${RUN_ROOT}" --out "${RUN_ROOT}/paper_results"

echo "Smoke pipeline PASS: ${RUN_ROOT}"
