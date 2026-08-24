#!/usr/bin/env bash
set -euo pipefail

PIPELINE_CONFIG="${1:-configs/pipeline_iclr.yaml}"
SWEEP_CONFIG="${2:-configs/full_sweep.yaml}"

bash scripts/check_server.sh
python -m darwin_reasoner.cli pipeline --config "${PIPELINE_CONFIG}"

WORK_DIR=$(python - <<PY
import yaml
from pathlib import Path
spec=yaml.safe_load(Path('${PIPELINE_CONFIG}').read_text())
print(spec.get('work_dir','runs/iclr_pipeline'))
PY
)

DECISION=$(python - <<PY
import json
from pathlib import Path
p=Path('${WORK_DIR}')/'pipeline_status.json'
print(json.loads(p.read_text())['pilot_gate']['decision'])
PY
)

echo "Pilot decision: ${DECISION}"

if [[ "${DECISION}" == "continue_full_sweep" ]]; then
  if [[ -f "${SWEEP_CONFIG}" ]]; then
    python -m darwin_reasoner.cli sweep --config "${SWEEP_CONFIG}"
    python scripts/build_paper_results.py --runs runs --out paper_results
  else
    echo "Pilot passed, but ${SWEEP_CONFIG} does not exist."
    echo "Copy configs/full_sweep.example.yaml to ${SWEEP_CONFIG}, edit dataset/model paths, then rerun."
  fi
elif [[ "${DECISION}" == "diagnostics_only" ]]; then
  echo "Pilot is inconclusive. Run ablations/diagnostics before the expensive sweep."
else
  echo "Pilot failed the configured gate. Full sweep intentionally skipped to protect GPU time."
fi
