#!/bin/bash
#SBATCH --job-name=dr_a2
#SBATCH --account=yanggrp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100-01,gpu-v100-02,gpu-v100-03,gpu-v100-04
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner/logs/%x-%A_%a.out

# Phase A2 - E2 matched counterfactual collection, one array task per batch.
# 10 problems x 14.6 min = ~146 min + ~3 min engine load, against a 240 min wall
# (~60% headroom). Do NOT raise the batch size without re-measuring: `cli
# collect` cannot resume (experiment.py:215 re-executes every rollout before
# checking is_completed), so a task that overruns the wall loses the whole batch.

set -uo pipefail
source /deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner/darwin-reasoner/scripts/deac/env.sh
cd $REPO

K=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
C=configs/batches/batch_${K}.yaml
OUT=runs/pilot/batch_${K}/counterfactual

echo "=== A2 batch ${K} | $(hostname) | $(date +%F' '%H:%M:%S) ==="
nvidia-smi --query-gpu=name --format=csv,noheader
test -f "$C" || { echo "FATAL: missing $C -- run make_batches.py first"; exit 2; }
grep -E '^(experiment_name|dataset_path|limit)' "$C"

# A partially written batch is worse than none: the merge would count it as
# present and the gap would be invisible. Start clean every time.
rm -rf "$OUT"

$PY -m darwin_reasoner.cli collect --config "$C" --run-dir "$OUT" 2>&1 \
  | grep -vE 'Processed prompts|Capturing CUDA graphs|Loading safetensors'
rc=${PIPESTATUS[0]}

echo "=== A2 batch ${K} | exit=$rc | $(date +%H:%M:%S) ==="
if [ -f "$OUT/raw_results.jsonl" ]; then
  n=$(wc -l < $OUT/raw_results.jsonl)
  echo "rows: $n  (expect 240 = 10 problems x execute_top_k 8 x repeats 3)"
fi
exit $rc
