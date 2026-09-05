#!/bin/bash
#SBATCH --job-name=dr_a1
#SBATCH --account=yanggrp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100-01,gpu-v100-02,gpu-v100-03,gpu-v100-04
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner/logs/%x-%A_%a.out

# Phase A1 - E1 baselines, one array task per batch.
# 10 problems x 4.1 min = ~41 min + ~3 min engine load, against a 90 min wall.
# All four V100 nodes are excluded: they are V100_32 with no bf16. The repo's
# own sbatch scripts in this project excluded only v100-01..03 and would land
# on v100-04 and die.

set -uo pipefail
source /deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner/darwin-reasoner/scripts/deac/env.sh
cd $REPO

K=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
C=configs/batches/batch_${K}.yaml
OUT=runs/pilot/batch_${K}/baselines

echo "=== A1 batch ${K} | $(hostname) | $(date +%F' '%H:%M:%S) ==="
nvidia-smi --query-gpu=name --format=csv,noheader
test -f "$C" || { echo "FATAL: missing $C -- run make_batches.py first"; exit 2; }
grep -E '^(experiment_name|dataset_path|limit)' "$C"

$PY -m darwin_reasoner.cli baselines --config "$C" --run-dir "$OUT" 2>&1 \
  | grep -vE 'Processed prompts|Capturing CUDA graphs|Loading safetensors'
rc=${PIPESTATUS[0]}

echo "=== A1 batch ${K} | exit=$rc | $(date +%H:%M:%S) ==="
if [ -f "$OUT/raw_results.jsonl" ]; then
  echo "rows: $(wc -l < $OUT/raw_results.jsonl)"
fi
exit $rc
