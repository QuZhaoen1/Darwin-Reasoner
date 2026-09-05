#!/bin/bash
#SBATCH --job-name=dr_b
#SBATCH --account=yanggrp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100-01,gpu-v100-02,gpu-v100-03,gpu-v100-04
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner/logs/%x-%j.out

# Phase B - the barrier. Merge every batch, then train and evaluate the world
# model and mine macro candidates on the pooled data. No LLM calls happen here,
# so it is minutes, not hours; it takes a GPU only because the hash-MLP ensemble
# trains faster on one.
#
# Submit with --dependency=afterany:<A2 array id>, not afterok: if one batch
# failed we still want this to run and SAY SO, rather than never starting and
# leaving the pipeline silently stalled.

set -uo pipefail
source /deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner/darwin-reasoner/scripts/deac/env.sh
cd $REPO

N=${N_BATCHES:-10}
SIZE=${BATCH_SIZE:-10}
R=runs/pilot
C=configs/batches/batch_00.yaml   # batch configs are identical except dataset_path
FORCE=${FORCE_INCOMPLETE:-0}

echo "=== B | $(hostname) | $(date +%F' '%H:%M:%S) | batches=$N size=$SIZE ==="

echo; echo "--- merge E1 ---"
$PY scripts/deac/merge_batches.py --run-root $R --stage baselines \
    --expect-batches $N --expect-rows-per-batch $((SIZE * 6))
echo; echo "--- merge E2 ---"
$PY scripts/deac/merge_batches.py --run-root $R --stage counterfactual \
    --expect-batches $N --expect-rows-per-batch $((SIZE * 24))

# A world model trained on a silently truncated atlas produces plausible numbers
# that mean nothing, so gate on completeness unless explicitly overridden.
complete=$($PY -c "import json;print(json.load(open('$R/counterfactual_MERGE_INFO.json'))['complete'])")
echo; echo "E2 complete: $complete  (FORCE_INCOMPLETE=$FORCE)"
if [ "$complete" != "True" ] && [ "$FORCE" != "1" ]; then
  echo "ABORT: counterfactual atlas is incomplete. Resubmit the missing batches"
  echo "       (see batches_missing in $R/counterfactual_MERGE_INFO.json), then rerun."
  echo "       Override deliberately with FORCE_INCOMPLETE=1 if a partial atlas is wanted."
  exit 3
fi

# Downstream phases chain off this job with --dependency=afterok, so a step that
# crashes MUST make the job exit non-zero. Echoing $? and continuing would let
# C1 start against a missing world model or an empty macro list.
rc=0

echo; echo "--- E3 train world model ---"
$PY -m darwin_reasoner.cli train-world-model --config $C \
    --data $R/counterfactual_merged.jsonl --output $R/world_model.pt
s=$?; echo "exit=$s"; [ $s -eq 0 ] || rc=$s

echo; echo "--- E3 evaluate (the pre-registered gate numbers) ---"
# Takes --config and retrains internally on a state-grouped 80/20 split; it does
# NOT load world_model.pt, so the gate numbers are held-out by construction.
$PY scripts/evaluate_world_model.py --config $C --data $R/counterfactual_merged.jsonl \
    --output $R/world_model_eval.json
s=$?; echo "exit=$s"; [ $s -eq 0 ] || rc=$s

echo; echo "--- E9 mine macros ---"
$PY -m darwin_reasoner.cli mine-macros --config $C \
    --data $R/counterfactual_merged.jsonl --output $R/macro_candidates.json
s=$?; echo "exit=$s"; [ $s -eq 0 ] || rc=$s

echo; echo "=== WORLD MODEL EVAL ==="
[ -f $R/world_model_eval.json ] && cat $R/world_model_eval.json
echo; echo "=== MACRO CANDIDATES ==="
[ -f $R/macro_candidates.json ] && $PY -c "
import json; d=json.load(open('$R/macro_candidates.json'))
print(f'{len(d)} candidates' if isinstance(d,list) else json.dumps(d, indent=2)[:1500])"
echo "=== B DONE $(date +%H:%M:%S) | rc=$rc ==="
exit $rc
