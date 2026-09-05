#!/bin/bash
#SBATCH --job-name=dr_c1
#SBATCH --account=yanggrp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100-01,gpu-v100-02,gpu-v100-03,gpu-v100-04
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner/logs/%x-%A_%a.out

# Phase C1 - E9 matched macro-vs-expanded validation, ONE ARRAY TASK PER MACRO.
#
# Why sharded by macro and NOT by problem batch:
#   experiment.py:478 calls evaluate_macro_admission on all rows for ONE macro at
#   once. The admission decision is a paired bootstrap CI over the macro's 50
#   problem pairs, so the 50 pairs of a given macro must live in one process.
#   Sharding by batch would give ten independent n=10 decisions per macro and ten
#   conflicting admitted.json files. Sharding by macro is exact: macros never
#   interact inside validate_macros, so per-macro shards reproduce the pooled
#   result (only config.seed + macro_index*1e6 changes, and pairing is preserved
#   within each macro, which is all the test uses).
#
# Why one job could NOT do all 8:
#   8 macros x 50 problems x (1 macro call + L expanded calls), L = motif length
#   in [2,5]. At the measured 7.2 s/call (650 real A1 calls, 4701 s) that is
#   3.8 h at L=3.5 and 5.0 h at L=5 -- past the 4 h backfill ceiling. Per macro
#   it is 19-38 min.
#
# validate_macros DOES resume, unlike collect: experiment.py:451 checks
# store.is_completed(pair_key) BEFORE both executes, and reconstructs the paired
# arrays from raw_results.jsonl after the loop (experiment.py:472). So a
# preempted/requeued task continues where it stopped -- do NOT rm -rf the run
# dir here the way a2_collect.sbatch has to.
#
# Submit (after B):
#   sbatch --dependency=afterok:<B jobid> --array=0-7%4 scripts/deac/c1_validate_macros.sbatch
# Array is 0-7 because extra.macro_validation_candidates=8 is the hard cap
# (experiment.py:420, candidates[:max_candidates]). Slots past the number of
# mined candidates no-op and exit 0.

set -uo pipefail
source /deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner/darwin-reasoner/scripts/deac/env.sh
cd $REPO

M=${SLURM_ARRAY_TASK_ID}
MM=$(printf "%02d" ${M})
CAND=runs/pilot/macro_candidates.json
OUT=runs/macroval/macro_${MM}

echo "=== C1 macro slot ${MM} | $(hostname) | $(date +%F' '%H:%M:%S) ==="
nvidia-smi --query-gpu=name --format=csv,noheader
test -f "$CAND" || { echo "FATAL: missing $CAND -- phase B (mine-macros) did not run"; exit 2; }

mkdir -p "$OUT"

# --- pick this task's single macro -------------------------------------------
$PY - "$CAND" "$M" "$OUT/candidates.json" <<'PY'
import json, sys
cands = json.loads(open(sys.argv[1], encoding="utf-8").read())
k = int(sys.argv[2])
cands = cands[:8]                     # extra.macro_validation_candidates
if k >= len(cands):
    print(f"NOOP: mine-macros produced {len(cands)} candidate(s); slot {k} is empty.")
    sys.exit(9)
m = cands[k]
open(sys.argv[3], "w", encoding="utf-8").write(json.dumps([m], indent=2))
print(f"macro slot {k}: name={m['name']} operators={m['operators']} "
      f"support={m.get('support')} lift={round(float(m.get('success_lift', 0.0)), 4)}")
print(f"expanded arm is {len(m['operators'])} calls, macro arm is 1 call "
      f"-> {50 * (1 + len(m['operators']))} generation calls for this task")
PY
rc=$?
if [ $rc -eq 9 ]; then
  echo "=== C1 macro slot ${MM} | nothing to do | exit=0 ==="
  exit 0
fi
if [ $rc -ne 0 ]; then
  echo "FATAL: could not split $CAND (exit $rc)"
  exit 2
fi

# --- pooled 100-problem config ------------------------------------------------
# validate_macros uses problems[:extra.macro_validation_problems] = problems[:50]
# (experiment.py:421,447). A per-batch config only holds 10 problems, so it would
# silently validate on 10. Pool all ten batch files back into problem order.
# Derive the config from batch_00.yaml, NOT configs/pilot_8b.yaml: pilot_8b.yaml
# predates the batch generation and lacks use_chat_template/enable_thinking and
# still has max_tokens_per_call 1536. Under bare completions Qwen3-8B never emits
# EOS and runs to the cap, which would make these numbers incomparable with E1/E2.
# Paths are per-task, so the 8 array tasks never write the same file.
POOL_D=$OUT/pooled_100.jsonl
POOL_C=$OUT/pooled_100.yaml
cat data/batches/batch_0[0-9].jsonl > "$POOL_D"
n_pool=$(wc -l < "$POOL_D")
echo "pooled dataset: $POOL_D ($n_pool problems; validate_macros uses the first 50)"
[ "$n_pool" -ge 50 ] || { echo "FATAL: pooled dataset has $n_pool < 50 problems"; exit 2; }

$PY - configs/batches/batch_00.yaml "$POOL_D" "$POOL_C" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
cfg["dataset_path"] = sys.argv[2]
cfg["limit"] = None
cfg["experiment_name"] = "pilot_macroval"
yaml.safe_dump(cfg, open(sys.argv[3], "w", encoding="utf-8"), sort_keys=False)
print(f"pooled config: {sys.argv[3]}  "
      f"max_tokens_per_call={cfg['backend']['max_tokens_per_call']} "
      f"use_chat_template={cfg['backend'].get('use_chat_template')} "
      f"enable_thinking={cfg['backend'].get('enable_thinking')} seed={cfg['seed']}")
PY
[ $? -eq 0 ] || { echo "FATAL: could not build pooled config"; exit 2; }

# --- run ----------------------------------------------------------------------
# NO rm -rf: this stage resumes correctly (see header).
$PY -m darwin_reasoner.cli validate-macros \
  --config "$POOL_C" \
  --candidates "$OUT/candidates.json" \
  --output "$OUT/admitted.json" \
  --run-dir "$OUT" 2>&1 \
  | grep -vE 'Processed prompts|Capturing CUDA graphs|Loading safetensors'
rc=${PIPESTATUS[0]}

echo "=== C1 macro slot ${MM} | exit=$rc | $(date +%H:%M:%S) ==="
if [ -f "$OUT/raw_results.jsonl" ]; then
  n=$(wc -l < $OUT/raw_results.jsonl)
  echo "pairs: $n  (expect 50 = macro_validation_problems)"
  if [ "$n" -lt 50 ]; then
    echo "SHORT: resubmit this slot alone with --array=${M}; it resumes from pair $n."
  fi
fi
[ -f "$OUT/admitted.json" ] && $PY -c "
import json; d=json.load(open('$OUT/admitted.json'))
print(f'admitted: {len(d)}')
for m in d: print('   ', m['name'], m['admission'])"
exit $rc