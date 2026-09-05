#!/bin/bash
#SBATCH --job-name=dr_c2
#SBATCH --account=yanggrp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100-01,gpu-v100-02,gpu-v100-03,gpu-v100-04
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner/logs/%x-%A_%a.out

# Phase C2 - E4 the DarwinReasoner main loop, one array task per batch.
#
# Shardable exactly like A1/A2: run_darwin loops over problems from
# config.dataset_path and writes one row per problem (experiment.py:278-362).
# Nothing is pooled -- the world model and the macro set are read-only inputs.
#
# It also RESUMES, unlike collect: experiment.py:279-281 checks
# store.is_completed(f"darwin|{problem_id}|{seed}|{max_tokens}") at the top of the
# problem loop, before any generation. A preempted task re-does at most the one
# problem it was mid-way through. Hence NO rm -rf of the run dir.
#
# Cost per problem (extra.online_rounds=2, budget.max_tokens=8192, max_calls=16):
#   round 0 budget = max(128, 8192//2) = 4096 tokens. Measured cumulative tokens
#   per chained call are 795 / 2047 / 4140, so round 0 breaks entering node 4.
#   round 1 budget = remaining = 8192 - T0, but the executor carries used_tokens
#   into the check (executor.py:44,49), so round 1 runs iff T0 < 8192 - T0, i.e.
#   iff T0 < 4096. Deep first pick -> round 1 is 0 calls (~3 total); shallow first
#   pick -> round 1 continues to ~7397 cumulative tokens (~6 total). Hard ceiling
#   is budget.max_calls=16. At 7.2 s/call: 4-8 min typical, 20 min worst per
#   batch of 10, plus ~3-5 min engine load.
#
# Submit (after C1):
#   sbatch --dependency=afterany:<C1 array jobid> --array=0-9%3 scripts/deac/c2_darwin.sbatch
# afterany, not afterok: a macro slot that no-ops or fails must not stall E4.

set -uo pipefail
source /deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner/darwin-reasoner/scripts/deac/env.sh
cd $REPO

K=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
C=configs/batches/batch_${K}.yaml
OUT=runs/pilot/batch_${K}/darwin
WM=runs/pilot/world_model.pt
MAC=$OUT/admitted_macros.json

echo "=== C2 batch ${K} | $(hostname) | $(date +%F' '%H:%M:%S) ==="
nvidia-smi --query-gpu=name --format=csv,noheader
test -f "$C"  || { echo "FATAL: missing $C -- run make_batches.py first"; exit 2; }
test -f "$WM" || { echo "FATAL: missing $WM -- phase B did not train a world model"; exit 2; }
grep -E '^(experiment_name|dataset_path|limit)' "$C"

mkdir -p "$OUT"

# --- gather the C1 shards into this task's own macro file ---------------------
# Per-task path on purpose: the 10 array tasks must not write one shared file.
# Content is identical in every task because the inputs are frozen by then.
$PY - "$MAC" runs/macroval <<'PY'
import glob, json, os, sys
out, root = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(root, "macro_*", "admitted.json")))
merged = []
for f in files:
    merged.extend(json.loads(open(f, encoding="utf-8").read()))
open(out, "w", encoding="utf-8").write(json.dumps(merged, indent=2))
print(f"admitted macros: {len(merged)} from {len(files)} C1 shard(s) -> {out}")
for m in merged:
    print("   ", m["name"], m["admission"]["reason"])
PY
[ $? -eq 0 ] || { echo "FATAL: could not gather C1 shards"; exit 2; }

# --- preflight: do the admitted macros actually reach the candidate pool? -----
# CPU-only, sub-second, no LLM. This exists because search/generator.py:60-87
# fills `candidates` to num_candidates with mutants FIRST, appends the macros
# AFTER, then returns list(candidates.values())[:num_candidates] -- so with 7
# seed templates and num_candidates=24 the macro is always element 25 and is
# always truncated away -- now fixed in generator.py, so this stays as a
# REGRESSION GUARD, because the failure was completely invisible in the output.
# Without it E4 would burn a full pass producing a
# "DarwinReasoner" arm that provably ignores every E9-admitted macro.
$PY - "$MAC" "$C" <<'PY'
import json, sys, yaml
from darwin_reasoner.schema import OperatorKind, ReasoningState
from darwin_reasoner.search import ArchitectureGenerator
macros = json.loads(open(sys.argv[1], encoding="utf-8").read())
cfg = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
if not macros:
    print("preflight: 0 admitted macros -- E4 runs on the seed grammar only.")
    sys.exit(0)
g = ArchitectureGenerator(seed=int(cfg["seed"]))
for m in macros:
    g.register_macro(m["name"], m["compiled_prompt"])
cands = g.generate(
    ReasoningState(problem_id="preflight", problem_prompt="preflight"),
    num_candidates=int(cfg["search"]["num_candidates"]),
    mutation_rounds=int(cfg["search"]["mutation_rounds"]),
    max_nodes=int(cfg["budget"]["max_nodes"]),
)
n = sum(1 for a in cands for nd in a.nodes if nd.operator == OperatorKind.MACRO)
print(f"preflight: {n} MACRO node(s) across {len(cands)} candidates "
      f"(num_candidates={cfg['search']['num_candidates']}, {len(macros)} macros registered)")
sys.exit(0 if n > 0 else 4)
PY
rc=$?
if [ $rc -eq 4 ]; then
  echo "FATAL: admitted macros never enter the candidate pool."
  echo "  search/generator.py:60-87 fills candidates to num_candidates with"
  echo "  mutants, THEN appends macros, THEN truncates [:num_candidates]."
  echo "  The macros are always the overflow. Raising num_candidates does not"
  echo "  help -- the while-loop fills to whatever num_candidates is."
  echo "  Fix: move the 'Add a few explicit macro candidates' block ABOVE the"
  echo "  while-loop, so macros occupy slots 7..7+M-1 and survive truncation."
  echo "  Then resubmit. Refusing to spend E4 on a null macro arm."
  exit 4
fi
if [ $rc -ne 0 ]; then
  echo "FATAL: preflight failed (exit $rc)"
  exit 2
fi

# --- run ----------------------------------------------------------------------
# NO rm -rf: this stage resumes correctly (see header).
$PY -m darwin_reasoner.cli run \
  --config "$C" \
  --world-model "$WM" \
  --run-dir "$OUT" \
  --macros "$MAC" 2>&1 \
  | grep -vE 'Processed prompts|Capturing CUDA graphs|Loading safetensors'
rc=${PIPESTATUS[0]}

echo "=== C2 batch ${K} | exit=$rc | $(date +%H:%M:%S) ==="
if [ -f "$OUT/raw_results.jsonl" ]; then
  n=$(wc -l < $OUT/raw_results.jsonl)
  echo "rows: $n  (expect 10 = one DarwinReasoner row per problem)"
  $PY -c "
import json, collections
rows=[json.loads(l) for l in open('$OUT/raw_results.jsonl',encoding='utf-8') if l.strip()]
if rows:
    print('  acc=%.3f  avg_calls=%.2f  avg_tokens=%.0f' % (
        sum(r['reward'] for r in rows)/len(rows),
        sum(r.get('calls',0) for r in rows)/len(rows),
        sum(r.get('tokens',0) for r in rows)/len(rows)))
    c=collections.Counter(a for r in rows for a in r.get('chosen_architectures',[]))
    print('  chosen:', dict(c))
    print('  macro picks:', sum(v for k,v in c.items() if k.startswith('macro::')))"
fi
exit $rc