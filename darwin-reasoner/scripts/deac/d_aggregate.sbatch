#!/bin/bash
#SBATCH --job-name=dr_d
#SBATCH --account=yanggrp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100-01,gpu-v100-02,gpu-v100-03,gpu-v100-04
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --output=/deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner/logs/%x-%j.out

# Phase D - final aggregation. Pure pandas/numpy over ~3.1k rows: seconds, not
# minutes. It asks for a GPU only to stay on the one partition this account is
# set up for; move it to a CPU partition if one is ever granted.
#
# Submit with --dependency=afterany:<C2 array jobid>, matching phase B's reason:
# if a batch failed we still want the tables built and the gap PRINTED, rather
# than a job that never starts and a pipeline that stalls silently.
#   sbatch --dependency=afterany:<C2 array jobid> scripts/deac/d_aggregate.sbatch

set -uo pipefail
source /deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner/darwin-reasoner/scripts/deac/env.sh
cd $REPO

N=${N_BATCHES:-10}
SIZE=${BATCH_SIZE:-10}
R=runs/pilot
MV=runs/macroval
OUTDIR=${OUT_DIR:-paper_results/pilot}
FORCE=${FORCE_INCOMPLETE:-0}

echo "=== D | $(hostname) | $(date +%F' '%H:%M:%S) | batches=$N size=$SIZE ==="
mkdir -p "$OUTDIR"

# --- re-merge E1/E2 (idempotent; picks up any batch resubmitted after B) ------
echo; echo "--- merge E1 ---"
$PY scripts/deac/merge_batches.py --run-root $R --stage baselines \
    --expect-batches $N --expect-rows-per-batch $((SIZE * 6))
echo; echo "--- merge E2 ---"
$PY scripts/deac/merge_batches.py --run-root $R --stage counterfactual \
    --expect-batches $N --expect-rows-per-batch $((SIZE * 24))

# --- merge E4 ------------------------------------------------------------------
# merge_batches.py cannot do this stage: its --stage argument is
# choices=["baselines","counterfactual"] (merge_batches.py:21) and would exit 2.
# Same dedup key and same MERGE_INFO contract, inline.
echo; echo "--- merge E4 (darwin) ---"
$PY - "$R" "$N" "$((SIZE * 1))" <<'PY'
import json, sys
from pathlib import Path
root, n, expect = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
out = root / "darwin_merged.jsonl"
seen, kept, dups = set(), 0, 0
present, missing, short = [], [], []
with out.open("w", encoding="utf-8") as fh:
    for k in range(n):
        src = root / f"batch_{k:02d}" / "darwin" / "raw_results.jsonl"
        if not src.exists():
            missing.append(k)
            continue
        before = kept
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("problem_id"), row.get("state_id"),
                   row.get("architecture_id"), row.get("method"), row.get("seed"))
            if key in seen:
                dups += 1
                continue
            seen.add(key)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
        got = kept - before
        present.append((k, got))
        if got < expect:
            short.append((k, got))
rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
info = {
    "stage": "darwin", "out": str(out), "rows": kept, "duplicates_dropped": dups,
    "distinct_problems": len({r.get("problem_id") for r in rows}),
    "batches_expected": n, "batches_present": [k for k, _ in present],
    "batches_missing": missing, "batches_short": short,
    "complete": not missing and not short,
}
(root / "darwin_MERGE_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print(f"darwin: {kept} rows from {len(present)}/{n} batches "
      f"({info['distinct_problems']} distinct problems, {dups} dups dropped) -> {out}")
if missing:
    print(f"  MISSING batches {missing}  -> resubmit with --array={','.join(map(str, missing))}")
if short:
    print(f"  SHORT batches {short}  (run_darwin resumes, so just resubmit those tasks)")
if info["complete"]:
    print("  complete")
PY

# --- canonical admitted-macro record (E9) --------------------------------------
echo; echo "--- collect E9 admissions ---"
$PY - "$MV" "$R/admitted_macros.json" "$R/macro_admission_summary.json" <<'PY'
import glob, json, os, sys
root, out_admitted, out_summary = sys.argv[1], sys.argv[2], sys.argv[3]
admitted, rows = [], []
for f in sorted(glob.glob(os.path.join(root, "macro_*", "admitted.json"))):
    admitted.extend(json.loads(open(f, encoding="utf-8").read()))
for f in sorted(glob.glob(os.path.join(root, "macro_*", "candidates.json"))):
    slot = os.path.basename(os.path.dirname(f))
    payload = json.loads(open(f, encoding="utf-8").read())
    if not payload:          # a slot beyond the mined-candidate count writes []
        continue
    cand = payload[0]
    pairs = os.path.join(os.path.dirname(f), "raw_results.jsonl")
    n = sum(1 for l in open(pairs, encoding="utf-8") if l.strip()) if os.path.exists(pairs) else 0
    rows.append({"slot": slot, "macro": cand["name"], "operators": cand["operators"],
                 "pairs_run": n, "pairs_expected": 50})
json.dump(admitted, open(out_admitted, "w", encoding="utf-8"), indent=2)
json.dump({"tested": len(rows), "admitted": len(admitted), "shards": rows},
          open(out_summary, "w", encoding="utf-8"), indent=2)
print(f"{len(admitted)} admitted of {len(rows)} tested -> {out_admitted}")
for r in rows:
    flag = "" if r["pairs_run"] >= r["pairs_expected"] else "  <-- SHORT"
    print(f"  {r['slot']}  {r['macro']:<40s} pairs={r['pairs_run']}/50{flag}")
for m in admitted:
    a = m["admission"]
    print(f"  ADMITTED {m['name']}: dR={a['reward_delta']:+.4f} "
          f"CI[{a['reward_ci_low']:+.4f},{a['reward_ci_high']:+.4f}] "
          f"token_saving={a['token_delta']:.3f} ({a['reason']})")
PY

# --- completeness gate ---------------------------------------------------------
echo; echo "--- completeness ---"
for stage in baselines counterfactual darwin; do
  c=$($PY -c "import json;print(json.load(open('$R/${stage}_MERGE_INFO.json'))['complete'])" 2>/dev/null || echo "MISSING")
  echo "  $stage complete: $c"
  if [ "$c" != "True" ] && [ "$FORCE" != "1" ]; then
    echo "ABORT: $stage is incomplete. A truncated merge produces a paper table that"
    echo "       looks finished and is not. Resubmit the missing tasks listed in"
    echo "       $R/${stage}_MERGE_INFO.json, then rerun."
    echo "       Override deliberately with FORCE_INCOMPLETE=1."
    exit 3
  fi
done

# --- paper tables ---------------------------------------------------------------
# --runs MUST be runs/pilot. The script's default is "runs" and it rglobs for
# raw_results.jsonl (build_paper_results.py:14), which on this tree would also
# swallow runs/e2e, runs/gpu_probe and runs/smoke_e2e -- mock-backend rows with
# tokens=111 and latency=0 -- straight into table_main.csv.
echo; echo "--- build_paper_results ---"
$PY scripts/build_paper_results.py --runs "$R" --out "$OUTDIR"
rc=$?
echo "exit=$rc"

echo; echo "=== MAIN TABLE ==="
[ -f "$OUTDIR/table_main.csv" ] && cat "$OUTDIR/table_main.csv"
echo; echo "=== COMPUTE FRONTIER ==="
[ -f "$OUTDIR/compute_frontier.csv" ] && cat "$OUTDIR/compute_frontier.csv"
echo; echo "=== WORLD MODEL EVAL (from B) ==="
[ -f "$R/world_model_eval.json" ] && cat "$R/world_model_eval.json"
echo; echo "=== D DONE $(date +%H:%M:%S) | exit=$rc ==="
exit $rc