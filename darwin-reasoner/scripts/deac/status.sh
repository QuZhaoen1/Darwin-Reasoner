#!/usr/bin/env bash
# One-screen view of the batched pilot: queue state + per-batch row counts.
# Usage: bash scripts/deac/status.sh [n_batches] [problems_per_batch]

set -uo pipefail
D=/deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner
cd $D/darwin-reasoner

N=${1:-10}
SIZE=${2:-10}
E1_EXPECT=$((SIZE * 6))    # 6 baseline methods per problem
E2_EXPECT=$((SIZE * 24))   # execute_top_k 8 x repeats 3

echo "=== queue ==="
squeue -u qianr -o '%.14i %.10j %.9P %.8T %.10M %.11l %R' 2>/dev/null

echo
printf "=== batches (E1 expect %d rows, E2 expect %d rows) ===\n" $E1_EXPECT $E2_EXPECT
printf "%-7s %12s %12s   %s\n" batch E1 E2 state
e1_done=0; e2_done=0
for k in $(seq 0 $((N-1))); do
  K=$(printf "%02d" $k)
  f1=runs/pilot/batch_${K}/baselines/raw_results.jsonl
  f2=runs/pilot/batch_${K}/counterfactual/raw_results.jsonl
  n1=$([ -f "$f1" ] && wc -l < "$f1" || echo 0)
  n2=$([ -f "$f2" ] && wc -l < "$f2" || echo 0)
  s=""
  [ "$n1" -ge "$E1_EXPECT" ] && { s="${s}E1ok "; e1_done=$((e1_done+1)); } || { [ "$n1" -gt 0 ] && s="${s}E1partial "; }
  [ "$n2" -ge "$E2_EXPECT" ] && { s="${s}E2ok"; e2_done=$((e2_done+1)); } || { [ "$n2" -gt 0 ] && s="${s}E2partial"; }
  [ -z "$s" ] && s="-"
  printf "%-7s %12s %12s   %s\n" "$K" "$n1/$E1_EXPECT" "$n2/$E2_EXPECT" "$s"
done

echo
echo "complete: E1 $e1_done/$N   E2 $e2_done/$N"
echo
echo "=== recent failures ==="
sacct -u qianr --starttime today --format=JobID%18,JobName%10,State,Elapsed,ExitCode -P 2>/dev/null \
  | awk -F'|' 'NR==1 || ($3!="COMPLETED" && $3!="RUNNING" && $3!="PENDING" && $1 !~ /\./)' | head -12
