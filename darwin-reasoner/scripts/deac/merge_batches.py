#!/usr/bin/env python3
"""Merge per-batch raw_results.jsonl into one file the repo's own tooling reads.

The dedup key is the same tuple scripts/collect_counterfactuals_8gpu.py:107-110
uses, extended with `method` so baseline rows (which share a problem_id across
cot/verify/critical_repair/...) are not collapsed into one.

Also reports which batches are missing or short, because a silently truncated
merge looks exactly like a completed run to build_paper_results.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True, help="e.g. runs/pilot")
    p.add_argument("--stage", required=True, choices=["baselines", "counterfactual"])
    p.add_argument("--expect-batches", type=int, required=True)
    p.add_argument("--expect-rows-per-batch", type=int, default=None,
                   help="If given, batches with fewer rows are reported as SHORT")
    p.add_argument("--out", default=None, help="Defaults to <run-root>/<stage>_merged.jsonl")
    return p.parse_args()


def row_key(row: dict) -> tuple:
    return (
        row.get("problem_id"),
        row.get("state_id"),
        row.get("architecture_id"),
        row.get("method"),
        row.get("seed"),
        (row.get("metadata") or {}).get("counterfactual_repeat"),
    )


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    out = Path(args.out) if args.out else root / f"{args.stage}_merged.jsonl"

    seen: set[tuple] = set()
    kept = dups = 0
    present, missing, short = [], [], []

    with out.open("w", encoding="utf-8") as fh:
        for k in range(args.expect_batches):
            src = root / f"batch_{k:02d}" / args.stage / "raw_results.jsonl"
            if not src.exists():
                missing.append(k)
                continue
            n_before = kept
            for line in src.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row_key(row)
                if key in seen:
                    dups += 1
                    continue
                seen.add(key)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            n = kept - n_before
            present.append((k, n))
            if args.expect_rows_per_batch and n < args.expect_rows_per_batch:
                short.append((k, n))

    problems = len({r for r, *_ in (row_key(json.loads(l)) for l in
                                    out.read_text(encoding="utf-8").splitlines() if l.strip())})

    info = {
        "stage": args.stage, "out": str(out), "rows": kept, "duplicates_dropped": dups,
        "distinct_problems": problems,
        "batches_expected": args.expect_batches,
        "batches_present": [k for k, _ in present],
        "batches_missing": missing,
        "batches_short": short,
        "complete": not missing and not short,
    }
    (root / f"{args.stage}_MERGE_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"{args.stage}: {kept} rows from {len(present)}/{args.expect_batches} batches "
          f"({problems} distinct problems, {dups} dups dropped) -> {out}")
    if missing:
        print(f"  MISSING batches {missing}  -> resubmit with --array={','.join(map(str, missing))}")
    if short:
        print(f"  SHORT batches {short}  (likely hit the walltime mid-batch)")
    if not missing and not short:
        print("  complete")


if __name__ == "__main__":
    main()
