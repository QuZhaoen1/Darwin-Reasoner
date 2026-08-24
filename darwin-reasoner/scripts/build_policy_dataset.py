#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="counterfactual raw_results.jsonl")
    p.add_argument("--train", default="data/policy_train.jsonl")
    p.add_argument("--valid", default="data/policy_valid.jsonl")
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--min-group", type=int, default=2)
    return p.parse_args()


def stable_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def main() -> None:
    args = parse_args()
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                groups[(str(row["problem_id"]), str(row["state_id"]))].append(row)

    train_rows, valid_rows = [], []
    for (problem_id, state_id), rows in groups.items():
        if len(rows) < args.min_group:
            continue
        # Accuracy first, then lower token cost, then lower latency.
        best = max(
            rows,
            key=lambda r: (
                float(r.get("reward", 0.0)),
                -float(r.get("tokens", 0.0)),
                -float(r.get("latency_s", 0.0)),
            ),
        )
        budget = best.get("budget", {})
        state_text = str(best.get("state_text", ""))
        prompt = (
            "Select or synthesize the compute-efficient reasoning architecture for this state.\n\n"
            f"STATE:\n{state_text}\n\n"
            f"BUDGET:\n{json.dumps(budget, ensure_ascii=False, sort_keys=True)}\n\n"
            "Return one architecture JSON object with name, nodes, edges, and metadata."
        )
        target = json.dumps(best["architecture"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        item = {
            "prompt": prompt,
            "target": target,
            "metadata": {
                "problem_id": problem_id,
                "state_id": state_id,
                "reward": best.get("reward"),
                "tokens": best.get("tokens"),
                "architecture_id": best.get("architecture_id"),
            },
        }
        (valid_rows if stable_fraction(problem_id) < args.valid_ratio else train_rows).append(item)

    for path, items in [(Path(args.train), train_rows), (Path(args.valid), valid_rows)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"train={len(train_rows)} valid={len(valid_rows)}")
    if not train_rows:
        raise SystemExit("No training rows were produced. Collect more counterfactual outcomes first.")


if __name__ == "__main__":
    main()
