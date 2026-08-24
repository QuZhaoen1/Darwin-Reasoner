#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Hugging Face dataset to DarwinReasoner JSONL")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--prompt-field", required=True)
    parser.add_argument("--answer-field", required=True)
    parser.add_argument("--id-field", default=None)
    parser.add_argument("--domain", default="unknown")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install GPU/data extras first: pip install -e '.[gpu]'") from exc

    ds = load_dataset(args.dataset, args.subset, split=args.split)
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(ds):
            problem_id = str(row[args.id_field]) if args.id_field else f"{args.domain}-{idx:06d}"
            record = {
                "problem_id": problem_id,
                "prompt": str(row[args.prompt_field]),
                "answer": str(row[args.answer_field]),
                "domain": args.domain,
                "metadata": {"source_dataset": args.dataset, "split": args.split},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(ds)} rows to {output}")


if __name__ == "__main__":
    main()
