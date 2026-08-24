#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_math500(output: Path, limit: int | None) -> int:
    from datasets import load_dataset

    # Public dataset: no Hugging Face token is required.
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            rec = {
                "problem_id": str(row.get("unique_id") or f"math500-{idx:04d}"),
                "prompt": str(row["problem"]),
                "answer": str(row["answer"]),
                "domain": "math",
                "metadata": {
                    "source_dataset": "HuggingFaceH4/MATH-500",
                    "split": "test",
                    "subject": row.get("subject"),
                    "level": row.get("level"),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(ds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download public, no-key benchmark data and convert to DarwinReasoner JSONL."
    )
    parser.add_argument("--preset", choices=["math500"], default="math500")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="data/math500_pilot.jsonl")
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing datasets package. Run: DARWIN_GPU_INSTALL=1 bash scripts/setup.sh"
        ) from exc

    output = Path(args.output)
    if args.preset == "math500":
        n = write_math500(output, args.limit)
    else:
        raise AssertionError(args.preset)

    print(f"Prepared {n} public benchmark rows at {output}")
    print("No HF_TOKEN was used or required.")


if __name__ == "__main__":
    main()
