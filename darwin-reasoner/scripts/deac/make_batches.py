#!/usr/bin/env python3
"""Split the pilot dataset into contiguous batches, one Slurm array task each.

Batching rather than resuming is deliberate. `cli collect` re-executes every
rollout before it checks store.is_completed (experiment.py:215), so a requeued
or timed-out collect restarts from zero compute and can never converge. Disjoint
batches with their own run-dirs bound that blast radius: a lost batch costs one
batch, and it can be resubmitted alone with `--array=<k>`.

Batches are contiguous, not round-robin, so batch k is exactly problems
[k*size, (k+1)*size) and a gap in the merged output is traceable to one task.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/pilot_8b.yaml",
                   help="Base config whose backend/search/budget settings are inherited")
    p.add_argument("--dataset", default=None,
                   help="Override dataset path (defaults to the base config's dataset_path)")
    p.add_argument("--total", type=int, default=100,
                   help="How many problems of the dataset to use in total")
    p.add_argument("--size", type=int, default=10,
                   help="Problems per batch. 10 keeps E2 near 150 min against a 4h wall.")
    p.add_argument("--max-tokens", type=int, default=1536,
                   help="Override backend.max_tokens_per_call. Under the chat template the "
                        "model stops on its own at ~500 tokens, so this is headroom, not a target.")
    p.add_argument("--raw-prompts", action="store_true",
                   help="Reproduce the original bare-completion prompting. Off by default: "
                        "Qwen3-8B never emits EOS that way and loops until the cap (8/8 "
                        "truncated, measured), which biases short single-call architectures.")
    p.add_argument("--out-data", default="data/batches")
    p.add_argument("--out-config", default="configs/batches")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    src = Path(args.dataset or base["dataset_path"])
    if not src.exists():
        raise SystemExit(f"Missing dataset {src}; run scripts/bootstrap_public_data.py first")

    rows = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.total > len(rows):
        raise SystemExit(f"Asked for {args.total} problems but {src} only has {len(rows)}")
    rows = rows[: args.total]

    data_dir, cfg_dir = Path(args.out_data), Path(args.out_config)
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    n_batches = (len(rows) + args.size - 1) // args.size
    manifest = []
    for k in range(n_batches):
        chunk = rows[k * args.size : (k + 1) * args.size]
        data_path = data_dir / f"batch_{k:02d}.jsonl"
        data_path.write_text("\n".join(chunk) + "\n", encoding="utf-8")

        cfg = json.loads(json.dumps(base))  # deep copy without aliasing
        cfg["dataset_path"] = str(data_path)
        cfg["limit"] = None  # the batch file IS the limit
        cfg["experiment_name"] = f"pilot_batch_{k:02d}"
        cfg["backend"]["tensor_parallel_size"] = 1
        cfg["backend"]["max_tokens_per_call"] = args.max_tokens
        cfg["backend"]["use_chat_template"] = not args.raw_prompts
        cfg["backend"]["enable_thinking"] = False
        cfg_path = cfg_dir / f"batch_{k:02d}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        ids = [json.loads(c).get("problem_id") for c in chunk]
        manifest.append({
            "batch": k, "config": str(cfg_path), "dataset": str(data_path),
            "n_problems": len(chunk), "first_problem_id": ids[0], "last_problem_id": ids[-1],
        })

    Path(args.out_data, "MANIFEST.json").write_text(
        json.dumps({"source": str(src), "total": len(rows), "batch_size": args.size,
                    "n_batches": n_batches, "max_tokens_per_call": args.max_tokens,
                    "batches": manifest}, indent=2),
        encoding="utf-8")

    # Measured on an A100 80GB at 12.8 s per single-stream call with every call
    # capped at 1024 tokens: E1 4.1 min/problem, E2 14.6 min/problem. Under the
    # chat template the model terminates at ~500 tokens instead of running to the
    # cap, so calls are roughly half as long.
    scale = 1.0 if args.raw_prompts else 0.5
    e1 = args.size * 4.1 * scale
    e2 = args.size * 14.6 * scale
    print(f"wrote {n_batches} batches of {args.size} ({len(rows)} problems) -> {data_dir}, {cfg_dir}")
    print(f"projected per batch:  E1 {e1:.0f} min   E2 {e2:.0f} min   (+~3 min engine load each)")
    print(f"projected total GPU:  E1 {n_batches*e1/60:.1f} h   E2 {n_batches*e2/60:.1f} h")
    print(f"submit with:  --array=0-{n_batches-1}%3")


if __name__ == "__main__":
    main()
