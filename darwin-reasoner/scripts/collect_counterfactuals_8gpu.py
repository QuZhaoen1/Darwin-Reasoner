#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shard counterfactual rollout collection over 8 A100s")
    p.add_argument("--config", default="configs/pilot_8b.yaml")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--run-root", default="runs/pilot_counterfactual_8gpu")
    return p.parse_args()


def split_jsonl(src: Path, root: Path, n: int) -> list[Path]:
    rows = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"Dataset is empty: {src}")
    shards = [[] for _ in range(n)]
    for i, row in enumerate(rows):
        shards[i % n].append(row)
    paths = []
    for i, items in enumerate(shards):
        path = root / f"dataset_shard_{i:02d}.jsonl"
        path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    if args.gpus != 8:
        print(f"NOTE: launcher requested {args.gpus} workers; the paper profile uses 8.")
    config_path = Path(args.config)
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_path = Path(base["dataset_path"])
    if not dataset_path.exists():
        raise SystemExit(f"Missing dataset {dataset_path}; run bootstrap_public_data.py first")

    run_root = Path(args.run_root)
    runtime = run_root / "_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    shard_paths = split_jsonl(dataset_path, runtime, args.gpus)

    processes: list[tuple[int, subprocess.Popen, Path]] = []
    for gpu_id, shard_path in enumerate(shard_paths):
        cfg = json.loads(json.dumps(base))
        cfg["dataset_path"] = str(shard_path)
        cfg["limit"] = None
        cfg["experiment_name"] = f"pilot_cf_gpu{gpu_id}"
        cfg["backend"]["tensor_parallel_size"] = 1
        cfg_path = runtime / f"config_gpu{gpu_id}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        shard_run = run_root / f"gpu{gpu_id}"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["TOKENIZERS_PARALLELISM"] = "false"
        log_path = runtime / f"gpu{gpu_id}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "darwin_reasoner.cli",
            "collect",
            "--config",
            str(cfg_path),
            "--run-dir",
            str(shard_run),
        ]
        proc = subprocess.Popen(cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        proc._darwin_log_handle = log_handle  # type: ignore[attr-defined]
        processes.append((gpu_id, proc, shard_run))
        print(f"launched GPU {gpu_id}: {' '.join(cmd)}")

    failed = []
    for gpu_id, proc, shard_run in processes:
        code = proc.wait()
        getattr(proc, "_darwin_log_handle").close()
        if code != 0:
            failed.append(gpu_id)
        else:
            print(f"GPU {gpu_id} finished: {shard_run}")
    if failed:
        raise SystemExit(f"Counterfactual workers failed on GPU(s): {failed}; inspect {runtime}/*.log")

    merged = run_root / "raw_results.jsonl"
    metrics = run_root / "MERGE_INFO.json"
    seen = set()
    count = 0
    with merged.open("w", encoding="utf-8") as out:
        for gpu_id, _, shard_run in processes:
            src = shard_run / "raw_results.jsonl"
            if not src.exists():
                continue
            for line in src.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    row.get("problem_id"), row.get("state_id"), row.get("architecture_id"),
                    row.get("seed"), row.get("metadata", {}).get("counterfactual_repeat"),
                )
                if key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    metrics.write_text(json.dumps({"workers": args.gpus, "rows": count}, indent=2), encoding="utf-8")
    print(f"PASS: merged {count} counterfactual outcomes -> {merged}")

    # Runtime configs/shards are reproducibility artifacts; keep them. No model weights are copied.


if __name__ == "__main__":
    main()
