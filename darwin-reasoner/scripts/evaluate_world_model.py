#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from darwin_reasoner.config import ExperimentConfig
from darwin_reasoner.evaluation.world_model_eval import evaluate_world_model_group_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="paper_results/world_model_eval.json")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    _, metrics = evaluate_world_model_group_split(
        cfg, data_path=args.data, train_fraction=args.train_fraction
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
