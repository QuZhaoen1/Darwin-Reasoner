#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from darwin_reasoner.config import ExperimentConfig
from darwin_reasoner.experiment import run_darwin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--macros", required=True)
    parser.add_argument("--out", default="runs/ablations")
    args = parser.parse_args()

    base = ExperimentConfig.load(args.config)
    out = Path(args.out)
    variants = {
        "full": {},
        "no_replan": {"online_rounds": 1},
        "random_architecture": {"selection_mode": "random"},
        "no_risk_penalty": {"risk_beta": 0.0},
        "fixed_grammar": {"no_macros": True},
        "untrained_world_model": {"untrained_world_model": True},
    }

    for name, change in variants.items():
        cfg = base.model_copy(deep=True)
        cfg.experiment_name = f"ablation_{name}"
        cfg.extra.update({k: v for k, v in change.items() if k not in {"risk_beta", "no_macros", "untrained_world_model"}})
        if "risk_beta" in change:
            cfg.search.risk_beta = float(change["risk_beta"])
        wm = args.world_model
        if change.get("untrained_world_model"):
            wm = str(out / "does_not_exist.pt")
        macros = None if change.get("no_macros") else args.macros
        run_darwin(
            cfg,
            world_model_path=wm,
            macros_path=macros,
            run_dir=str(out / name),
        )


if __name__ == "__main__":
    main()
