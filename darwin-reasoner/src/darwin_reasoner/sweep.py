from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import yaml

from .config import ExperimentConfig
from .experiment import run_baselines, run_darwin


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def run_sweep(spec_path: str) -> dict:
    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    base = ExperimentConfig.load(spec["base_config"])
    root = Path(spec.get("work_dir", "runs/full_sweep"))
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for dataset in spec["datasets"]:
        for model in spec["models"]:
            for budget_tokens in spec["budgets"]:
                for seed in spec["seeds"]:
                    cfg = base.model_copy(deep=True)
                    cfg.dataset_path = dataset["path"]
                    cfg.domain = dataset.get("domain", dataset["name"])
                    cfg.backend.model = model["name"]
                    cfg.backend.tensor_parallel_size = int(model.get("tensor_parallel_size", 1))
                    cfg.budget.max_tokens = int(budget_tokens)
                    cfg.seed = int(seed)
                    cfg.experiment_name = (
                        f"{dataset['name']}_{_slug(model['name'])}_b{budget_tokens}_s{seed}"
                    )

                    cell = (
                        root
                        / _slug(dataset["name"])
                        / _slug(model["name"])
                        / f"b{budget_tokens}"
                        / f"seed{seed}"
                    )
                    baseline_dir = cell / "baselines"
                    darwin_dir = cell / "darwin"
                    cell.mkdir(parents=True, exist_ok=True)

                    if spec.get("run_baselines", True):
                        run_baselines(cfg, run_dir=str(baseline_dir))

                    wm_path = model.get("world_model", spec.get("world_model"))
                    macros_path = model.get("macros", spec.get("macros"))
                    if spec.get("run_darwin", True):
                        if not wm_path:
                            raise ValueError(
                                f"No world model configured for {model['name']}. "
                                "Set world_model globally or per model."
                            )
                        run_darwin(
                            cfg,
                            world_model_path=wm_path,
                            run_dir=str(darwin_dir),
                            macros_path=macros_path,
                        )

                    record = {
                        "dataset": dataset["name"],
                        "model": model["name"],
                        "budget": budget_tokens,
                        "seed": seed,
                        "baseline_dir": str(baseline_dir),
                        "darwin_dir": str(darwin_dir),
                    }
                    records.append(record)
                    (root / "sweep_index.json").write_text(
                        json.dumps(records, indent=2), encoding="utf-8"
                    )
    return {"cells": len(records), "index": str(root / "sweep_index.json")}
