from __future__ import annotations

import json
from pathlib import Path

import yaml

from .config import ExperimentConfig
from .experiment import (
    collect_counterfactuals,
    mine_macros,
    run_baselines,
    run_darwin,
    train_world_model,
    validate_macros,
)


def run_pipeline(path: str) -> dict:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = Path(spec.get("work_dir", "runs/pipeline"))
    root.mkdir(parents=True, exist_ok=True)

    base_config_path = spec["experiment_config"]
    config = ExperimentConfig.load(base_config_path)
    status_path = root / "pipeline_status.json"
    status = _load_status(status_path)

    # E1 baselines
    baseline_dir = root / "e1_baselines"
    if not status.get("e1_baselines"):
        run_baselines(config, run_dir=str(baseline_dir))
        status["e1_baselines"] = "done"
        _save_status(status_path, status)

    # E2 counterfactual data
    cf_dir = root / "e2_counterfactual"
    if not status.get("e2_counterfactual"):
        collect_counterfactuals(config, run_dir=str(cf_dir))
        status["e2_counterfactual"] = "done"
        _save_status(status_path, status)

    cf_data = cf_dir / "raw_results.jsonl"
    wm_path = root / "e3_world_model" / "world_model.pt"
    if not status.get("e3_world_model"):
        wm_path.parent.mkdir(parents=True, exist_ok=True)
        train_world_model(config, data_path=str(cf_data), output_path=str(wm_path))
        status["e3_world_model"] = "done"
        _save_status(status_path, status)

    macro_candidates_path = root / "e3_world_model" / "macro_candidates.json"
    if not status.get("e3_macro_candidates"):
        mine_macros(config, data_path=str(cf_data), output_path=str(macro_candidates_path))
        status["e3_macro_candidates"] = "done"
        _save_status(status_path, status)

    macros_path = root / "e3_world_model" / "admitted_macros.json"
    macro_validation_dir = root / "e3_macro_validation"
    if not status.get("e3_macro_validation"):
        validate_macros(
            config,
            candidates_path=str(macro_candidates_path),
            output_path=str(macros_path),
            run_dir=str(macro_validation_dir),
        )
        status["e3_macro_validation"] = "done"
        _save_status(status_path, status)

    # Pilot gate. The threshold applies to reward at the same configured budget.
    darwin_dir = root / "e4_darwin"
    if not status.get("e4_darwin"):
        run_darwin(
            config,
            world_model_path=str(wm_path),
            run_dir=str(darwin_dir),
            macros_path=str(macros_path),
        )
        status["e4_darwin"] = "done"
        _save_status(status_path, status)

    baseline_accuracy = _metric(baseline_dir / "metrics.json", "best_method_accuracy")
    darwin_accuracy = _metric(darwin_dir / "metrics.json", "accuracy")
    delta = darwin_accuracy - baseline_accuracy
    gate = spec.get("gate", {})
    abort_below = float(gate.get("abort_below", -0.02))
    diagnose_below = float(gate.get("diagnose_below", 0.01))

    if delta < abort_below:
        decision = "abort_full_sweep"
    elif delta < diagnose_below:
        decision = "diagnostics_only"
    else:
        decision = "continue_full_sweep"

    status["pilot_gate"] = {
        "baseline_accuracy": baseline_accuracy,
        "darwin_accuracy": darwin_accuracy,
        "delta": delta,
        "decision": decision,
    }
    _save_status(status_path, status)
    return status


def _metric(path: Path, key: str) -> float:
    if not path.exists():
        return 0.0
    return float(json.loads(path.read_text())[key])


def _load_status(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_status(path: Path, status: dict) -> None:
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")
