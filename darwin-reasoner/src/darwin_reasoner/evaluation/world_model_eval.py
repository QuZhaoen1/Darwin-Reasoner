from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from darwin_reasoner.config import ExperimentConfig
from darwin_reasoner.schema import Budget, ReasoningArchitecture, ReasoningState
from darwin_reasoner.world_model import EnsembleWorldModel
from darwin_reasoner.world_model.model import WorldModelExample


def evaluate_world_model_group_split(
    config: ExperimentConfig,
    *,
    data_path: str,
    train_fraction: float = 0.8,
) -> tuple[EnsembleWorldModel, dict[str, float]]:
    rows = [
        json.loads(line)
        for line in Path(data_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["state_id"])].append(row)
    state_ids = sorted(groups)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(state_ids)
    split = max(1, min(len(state_ids) - 1, int(len(state_ids) * train_fraction))) if len(state_ids) > 1 else len(state_ids)
    train_ids = set(state_ids[:split])
    test_ids = state_ids[split:] or state_ids[:]

    train_examples: list[WorldModelExample] = []
    for state_id in train_ids:
        for row in groups[state_id]:
            train_examples.append(
                WorldModelExample(
                    state_text=row.get("state_text", ""),
                    architecture=ReasoningArchitecture.model_validate(row["architecture"]),
                    budget=Budget.model_validate(row.get("budget", config.budget.model_dump())),
                    reward=float(row["reward"]),
                    tokens=float(row.get("tokens", 0.0)),
                    latency_s=float(row.get("latency_s", 0.0)),
                )
            )

    model = EnsembleWorldModel(
        text_features=config.world_model.text_features,
        hidden_dim=config.world_model.hidden_dim,
        ensemble_size=config.world_model.ensemble_size,
        device=config.world_model.device,
    )
    model.fit(
        train_examples,
        epochs=config.world_model.epochs,
        batch_size=config.world_model.batch_size,
        learning_rate=config.world_model.learning_rate,
        seed=config.seed,
    )

    brier_values = []
    spearman_values = []
    top1_hits = []
    regrets = []
    pred_probs = []
    outcomes = []

    for state_id in test_ids:
        state_rows = groups[state_id]
        # Average repeated matched rollouts per architecture before evaluating ranking.
        arch_groups: dict[str, list[dict]] = defaultdict(list)
        for row in state_rows:
            arch_groups[str(row["architecture_id"])].append(row)
        reps = []
        for arch_id, arch_rows in arch_groups.items():
            first = arch_rows[0]
            reps.append(
                {
                    "architecture": ReasoningArchitecture.model_validate(first["architecture"]),
                    "budget": Budget.model_validate(first.get("budget", config.budget.model_dump())),
                    "reward": float(np.mean([float(r["reward"]) for r in arch_rows])),
                    "state_text": first.get("state_text", ""),
                }
            )
        if not reps:
            continue
        state = ReasoningState(
            problem_id=state_id,
            problem_prompt=reps[0]["state_text"],
            transcript="",
        )
        architectures = [r["architecture"] for r in reps]
        # Budgets are fixed within a matched state in the protocol.
        preds = model.predict_many(state, architectures, reps[0]["budget"])
        predicted = np.asarray([p.reward_mean for p in preds], dtype=float)
        realized = np.asarray([r["reward"] for r in reps], dtype=float)
        brier_values.extend((predicted - realized) ** 2)
        pred_probs.extend(predicted.tolist())
        outcomes.extend(realized.tolist())
        if len(realized) > 1 and np.std(realized) > 1e-9 and np.std(predicted) > 1e-9:
            corr = spearmanr(predicted, realized).statistic
            if np.isfinite(corr):
                spearman_values.append(float(corr))
        chosen = int(np.argmax(predicted))
        best_value = float(np.max(realized))
        top1_hits.append(float(np.isclose(realized[chosen], best_value)))
        regrets.append(best_value - float(realized[chosen]))

    metrics = {
        "n_states_train": float(len(train_ids)),
        "n_states_test": float(len(test_ids)),
        "brier": float(np.mean(brier_values)) if brier_values else float("nan"),
        "within_state_spearman": float(np.mean(spearman_values)) if spearman_values else float("nan"),
        "top1_best_architecture_accuracy": float(np.mean(top1_hits)) if top1_hits else float("nan"),
        "mean_selection_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "ece_10bin": _ece(np.asarray(pred_probs), np.asarray(outcomes), bins=10),
    }
    return model, metrics


def _ece(probs: np.ndarray, outcomes: np.ndarray, *, bins: int) -> float:
    if len(probs) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (probs >= edges[i]) & (probs < edges[i + 1] if i < bins - 1 else probs <= edges[i + 1])
        if not np.any(mask):
            continue
        total += (mask.mean()) * abs(float(probs[mask].mean()) - float(outcomes[mask].mean()))
    return float(total)
