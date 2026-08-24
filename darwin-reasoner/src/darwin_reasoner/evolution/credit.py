from __future__ import annotations

from collections import defaultdict

import numpy as np


def estimate_operator_credit(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Approximate structural credit from observed outcomes.

    This is deliberately labeled an observational diagnostic, not a causal estimator. The paper's
    causal/interventional claims should be based on matched intervention groups collected by the
    counterfactual collector. The diagnostic is useful for prioritizing motifs to test next.
    """
    present: dict[str, list[float]] = defaultdict(list)
    absent: dict[str, list[float]] = defaultdict(list)
    all_ops = sorted({op for row in rows for op in row.get("operator_sequence", [])})
    for row in rows:
        seq = set(row.get("operator_sequence", []))
        reward = float(row.get("reward", 0.0))
        for op in all_ops:
            (present if op in seq else absent)[op].append(reward)

    result: dict[str, dict[str, float]] = {}
    for op in all_ops:
        mean_present = float(np.mean(present[op])) if present[op] else 0.0
        mean_absent = float(np.mean(absent[op])) if absent[op] else 0.0
        result[op] = {
            "mean_present": mean_present,
            "mean_absent": mean_absent,
            "lift": mean_present - mean_absent,
            "support": float(len(present[op])),
        }
    return result
