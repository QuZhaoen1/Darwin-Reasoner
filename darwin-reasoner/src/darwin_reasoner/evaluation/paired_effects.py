from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedEffect:
    mean_delta: float
    ci_low: float
    ci_high: float
    n_pairs: int
    win_rate: float


def estimate_paired_effect(
    treatment: list[float],
    control: list[float],
    *,
    draws: int = 5000,
    seed: int = 0,
) -> PairedEffect:
    """Estimate a matched intervention effect with a paired bootstrap."""
    if len(treatment) != len(control):
        raise ValueError("treatment/control must be paired")
    if not treatment:
        return PairedEffect(0.0, 0.0, 0.0, 0, 0.0)
    diff = np.asarray(treatment, dtype=float) - np.asarray(control, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(draws, len(diff)))
    means = diff[idx].mean(axis=1)
    return PairedEffect(
        mean_delta=float(diff.mean()),
        ci_low=float(np.quantile(means, 0.025)),
        ci_high=float(np.quantile(means, 0.975)),
        n_pairs=len(diff),
        win_rate=float((diff > 0).mean()),
    )
