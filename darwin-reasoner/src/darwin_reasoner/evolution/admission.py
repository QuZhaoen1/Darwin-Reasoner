from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MacroAdmissionResult:
    admitted: bool
    reward_delta: float
    token_delta: float
    reward_ci_low: float
    reward_ci_high: float
    reason: str


def evaluate_macro_admission(
    macro_rewards: list[float],
    expanded_rewards: list[float],
    macro_tokens: list[float],
    expanded_tokens: list[float],
    *,
    noninferiority_margin: float = 0.01,
    min_token_saving: float = 0.05,
    bootstrap_draws: int = 4000,
    seed: int = 0,
) -> MacroAdmissionResult:
    """Paired non-inferiority test before a mined motif becomes a reusable macro.

    The two reward/token lists must be aligned by matched problem-state and seed block. A macro is
    admitted only when its lower bootstrap confidence bound is above the non-inferiority margin and
    it saves a minimum fraction of tokens. This prevents the search grammar from expanding merely
    because an observational motif looked good.
    """
    if not (
        len(macro_rewards)
        == len(expanded_rewards)
        == len(macro_tokens)
        == len(expanded_tokens)
    ):
        raise ValueError("Macro admission inputs must be paired and have equal length")
    if not macro_rewards:
        return MacroAdmissionResult(False, 0.0, 0.0, 0.0, 0.0, "no_data")

    r_macro = np.asarray(macro_rewards, dtype=float)
    r_expanded = np.asarray(expanded_rewards, dtype=float)
    t_macro = np.asarray(macro_tokens, dtype=float)
    t_expanded = np.asarray(expanded_tokens, dtype=float)
    reward_diff = r_macro - r_expanded
    token_saving = (t_expanded - t_macro) / np.maximum(t_expanded, 1.0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(reward_diff), size=(bootstrap_draws, len(reward_diff)))
    boot = reward_diff[idx].mean(axis=1)
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    mean_reward_delta = float(reward_diff.mean())
    mean_token_saving = float(token_saving.mean())

    reward_ok = lo >= -noninferiority_margin
    token_ok = mean_token_saving >= min_token_saving
    admitted = reward_ok and token_ok
    if admitted:
        reason = "noninferior_and_cheaper"
    elif not reward_ok:
        reason = "reward_noninferiority_failed"
    else:
        reason = "insufficient_token_saving"
    return MacroAdmissionResult(
        admitted=admitted,
        reward_delta=mean_reward_delta,
        token_delta=mean_token_saving,
        reward_ci_low=lo,
        reward_ci_high=hi,
        reason=reason,
    )
