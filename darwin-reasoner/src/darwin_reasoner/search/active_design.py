from __future__ import annotations

from dataclasses import dataclass

from darwin_reasoner.schema import ReasoningArchitecture, WorldModelPrediction


@dataclass
class DesignChoice:
    architecture: ReasoningArchitecture
    acquisition_score: float
    reason: str


class ActiveCounterfactualDesigner:
    """Choose real interventions that are useful for both solving and world-model learning.

    The acquisition score is a lightweight value-of-information proxy: exploit predicted reward,
    explore epistemic uncertainty, and penalize predicted compute. The experiment protocol keeps a
    random-control fraction so efficiency claims can be audited against uniform intervention.
    """

    def __init__(
        self,
        *,
        uncertainty_weight: float = 0.6,
        reward_weight: float = 0.4,
        token_penalty: float = 0.00001,
    ) -> None:
        self.uncertainty_weight = uncertainty_weight
        self.reward_weight = reward_weight
        self.token_penalty = token_penalty

    def select(
        self,
        architectures: list[ReasoningArchitecture],
        predictions: list[WorldModelPrediction],
        *,
        k: int,
    ) -> list[DesignChoice]:
        scored: list[DesignChoice] = []
        for architecture, pred in zip(architectures, predictions, strict=True):
            acquisition = (
                self.reward_weight * pred.reward_mean
                + self.uncertainty_weight * pred.reward_std
                - self.token_penalty * pred.tokens_mean
            )
            reason = "uncertain_high_value" if pred.reward_std > 0.2 else "predicted_high_value"
            scored.append(DesignChoice(architecture, acquisition, reason))
        scored.sort(key=lambda item: item.acquisition_score, reverse=True)
        return scored[:k]
