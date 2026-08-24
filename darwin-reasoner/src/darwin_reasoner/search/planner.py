from __future__ import annotations

from dataclasses import dataclass

from darwin_reasoner.config import SearchConfig
from darwin_reasoner.schema import ReasoningArchitecture, WorldModelPrediction


@dataclass
class RankedCandidate:
    architecture: ReasoningArchitecture
    prediction: WorldModelPrediction
    score: float


class BudgetConditionedPlanner:
    """Risk-aware Pareto planner over reward, token cost, and latency."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config

    def score(self, prediction: WorldModelPrediction) -> float:
        pessimistic_reward = prediction.reward_mean - self.config.risk_beta * prediction.reward_std
        return (
            pessimistic_reward
            - self.config.token_penalty * prediction.tokens_mean
            - self.config.latency_penalty * prediction.latency_mean
        )

    def rank(
        self,
        architectures: list[ReasoningArchitecture],
        predictions: list[WorldModelPrediction],
    ) -> list[RankedCandidate]:
        if len(architectures) != len(predictions):
            raise ValueError("architectures and predictions length mismatch")
        ranked = [
            RankedCandidate(architecture=a, prediction=p, score=self.score(p))
            for a, p in zip(architectures, predictions, strict=True)
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked
