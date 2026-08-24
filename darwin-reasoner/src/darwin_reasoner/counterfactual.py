from __future__ import annotations

from dataclasses import dataclass

from .config import CounterfactualConfig
from .executor import ArchitectureExecutor
from .schema import Budget, Outcome, Problem, ReasoningArchitecture, ReasoningState


@dataclass
class CounterfactualBatch:
    problem_id: str
    state_id: str
    outcomes: list[Outcome]

    def best(self) -> Outcome:
        return max(self.outcomes, key=lambda o: (o.reward, -o.tokens, -o.latency_s))


class CounterfactualCollector:
    """Matched architecture interventions from an identical reasoning prefix/state."""

    def __init__(
        self,
        executor: ArchitectureExecutor,
        config: CounterfactualConfig,
    ) -> None:
        self.executor = executor
        self.config = config

    def collect(
        self,
        problem: Problem,
        state: ReasoningState,
        architectures: list[ReasoningArchitecture],
        budget: Budget,
        *,
        base_seed: int,
    ) -> CounterfactualBatch:
        outcomes: list[Outcome] = []
        for repeat in range(self.config.repeats_per_architecture):
            for index, architecture in enumerate(architectures):
                if self.config.matched_seed_family:
                    # Same repeat receives the same base stochastic condition across actions.
                    seed = base_seed + repeat * 100_000
                else:
                    seed = base_seed + repeat * 100_000 + index * 1_000
                outcome = self.executor.execute(
                    problem,
                    architecture,
                    budget,
                    seed=seed,
                    initial_state=state,
                )
                outcome.metadata["counterfactual_repeat"] = repeat
                outcome.metadata["matched_seed_family"] = self.config.matched_seed_family
                outcomes.append(outcome)
        return CounterfactualBatch(problem.problem_id, state.state_id, outcomes)
