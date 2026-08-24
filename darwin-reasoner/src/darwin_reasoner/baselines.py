from __future__ import annotations

from collections import Counter

from .dsl import chain_architecture
from .executor import ArchitectureExecutor
from .schema import Budget, OperatorKind, Outcome, Problem, ReasoningState
from .verifier import normalize_answer, verify_exact


def run_self_consistency(
    executor: ArchitectureExecutor,
    problem: Problem,
    state: ReasoningState,
    budget: Budget,
    *,
    samples: int,
    seed: int,
) -> Outcome:
    """Self-consistency baseline with majority vote and no reference-answer access during selection."""
    architecture = chain_architecture("self_consistency", [OperatorKind.SOLVE])
    per_sample_budget = budget.model_copy(
        update={"max_tokens": max(128, budget.max_tokens // max(samples, 1))}
    )
    outcomes = [
        executor.execute(
            problem,
            architecture,
            per_sample_budget,
            seed=seed + i * 1009,
            initial_state=state,
        )
        for i in range(samples)
    ]
    normalized = [normalize_answer(o.final_answer) for o in outcomes]
    counts = Counter(normalized)
    winner_norm, _ = counts.most_common(1)[0]
    winner = next(o.final_answer for o, n in zip(outcomes, normalized, strict=True) if n == winner_norm)
    verification = verify_exact(f"FINAL: {winner}", problem.answer)
    return Outcome(
        problem_id=problem.problem_id,
        state_id=state.state_id,
        architecture_id=architecture.architecture_id,
        architecture_name=f"self_consistency_{samples}",
        reward=verification.reward,
        correct=verification.correct,
        final_answer=winner,
        tokens=sum(o.tokens for o in outcomes),
        latency_s=sum(o.latency_s for o in outcomes),
        calls=sum(o.calls for o in outcomes),
        seed=seed,
        trace=[{"sample": i, "final_answer": o.final_answer, "trace": o.trace} for i, o in enumerate(outcomes)],
        metadata={
            "selection": "majority_vote",
            "samples": samples,
            "architecture": architecture.canonical_dict(),
        },
    )
