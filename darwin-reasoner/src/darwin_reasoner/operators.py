from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .schema import OperatorKind, ReasoningState


@dataclass(frozen=True)
class OperatorSpec:
    kind: OperatorKind
    prompt_builder: Callable[[ReasoningState], str]
    estimated_multiplier: float = 1.0
    description: str = ""


def _base(state: ReasoningState) -> str:
    return (
        f"Problem:\n{state.problem_prompt}\n\n"
        f"Current reasoning:\n{state.transcript or '[empty]'}\n\n"
    )


def default_operator_registry() -> dict[OperatorKind, OperatorSpec]:
    return {
        OperatorKind.SOLVE: OperatorSpec(
            OperatorKind.SOLVE,
            lambda s: _base(s)
            + "Solve the problem carefully. End with 'FINAL: <answer>'.",
            1.0,
            "Single-path solve.",
        ),
        OperatorKind.CONTINUE: OperatorSpec(
            OperatorKind.CONTINUE,
            lambda s: _base(s)
            + "Continue the current line of reasoning. Do not restart unless necessary. End with FINAL.",
            1.0,
            "Continue current trajectory.",
        ),
        OperatorKind.BRANCH: OperatorSpec(
            OperatorKind.BRANCH,
            lambda s: _base(s)
            + "Generate a genuinely different solution route. Contrast it with the current route and end with FINAL.",
            1.35,
            "Explore an alternative route.",
        ),
        OperatorKind.VERIFY: OperatorSpec(
            OperatorKind.VERIFY,
            lambda s: _base(s)
            + "Audit the reasoning step by step. Locate the earliest questionable step, repair it if needed, and end with FINAL.",
            0.85,
            "Check correctness and repair local errors.",
        ),
        OperatorKind.CRITIQUE: OperatorSpec(
            OperatorKind.CRITIQUE,
            lambda s: _base(s)
            + "Act as an adversarial critic. Try to falsify the current reasoning, then give the strongest corrected answer. End with FINAL.",
            1.0,
            "Adversarial critique.",
        ),
        OperatorKind.BACKTRACK: OperatorSpec(
            OperatorKind.BACKTRACK,
            lambda s: _base(s)
            + "Backtrack to the earliest likely mistake or weak assumption, reconstruct from there, and end with FINAL.",
            1.15,
            "Repair by revisiting an earlier state.",
        ),
        OperatorKind.DEBATE: OperatorSpec(
            OperatorKind.DEBATE,
            lambda s: _base(s)
            + "Simulate two independent reasoners arguing for different solutions. Resolve their disagreement and end with FINAL.",
            1.7,
            "Internal debate.",
        ),
        OperatorKind.TOOL: OperatorSpec(
            OperatorKind.TOOL,
            lambda s: _base(s)
            + "Identify whether a deterministic calculation/tool would resolve uncertainty. Perform the calculation explicitly and end with FINAL.",
            0.9,
            "Tool-oriented reasoning placeholder.",
        ),
        OperatorKind.MERGE: OperatorSpec(
            OperatorKind.MERGE,
            lambda s: _base(s)
            + "Consolidate the strongest consistent claims in the available reasoning, remove duplication, and end with FINAL.",
            0.65,
            "Merge evidence from branches.",
        ),
        OperatorKind.COMPRESS: OperatorSpec(
            OperatorKind.COMPRESS,
            lambda s: _base(s)
            + "Compress the useful reasoning into a minimal set of verified facts, then continue to a final answer. End with FINAL.",
            0.55,
            "Compress context to save compute.",
        ),
        OperatorKind.STOP: OperatorSpec(
            OperatorKind.STOP,
            lambda s: _base(s)
            + "Return the best final answer supported by the current reasoning. End with FINAL.",
            0.2,
            "Stop and answer.",
        ),
    }
