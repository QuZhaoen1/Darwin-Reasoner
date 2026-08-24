from __future__ import annotations

from dataclasses import dataclass

from .schema import OperatorKind, ReasoningArchitecture, ReasoningEdge, ReasoningNode


@dataclass(frozen=True)
class MacroDefinition:
    name: str
    operators: tuple[OperatorKind, ...]
    success_lift: float = 0.0
    support: int = 0


def chain_architecture(name: str, operators: list[OperatorKind]) -> ReasoningArchitecture:
    nodes = [
        ReasoningNode(node_id=f"n{i}", operator=op)
        for i, op in enumerate(operators)
    ]
    edges = [
        ReasoningEdge(source=f"n{i}", target=f"n{i + 1}")
        for i in range(len(nodes) - 1)
    ]
    return ReasoningArchitecture(name=name, nodes=nodes, edges=edges)


def template_library() -> list[ReasoningArchitecture]:
    return [
        chain_architecture("cot", [OperatorKind.SOLVE]),
        chain_architecture("verify", [OperatorKind.SOLVE, OperatorKind.VERIFY]),
        chain_architecture(
            "branch_verify_merge",
            [OperatorKind.BRANCH, OperatorKind.VERIFY, OperatorKind.MERGE],
        ),
        chain_architecture(
            "critical_repair",
            [
                OperatorKind.SOLVE,
                OperatorKind.CRITIQUE,
                OperatorKind.BACKTRACK,
                OperatorKind.VERIFY,
            ],
        ),
        chain_architecture(
            "debate_verify",
            [OperatorKind.DEBATE, OperatorKind.VERIFY, OperatorKind.STOP],
        ),
        chain_architecture(
            "tool_verify",
            [OperatorKind.TOOL, OperatorKind.VERIFY],
        ),
        chain_architecture(
            "compress_continue",
            [OperatorKind.COMPRESS, OperatorKind.CONTINUE],
        ),
    ]
