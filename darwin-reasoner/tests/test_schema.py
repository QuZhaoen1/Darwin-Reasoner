import pytest

from darwin_reasoner.dsl import chain_architecture
from darwin_reasoner.schema import OperatorKind, ReasoningArchitecture, ReasoningEdge, ReasoningNode


def test_architecture_hash_stable():
    a = chain_architecture("x", [OperatorKind.SOLVE, OperatorKind.VERIFY])
    b = chain_architecture("x", [OperatorKind.SOLVE, OperatorKind.VERIFY])
    assert a.architecture_id == b.architecture_id


def test_cycle_rejected():
    with pytest.raises(ValueError):
        ReasoningArchitecture(
            name="cycle",
            nodes=[
                ReasoningNode(node_id="a", operator=OperatorKind.SOLVE),
                ReasoningNode(node_id="b", operator=OperatorKind.VERIFY),
            ],
            edges=[ReasoningEdge(source="a", target="b"), ReasoningEdge(source="b", target="a")],
        )
