from darwin_reasoner.backends.mock import MockBackend
from darwin_reasoner.config import BackendConfig
from darwin_reasoner.dsl import chain_architecture
from darwin_reasoner.executor import ArchitectureExecutor
from darwin_reasoner.schema import Budget, OperatorKind, Problem


def test_mock_executor_solves_arithmetic():
    backend = MockBackend()
    executor = ArchitectureExecutor(backend, BackendConfig(kind="mock", max_tokens_per_call=64))
    problem = Problem(problem_id="p", prompt="Compute 2 + 2.", answer="4", domain="math")
    arch = chain_architecture("cot", [OperatorKind.SOLVE])
    outcome = executor.execute(problem, arch, Budget(max_tokens=128, max_calls=2), seed=1)
    assert outcome.correct is True
