from __future__ import annotations

import time
from collections import defaultdict

import networkx as nx

from .backends.base import InferenceBackend
from .config import BackendConfig
from .operators import default_operator_registry
from .schema import (
    Budget,
    Outcome,
    Problem,
    ReasoningArchitecture,
    ReasoningState,
)
from .verifier import verify_exact


class ArchitectureExecutor:
    def __init__(self, backend: InferenceBackend, backend_config: BackendConfig) -> None:
        self.backend = backend
        self.backend_config = backend_config
        self.registry = default_operator_registry()

    def execute(
        self,
        problem: Problem,
        architecture: ReasoningArchitecture,
        budget: Budget,
        *,
        seed: int,
        initial_state: ReasoningState | None = None,
    ) -> Outcome:
        state = initial_state or ReasoningState(
            problem_id=problem.problem_id,
            problem_prompt=problem.prompt,
        )
        graph = architecture.to_networkx()
        by_id = {node.node_id: node for node in architecture.nodes}
        outputs: dict[str, str] = {}
        trace: list[dict] = []
        used_tokens = state.used_tokens
        used_calls = state.used_calls
        started = time.perf_counter()

        for node_id in nx.topological_sort(graph):
            if used_calls >= budget.max_calls or used_tokens >= budget.max_tokens:
                break
            node = by_id[node_id]
            spec = self.registry.get(node.operator)
            if spec is None:
                # A macro node carries a compiled routine, but still receives the current problem/state.
                routine = str(node.params.get("compiled_prompt", node.prompt or "Solve and verify."))
                prompt = (
                    f"Problem:\n{state.problem_prompt}\n\n"
                    f"Current reasoning:\n{state.transcript or '[empty]'}\n\n"
                    f"Reusable reasoning routine:\n{routine}"
                )
            else:
                parent_outputs = []
                for parent in graph.predecessors(node_id):
                    if parent in outputs:
                        parent_outputs.append(f"[{parent}]\n{outputs[parent]}")
                parent_context = "\n\n".join(parent_outputs)
                local_state = state.model_copy(
                    update={
                        "transcript": "\n\n".join(
                            value for value in [state.transcript, parent_context] if value
                        ),
                        "used_tokens": used_tokens,
                        "used_calls": used_calls,
                    }
                )
                prompt = spec.prompt_builder(local_state)
                if node.prompt:
                    prompt += f"\n\nNode instruction:\n{node.prompt}"

            remaining_tokens = max(1, budget.max_tokens - used_tokens)
            call_budget = min(self.backend_config.max_tokens_per_call, remaining_tokens)
            generation = self.backend.generate(
                [prompt],
                max_tokens=call_budget,
                temperature=self.backend_config.temperature,
                top_p=self.backend_config.top_p,
                seed=seed + used_calls,
            )[0]
            outputs[node_id] = generation.text
            used_calls += 1
            used_tokens += generation.total_tokens
            trace.append(
                {
                    "node_id": node_id,
                    "operator": node.operator.value,
                    "text": generation.text,
                    "prompt_tokens": generation.prompt_tokens,
                    "completion_tokens": generation.completion_tokens,
                    "latency_s": generation.latency_s,
                }
            )

        final_text = trace[-1]["text"] if trace else state.transcript
        verification = verify_exact(final_text, problem.answer)
        elapsed = time.perf_counter() - started
        return Outcome(
            problem_id=problem.problem_id,
            state_id=state.state_id,
            architecture_id=architecture.architecture_id,
            architecture_name=architecture.name,
            reward=verification.reward,
            correct=verification.correct,
            final_answer=verification.predicted_answer,
            tokens=used_tokens - state.used_tokens,
            latency_s=elapsed,
            calls=used_calls - state.used_calls,
            seed=seed,
            trace=trace,
            metadata={
                "verification_reason": verification.reason,
                "architecture": architecture.canonical_dict(),
            },
        )

    def execute_many(
        self,
        problem: Problem,
        architectures: list[ReasoningArchitecture],
        budget: Budget,
        *,
        seeds: list[int],
        initial_state: ReasoningState | None = None,
    ) -> list[Outcome]:
        """Reference implementation.

        This deliberately prioritizes reproducibility over maximum throughput. On the A100 server,
        candidate prompts are still continuously batched inside vLLM. A later optimization can fuse
        topological layers across architectures without changing the experiment protocol.
        """
        if len(seeds) != len(architectures):
            raise ValueError("seeds and architectures must have the same length")
        return [
            self.execute(problem, architecture, budget, seed=seed, initial_state=initial_state)
            for architecture, seed in zip(architectures, seeds, strict=True)
        ]
