from __future__ import annotations

import random
from copy import deepcopy

from darwin_reasoner.dsl import template_library
from darwin_reasoner.schema import (
    OperatorKind,
    ReasoningArchitecture,
    ReasoningEdge,
    ReasoningNode,
    ReasoningState,
)


_MUTABLE_OPERATORS = [
    OperatorKind.SOLVE,
    OperatorKind.CONTINUE,
    OperatorKind.BRANCH,
    OperatorKind.VERIFY,
    OperatorKind.CRITIQUE,
    OperatorKind.BACKTRACK,
    OperatorKind.TOOL,
    OperatorKind.DEBATE,
    OperatorKind.MERGE,
    OperatorKind.COMPRESS,
    OperatorKind.STOP,
]


class ArchitectureGenerator:
    """Generate and mutate typed reasoning DAGs.

    The generator begins from a small hand-written seed library, then accepts mined macros from the
    outer evolution loop. Search-space expansion is therefore explicit and auditable rather than
    silently changing prompts.
    """

    def __init__(self, *, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.templates = template_library()
        self.macro_prompts: dict[str, str] = {}

    def register_macro(self, name: str, compiled_prompt: str) -> None:
        self.macro_prompts[name] = compiled_prompt

    def generate(
        self,
        state: ReasoningState,
        *,
        num_candidates: int,
        mutation_rounds: int,
        max_nodes: int,
    ) -> list[ReasoningArchitecture]:
        del state  # Reserved for a learned query-conditioned generator.
        candidates: dict[str, ReasoningArchitecture] = {}
        for template in self.templates:
            candidates[template.architecture_id] = template

        while len(candidates) < num_candidates:
            base = deepcopy(self.rng.choice(list(candidates.values())))
            mutated = base
            for _ in range(max(1, mutation_rounds)):
                mutated = self._mutate(mutated, max_nodes=max_nodes)
            candidates[mutated.architecture_id] = mutated
            if len(candidates) >= num_candidates:
                break

        # Add a few explicit macro candidates once the grammar has evolved.
        for macro_name, prompt in self.macro_prompts.items():
            macro = ReasoningArchitecture(
                name=f"macro::{macro_name}",
                nodes=[
                    ReasoningNode(
                        node_id="n0",
                        operator=OperatorKind.MACRO,
                        params={"macro_name": macro_name, "compiled_prompt": prompt},
                    )
                ],
                edges=[],
                metadata={"origin": "mined_macro"},
            )
            candidates[macro.architecture_id] = macro
            if len(candidates) >= num_candidates:
                break

        return list(candidates.values())[:num_candidates]

    def _mutate(self, architecture: ReasoningArchitecture, *, max_nodes: int) -> ReasoningArchitecture:
        mutation = self.rng.choice(["replace", "insert", "append_verifier", "compress"])
        nodes = [n.model_copy(deep=True) for n in architecture.nodes]
        edges = [e.model_copy(deep=True) for e in architecture.edges]

        if mutation == "replace" and nodes:
            idx = self.rng.randrange(len(nodes))
            if nodes[idx].operator not in {OperatorKind.INPUT, OperatorKind.MACRO}:
                nodes[idx].operator = self.rng.choice(_MUTABLE_OPERATORS)

        elif mutation == "insert" and len(nodes) < max_nodes:
            op = self.rng.choice(_MUTABLE_OPERATORS[:-1])
            numeric_ids = [
                int(n.node_id[1:])
                for n in nodes
                if n.node_id.startswith("n") and n.node_id[1:].isdigit()
            ]
            next_index = max(numeric_ids, default=len(nodes) - 1) + 1
            new_id = f"n{next_index}"
            while any(n.node_id == new_id for n in nodes):
                next_index += 1
                new_id = f"n{next_index}"
            new_node = ReasoningNode(node_id=new_id, operator=op)
            # Insert before the last topological node in a chain-like manner. This keeps the
            # reference implementation valid while allowing richer DAG mutations later.
            if nodes:
                last = nodes[-1].node_id
                incoming = [e for e in edges if e.target == last]
                for e in incoming:
                    e.target = new_id
                edges.append(ReasoningEdge(source=new_id, target=last))
            nodes.append(new_node)

        elif mutation == "append_verifier" and len(nodes) < max_nodes:
            sink_ids = {n.node_id for n in nodes}
            for e in edges:
                sink_ids.discard(e.source)
            if not sink_ids and nodes:
                sink_ids = {nodes[-1].node_id}
            new_id = f"v{len(nodes)}"
            nodes.append(ReasoningNode(node_id=new_id, operator=OperatorKind.VERIFY))
            for sink in sorted(sink_ids):
                edges.append(ReasoningEdge(source=sink, target=new_id))

        elif mutation == "compress" and len(nodes) < max_nodes:
            sink_ids = {n.node_id for n in nodes}
            for e in edges:
                sink_ids.discard(e.source)
            new_id = f"c{len(nodes)}"
            nodes.append(ReasoningNode(node_id=new_id, operator=OperatorKind.COMPRESS))
            for sink in sorted(sink_ids):
                edges.append(ReasoningEdge(source=sink, target=new_id))

        return ReasoningArchitecture(
            name=f"{architecture.name}+{mutation}",
            nodes=nodes,
            edges=edges,
            metadata={**architecture.metadata, "mutation": mutation},
        )
