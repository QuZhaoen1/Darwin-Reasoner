from __future__ import annotations

import networkx as nx
import numpy as np

from .schema import OperatorKind, ReasoningArchitecture


OPERATOR_ORDER = list(OperatorKind)


def architecture_vector(architecture: ReasoningArchitecture) -> np.ndarray:
    counts = {kind.value: 0.0 for kind in OPERATOR_ORDER}
    for node in architecture.nodes:
        counts[node.operator.value] += 1.0

    graph = architecture.to_networkx()
    n_nodes = max(1, graph.number_of_nodes())
    n_edges = graph.number_of_edges()
    depth = float(nx.dag_longest_path_length(graph)) if n_nodes > 1 else 0.0
    indegrees = [graph.in_degree(n) for n in graph.nodes] or [0]
    outdegrees = [graph.out_degree(n) for n in graph.nodes] or [0]

    vec = [counts[k.value] for k in OPERATOR_ORDER]
    vec += [
        float(n_nodes),
        float(n_edges),
        float(depth),
        float(max(indegrees)),
        float(max(outdegrees)),
        float(n_edges / n_nodes),
    ]
    return np.asarray(vec, dtype=np.float32)
