from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field, model_validator


class OperatorKind(str, Enum):
    INPUT = "input"
    SOLVE = "solve"
    CONTINUE = "continue"
    BRANCH = "branch"
    VERIFY = "verify"
    CRITIQUE = "critique"
    BACKTRACK = "backtrack"
    TOOL = "tool"
    DEBATE = "debate"
    MERGE = "merge"
    COMPRESS = "compress"
    MACRO = "macro"
    STOP = "stop"


class Budget(BaseModel):
    max_tokens: int = 8192
    max_latency_s: float | None = None
    max_calls: int = 32
    max_nodes: int = 24


class ReasoningNode(BaseModel):
    node_id: str
    operator: OperatorKind
    prompt: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class ReasoningEdge(BaseModel):
    source: str
    target: str
    condition: str = "always"


class ReasoningArchitecture(BaseModel):
    name: str
    nodes: list[ReasoningNode]
    edges: list[ReasoningEdge]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_graph(self) -> "ReasoningArchitecture":
        ids = {n.node_id for n in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("node_id values must be unique")
        for e in self.edges:
            if e.source not in ids or e.target not in ids:
                raise ValueError(f"edge {e.source}->{e.target} references missing node")
        graph = self.to_networkx()
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("reasoning architecture must be a DAG; encode repair as a new node")
        if len(self.nodes) == 0:
            raise ValueError("architecture must contain at least one node")
        return self

    def to_networkx(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for node in self.nodes:
            graph.add_node(node.node_id, operator=node.operator.value, **node.params)
        for edge in self.edges:
            graph.add_edge(edge.source, edge.target, condition=edge.condition)
        return graph

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "operator": n.operator.value,
                    "prompt": n.prompt,
                    "params": n.params,
                }
                for n in sorted(self.nodes, key=lambda x: x.node_id)
            ],
            "edges": [
                {"source": e.source, "target": e.target, "condition": e.condition}
                for e in sorted(self.edges, key=lambda x: (x.source, x.target, x.condition))
            ],
            "metadata": self.metadata,
        }

    @property
    def architecture_id(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def operator_sequence(self) -> list[str]:
        graph = self.to_networkx()
        order = list(nx.topological_sort(graph))
        by_id = {n.node_id: n for n in self.nodes}
        return [by_id[node_id].operator.value for node_id in order]


class Problem(BaseModel):
    problem_id: str
    prompt: str
    answer: str | None = None
    domain: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReasoningState(BaseModel):
    problem_id: str
    problem_prompt: str
    transcript: str = ""
    step: int = 0
    used_tokens: int = 0
    used_calls: int = 0
    uncertainty: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def state_id(self) -> str:
        raw = f"{self.problem_id}|{self.step}|{self.transcript[-4096:]}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class GenerationResult(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Outcome(BaseModel):
    problem_id: str
    state_id: str
    architecture_id: str
    architecture_name: str
    reward: float
    correct: bool | None = None
    final_answer: str = ""
    tokens: int = 0
    latency_s: float = 0.0
    calls: int = 0
    seed: int = 0
    trace: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorldModelPrediction(BaseModel):
    reward_mean: float
    reward_std: float
    tokens_mean: float
    latency_mean: float
    success_prob: float
    metadata: dict[str, Any] = Field(default_factory=dict)
