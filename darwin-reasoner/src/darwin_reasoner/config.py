from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .schema import Budget


class BackendConfig(BaseModel):
    kind: str = "mock"
    model: str = "mock-reasoner"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    dtype: str = "bfloat16"
    max_model_len: int = 16384
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens_per_call: int = 1024
    seed: int = 42


class SearchConfig(BaseModel):
    num_candidates: int = 16
    execute_top_k: int = 4
    mutation_rounds: int = 2
    risk_beta: float = 0.5
    uncertainty_threshold: float = 0.25
    token_penalty: float = 0.00002
    latency_penalty: float = 0.005
    diversity_bonus: float = 0.05
    use_active_design: bool = True


class CounterfactualConfig(BaseModel):
    enabled: bool = True
    repeats_per_architecture: int = 2
    max_states_per_problem: int = 3
    matched_seed_family: bool = True
    save_every: int = 10


class WorldModelConfig(BaseModel):
    kind: str = "hash_mlp"
    hidden_dim: int = 256
    text_features: int = 2048
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    ensemble_size: int = 5
    device: str = "auto"


class EvolutionConfig(BaseModel):
    enabled: bool = True
    min_support: int = 8
    min_success_lift: float = 0.03
    motif_min_len: int = 2
    motif_max_len: int = 5
    max_macros: int = 32
    prune_after_uses: int = 50
    negative_transfer_threshold: float = -0.02


class TrackingConfig(BaseModel):
    wandb: bool = False
    project: str = "darwin-reasoner"
    entity: str | None = None
    tags: list[str] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    experiment_name: str = "smoke"
    output_dir: str = "runs"
    dataset_path: str = "data/smoke.jsonl"
    domain: str = "mixed"
    limit: int | None = None
    seed: int = 42
    backend: BackendConfig = Field(default_factory=BackendConfig)
    budget: Budget = Field(default_factory=Budget)
    search: SearchConfig = Field(default_factory=SearchConfig)
    counterfactual: CounterfactualConfig = Field(default_factory=CounterfactualConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.model_validate(payload)
