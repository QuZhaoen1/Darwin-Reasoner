from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from darwin_reasoner.architecture_features import architecture_vector
from darwin_reasoner.schema import (
    Budget,
    ReasoningArchitecture,
    ReasoningState,
    WorldModelPrediction,
)


@dataclass
class WorldModelExample:
    state_text: str
    architecture: ReasoningArchitecture
    budget: Budget
    reward: float
    tokens: float
    latency_s: float


class _Head(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnsembleWorldModel:
    """Lightweight uncertainty-aware world model.

    The default implementation intentionally uses hashed text features so the research pipeline can
    be debugged without downloading another encoder. The interface is designed so a stronger
    transformer state encoder can replace it without touching the planner or data protocol.
    """

    def __init__(
        self,
        *,
        text_features: int = 2048,
        hidden_dim: int = 256,
        ensemble_size: int = 5,
        device: str = "auto",
    ) -> None:
        self.text_features = text_features
        self.hidden_dim = hidden_dim
        self.ensemble_size = ensemble_size
        self.vectorizer = HashingVectorizer(
            n_features=text_features,
            alternate_sign=False,
            norm="l2",
        )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.models: list[_Head] = []
        self.input_dim: int | None = None

    @staticmethod
    def _budget_vector(budget: Budget) -> np.ndarray:
        latency = 0.0 if budget.max_latency_s is None else float(budget.max_latency_s)
        return np.asarray(
            [
                np.log1p(budget.max_tokens),
                np.log1p(budget.max_calls),
                np.log1p(budget.max_nodes),
                np.log1p(latency),
            ],
            dtype=np.float32,
        )

    def _features(
        self,
        state_texts: list[str],
        architectures: list[ReasoningArchitecture],
        budgets: list[Budget],
    ) -> np.ndarray:
        text_matrix = self.vectorizer.transform(state_texts).astype(np.float32).toarray()
        arch_matrix = np.stack([architecture_vector(a) for a in architectures])
        budget_matrix = np.stack([self._budget_vector(b) for b in budgets])
        return np.concatenate([text_matrix, arch_matrix, budget_matrix], axis=1).astype(np.float32)

    def fit(
        self,
        examples: list[WorldModelExample],
        *,
        epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> dict[str, float]:
        if not examples:
            raise ValueError("No world-model examples supplied")

        x = self._features(
            [e.state_text for e in examples],
            [e.architecture for e in examples],
            [e.budget for e in examples],
        )
        reward = np.asarray([e.reward for e in examples], dtype=np.float32)
        tokens = np.log1p(np.asarray([e.tokens for e in examples], dtype=np.float32))
        latency = np.log1p(np.asarray([e.latency_s for e in examples], dtype=np.float32))
        y = np.stack([reward, tokens, latency], axis=1)

        self.input_dim = x.shape[1]
        self.models = []
        rng = np.random.default_rng(seed)
        final_losses: list[float] = []

        for member in range(self.ensemble_size):
            torch.manual_seed(seed + member)
            indices = rng.integers(0, len(examples), size=len(examples))
            x_boot = torch.from_numpy(x[indices])
            y_boot = torch.from_numpy(y[indices])
            loader = DataLoader(
                TensorDataset(x_boot, y_boot),
                batch_size=min(batch_size, len(examples)),
                shuffle=True,
            )
            model = _Head(self.input_dim, self.hidden_dim).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            loss_value = 0.0
            for _ in range(epochs):
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = model(xb)
                    reward_loss = nn.functional.binary_cross_entropy_with_logits(pred[:, 0], yb[:, 0])
                    token_loss = nn.functional.smooth_l1_loss(pred[:, 1], yb[:, 1])
                    latency_loss = nn.functional.smooth_l1_loss(pred[:, 2], yb[:, 2])
                    loss = reward_loss + 0.2 * token_loss + 0.2 * latency_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    loss_value = float(loss.detach().cpu())
            model.eval()
            self.models.append(model)
            final_losses.append(loss_value)

        return {"loss": float(np.mean(final_losses)), "ensemble_size": float(len(self.models))}

    @torch.no_grad()
    def predict_many(
        self,
        state: ReasoningState,
        architectures: list[ReasoningArchitecture],
        budget: Budget,
    ) -> list[WorldModelPrediction]:
        if not self.models:
            # High uncertainty prior: planner should fall back to real execution / active design.
            return [
                WorldModelPrediction(
                    reward_mean=0.5,
                    reward_std=0.5,
                    tokens_mean=max(1.0, budget.max_tokens * 0.5),
                    latency_mean=0.0,
                    success_prob=0.5,
                    metadata={"untrained": True},
                )
                for _ in architectures
            ]

        state_text = f"{state.problem_prompt}\n{state.transcript}"
        x = self._features(
            [state_text] * len(architectures),
            architectures,
            [budget] * len(architectures),
        )
        xb = torch.from_numpy(x).to(self.device)
        member_outputs = []
        for model in self.models:
            raw = model(xb)
            reward = torch.sigmoid(raw[:, 0])
            tokens = torch.expm1(raw[:, 1]).clamp_min(0)
            latency = torch.expm1(raw[:, 2]).clamp_min(0)
            member_outputs.append(torch.stack([reward, tokens, latency], dim=1).cpu().numpy())
        arr = np.stack(member_outputs, axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        predictions: list[WorldModelPrediction] = []
        for i in range(len(architectures)):
            predictions.append(
                WorldModelPrediction(
                    reward_mean=float(mean[i, 0]),
                    reward_std=float(std[i, 0]),
                    tokens_mean=float(mean[i, 1]),
                    latency_mean=float(mean[i, 2]),
                    success_prob=float(mean[i, 0]),
                )
            )
        return predictions

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.input_dim is None:
            raise RuntimeError("Cannot save an uninitialized world model")
        torch.save(
            {
                "text_features": self.text_features,
                "hidden_dim": self.hidden_dim,
                "ensemble_size": self.ensemble_size,
                "input_dim": self.input_dim,
                "state_dicts": [m.state_dict() for m in self.models],
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str = "auto") -> "EnsembleWorldModel":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(
            text_features=int(payload["text_features"]),
            hidden_dim=int(payload["hidden_dim"]),
            ensemble_size=int(payload["ensemble_size"]),
            device=device,
        )
        obj.input_dim = int(payload["input_dim"])
        obj.models = []
        for state_dict in payload["state_dicts"]:
            model = _Head(obj.input_dim, obj.hidden_dim).to(obj.device)
            model.load_state_dict(state_dict)
            model.eval()
            obj.models.append(model)
        return obj


def load_examples_from_jsonl(path: str | Path, default_budget: Budget) -> list[WorldModelExample]:
    examples: list[WorldModelExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            architecture = ReasoningArchitecture.model_validate(row["architecture"])
            budget = Budget.model_validate(row.get("budget", default_budget.model_dump()))
            examples.append(
                WorldModelExample(
                    state_text=row.get("state_text", ""),
                    architecture=architecture,
                    budget=budget,
                    reward=float(row["reward"]),
                    tokens=float(row.get("tokens", 0.0)),
                    latency_s=float(row.get("latency_s", 0.0)),
                )
            )
    return examples
