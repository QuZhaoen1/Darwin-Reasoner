from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from .config import TrackingConfig


class Tracker(AbstractContextManager):
    def __init__(self, config: TrackingConfig, run_name: str, full_config: dict[str, Any]) -> None:
        self.config = config
        self.run_name = run_name
        self.full_config = full_config
        self.run = None

    def __enter__(self) -> "Tracker":
        if self.config.wandb:
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError("wandb tracking enabled but wandb is not installed") from exc
            self.run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=self.run_name,
                config=self.full_config,
                tags=self.config.tags,
            )
        return self

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(payload, step=step)

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        if self.run is not None:
            self.run.finish()
        return None
