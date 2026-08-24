from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from darwin_reasoner.schema import GenerationResult


class InferenceBackend(ABC):
    """Abstract text generation backend."""

    @abstractmethod
    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[GenerationResult]:
        raise NotImplementedError
