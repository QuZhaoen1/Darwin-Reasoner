from __future__ import annotations

import hashlib
import random
import re
import time
from collections.abc import Sequence

from darwin_reasoner.schema import GenerationResult
from .base import InferenceBackend


class MockBackend(InferenceBackend):
    """Deterministic-ish backend used for CI, pipeline testing, and zero-cost debugging.

    It is intentionally simple. It does NOT approximate a real LLM. The goal is to make
    every orchestration path runnable before expensive GPU time is available.
    """

    def __init__(self, model: str = "mock-reasoner") -> None:
        self.model = model

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        for index, prompt in enumerate(prompts):
            started = time.perf_counter()
            digest = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12], 16)
            rng = random.Random(seed + index + digest)
            lower = prompt.lower()

            answer = self._solve_tiny_arithmetic(prompt)
            if answer is None:
                # Produce diverse but stable responses for tests.
                confidence = rng.random()
                if "verify" in lower or "check" in lower:
                    text = f"Verification complete. confidence={confidence:.3f}. FINAL: 4"
                elif "branch" in lower or "alternative" in lower:
                    text = f"Alternative path explored. confidence={confidence:.3f}. FINAL: 4"
                else:
                    text = f"Reasoning trace. confidence={confidence:.3f}. FINAL: 4"
            else:
                text = f"Computed using the mock arithmetic solver. FINAL: {answer}"

            elapsed = time.perf_counter() - started
            prompt_tokens = max(1, len(prompt.split()))
            completion_tokens = min(max_tokens, max(4, len(text.split())))
            results.append(
                GenerationResult(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_s=elapsed,
                    metadata={"backend": "mock", "model": self.model},
                )
            )
        return results

    @staticmethod
    def _solve_tiny_arithmetic(prompt: str) -> str | None:
        match = re.search(r"(-?\d+)\s*([+\-*])\s*(-?\d+)", prompt)
        if not match:
            return None
        a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
        if op == "+":
            return str(a + b)
        if op == "-":
            return str(a - b)
        if op == "*":
            return str(a * b)
        return None
