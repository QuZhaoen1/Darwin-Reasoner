from __future__ import annotations

from collections.abc import Sequence

from darwin_reasoner.schema import GenerationResult
from .base import InferenceBackend


class VLLMBackend(InferenceBackend):
    """Offline vLLM backend.

    The model is created lazily so importing the project remains cheap on CPU-only machines.
    Multi-GPU inference uses vLLM tensor parallelism. For small models, launch several
    independent experiment processes with disjoint CUDA_VISIBLE_DEVICES values if data
    parallel throughput is preferred.
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
        max_model_len: int = 16384,
        use_chat_template: bool = False,
        enable_thinking: bool = False,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Install GPU extras with: pip install -e '.[gpu]'"
            ) from exc

        self.model_name = model
        self.use_chat_template = use_chat_template
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        if use_chat_template:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=False,
        )

    def _render(self, prompts: Sequence[str]) -> list[str]:
        """Wrap operator prompts in the model's chat format when configured.

        Operator prompts are written as bare instructions, which an instruct-tuned
        model never terminates: it answers and then repeats until max_tokens.
        """
        if not self.use_chat_template or self._tokenizer is None:
            return list(prompts)
        return [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            for p in prompts
        ]

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[GenerationResult]:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        outputs = self._llm.generate(self._render(prompts), params)
        results: list[GenerationResult] = []
        for output in outputs:
            candidate = output.outputs[0]
            prompt_token_ids = getattr(output, "prompt_token_ids", None) or []
            token_ids = getattr(candidate, "token_ids", None) or []
            results.append(
                GenerationResult(
                    text=candidate.text,
                    prompt_tokens=len(prompt_token_ids),
                    completion_tokens=len(token_ids),
                    # vLLM offline objects do not expose a stable per-request latency field.
                    latency_s=0.0,
                    metadata={"backend": "vllm", "model": self.model_name},
                )
            )
        return results
