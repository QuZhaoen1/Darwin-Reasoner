from __future__ import annotations

from darwin_reasoner.config import BackendConfig
from .base import InferenceBackend
from .mock import MockBackend


def build_backend(config: BackendConfig) -> InferenceBackend:
    kind = config.kind.lower()
    if kind == "mock":
        return MockBackend(model=config.model)
    if kind == "vllm":
        from .vllm_backend import VLLMBackend

        return VLLMBackend(
            model=config.model,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            use_chat_template=config.use_chat_template,
            enable_thinking=config.enable_thinking,
        )
    raise ValueError(f"Unsupported backend kind: {config.kind}")
