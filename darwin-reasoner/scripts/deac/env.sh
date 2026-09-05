#!/usr/bin/env bash
# Shared environment for every DarwinReasoner job on DEAC lovelace.
# Source this; do not execute it.
#
# Each export below fixes a failure that was observed on this cluster. Removing
# any one of them reproduces the corresponding crash, so keep them together.

D=/deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner
CONDA_ENV=/deac/csc/yangGrp/qianr/envs/qwen35_l40s
REPO=$D/darwin-reasoner
PY=$D/.venv/bin/python

# 1. The compute nodes' /lib64/libstdc++.so.6 stops at CXXABI_1.3.13, but the
#    conda base env's libicui18n.so.78 needs 1.3.15. Without this, `import vllm`
#    dies with an ImportError before anything else runs.
export LD_LIBRARY_PATH=$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}

# 2. There is no system CUDA toolkit (/usr/local/cuda absent, cluster modules
#    stop at cuda12) but vLLM 0.28 JIT-compiles at engine init. The pip cu13
#    wheels match torch 2.13.0+cu130 exactly.
export CUDA_HOME=$D/.venv/lib/python3.12/site-packages/nvidia/cu13

# 3. vLLM's JIT invokes `ninja` by bare name; it lives in the venv bin.
export PATH=$D/.venv/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# 4. flashinfer 0.6.16 bundles a CCCL that nvcc 13.3 rejects outright
#    ("CUDA compiler and CUDA toolkit headers are incompatible"). Only the
#    sampler JITs -- attention captures its CUDA graphs fine -- so fall back
#    to vLLM's native sampler rather than fighting the toolchain.
export VLLM_USE_FLASHINFER_SAMPLER=0

# 5. Compute nodes have no outbound internet; Qwen3-8B is already in the HF
#    cache. Fail loudly rather than hanging on a download attempt.
export HF_HUB_OFFLINE=1

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
export PYTHONPATH=$REPO/src

# Note on why stages are always separate CLI invocations: experiment.py:99
# builds a vLLM engine inside _prepare, and pipeline.py calls _prepare four
# times in ONE process at gpu_memory_utilization=0.90 without ever freeing it.
# The second engine OOMs. Never use `cli pipeline` on a real backend.
