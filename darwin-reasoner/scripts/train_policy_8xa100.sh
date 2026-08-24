#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/training/policy_qwen3_8b_a100.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

torchrun --standalone --nproc_per_node=8 \
  -m darwin_reasoner.training.train_policy_lora \
  --config "$CONFIG"
