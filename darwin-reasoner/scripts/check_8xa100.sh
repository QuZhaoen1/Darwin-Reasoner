#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
python scripts/check_8xa100.py
torchrun --standalone --nproc_per_node=8 scripts/nccl_smoke.py
