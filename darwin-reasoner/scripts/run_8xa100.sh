#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/a100_8gpu_32b.yaml}"
WM="${2:-runs/iclr_pipeline/e3_world_model/world_model.pt}"
MACROS="${3:-runs/iclr_pipeline/e3_world_model/admitted_macros.json}"
RUN_DIR="${4:-runs/manual_8xa100}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false

python -m darwin_reasoner.cli run \
  --config "${CONFIG}" \
  --world-model "${WM}" \
  --macros "${MACROS}" \
  --run-dir "${RUN_DIR}"
