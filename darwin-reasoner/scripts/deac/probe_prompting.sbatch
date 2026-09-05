#!/bin/bash
#SBATCH --job-name=dr_probe
#SBATCH --account=yanggrp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100-01,gpu-v100-02,gpu-v100-03,gpu-v100-04
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:50:00
#SBATCH --output=/deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner/logs/%x-%j.out

# A/B/C: is the 39.6% baseline accuracy real, or lost by the harness?
#   A raw completion prompt / 1024 tokens  -- reproduces the current pilot
#   B raw completion prompt / 3072 tokens  -- isolates truncation
#   C chat template + EOS  / 3072 tokens   -- isolates the missing chat template

set -uo pipefail
source /deac/csc/yangGrp/qianr/latent-skill-reasoning/darwin_reasoner/darwin-reasoner/scripts/deac/env.sh
cd "$REPO"

$PY scripts/deac/probe_prompting.py data/math500_pilot.jsonl 8 2>&1 \
  | grep -vE 'Processed prompts|Capturing CUDA|Loading safetensors'
echo "=== PROBE EXIT ${PIPESTATUS[0]} ==="
