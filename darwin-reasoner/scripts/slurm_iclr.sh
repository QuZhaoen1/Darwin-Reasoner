#!/usr/bin/env bash
#SBATCH --job-name=darwin-iclr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
mkdir -p logs
source .venv/bin/activate
bash scripts/run_iclr.sh configs/pipeline_iclr.yaml configs/full_sweep.yaml
