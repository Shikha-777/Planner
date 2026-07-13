#!/usr/bin/env bash
#SBATCH --job-name=taskdecomp_sft
#SBATCH --partition=gpu
#SBATCH --account=compute-yvorobeychik
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/train.%j.out
#SBATCH --error=logs/train.%j.err

set -euo pipefail

cd "${PROJECT_DIR:-/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss}"
mkdir -p logs checkpoints results
bash scripts/bootstrap_env.sh

.venv/bin/python scripts/train_sft.py \
  --model "${BASE_MODEL:-Qwen/Qwen3-14B}" \
  --train data/processed/train.sft.jsonl \
  --validation data/processed/validation.sft.jsonl \
  --output-dir checkpoints/qwen3-14b-taskdecomp-lora
