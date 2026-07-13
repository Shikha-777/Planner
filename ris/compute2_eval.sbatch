#!/usr/bin/env bash
#SBATCH --job-name=taskdecomp_eval
#SBATCH --partition=gpu
#SBATCH --account=compute-yvorobeychik
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval.%j.out
#SBATCH --error=logs/eval.%j.err

set -euo pipefail

cd "${PROJECT_DIR:-/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss}"
mkdir -p logs results
bash scripts/bootstrap_env.sh

.venv/bin/python scripts/predict_jsonl.py \
  --model "${MODEL_PATH:-checkpoints/qwen3-14b-taskdecomp-lora}" \
  --input data/processed/test.eval.jsonl \
  --output results/predictions.jsonl

.venv/bin/python -m taskdecomp.evaluate \
  --predictions results/predictions.jsonl \
  --references data/processed/test.eval.jsonl \
  --out results/metrics.json
