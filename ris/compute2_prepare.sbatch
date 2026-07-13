#!/usr/bin/env bash
#SBATCH --job-name=taskdecomp_prepare
#SBATCH --partition=general
#SBATCH --account=compute-yvorobeychik
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prepare.%j.out
#SBATCH --error=logs/prepare.%j.err

set -euo pipefail

cd "${PROJECT_DIR:-/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss}"
mkdir -p logs data/raw data/processed
export PYTHONPATH="${PWD}/src"
python scripts/prepare_data.py --raw-dir data/raw --out-dir data/processed
