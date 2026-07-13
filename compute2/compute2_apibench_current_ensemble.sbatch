#!/bin/bash
# Run the current GPT-OSS planner/APIGen/recovery stack on Gorilla APIBench.

#SBATCH --job-name=apibench_ensemble
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.apibench_ensemble.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.apibench_ensemble.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
export WF="${WF:-$STOR/workflow}"
export BR="${BR:-$WF/benchmark_results}"

source "$PROJ/.venv-c2/bin/activate"

export HOME="$PROJ/.home-c2"
export HF_HOME="$PROJ/.hf_cache"
export TRANSFORMERS_CACHE="$PROJ/.hf_cache/transformers"
export HF_DATASETS_CACHE="$PROJ/.hf_cache/datasets"
export HUGGINGFACE_HUB_CACHE="$PROJ/.hf_cache/hub"
export XDG_CACHE_HOME="$PROJ/.cache-c2"
export TMPDIR="$PROJ/tmp"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$WF/scripts:${PYTHONPATH:-}"

mkdir -p "$BR" "$PROJ/logs/compute2" "$TMPDIR" "$BR/apibench_data"

LIMIT="${APIBENCH_LIMIT:-9}"
OFFSET="${APIBENCH_OFFSET:-0}"
PROVIDERS="${APIBENCH_PROVIDERS:-huggingface,torchhub,tensorflow}"
TAG="${APIBENCH_TAG:-smoke${LIMIT}_offset${OFFSET}}"
OUT="$BR/apibench_current_ensemble_${TAG}_${SLURM_JOB_ID}.jsonl"
SCORE="$BR/apibench_current_ensemble_${TAG}_${SLURM_JOB_ID}.scored.json"

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "PROVIDERS=$PROVIDERS LIMIT=$LIMIT OFFSET=$OFFSET"
echo "OUT=$OUT"
echo "SCORE=$SCORE"
nvidia-smi || true

python "$WF/scripts/eval_apibench_current_ensemble.py" \
  --catalog-dir "$BR/apibench_data" \
  --output "$OUT" \
  --score-output "$SCORE" \
  --providers "$PROVIDERS" \
  --limit "$LIMIT" \
  --offset "$OFFSET" \
  --retrieval-k "${APIBENCH_RETRIEVAL_K:-50}" \
  --max-plan-tokens "${APIBENCH_PLAN_TOKENS:-220}" \
  --max-action-tokens "${APIBENCH_ACTION_TOKENS:-160}" \
  --max-recovery-tokens "${APIBENCH_RECOVERY_TOKENS:-180}" \
  --max-error-details "${APIBENCH_MAX_ERROR_DETAILS:-40}"

echo "DONE"
