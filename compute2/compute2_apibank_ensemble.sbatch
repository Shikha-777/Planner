#!/bin/bash
#SBATCH --job-name=apibank_ensemble
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.apibank_ensemble.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.apibank_ensemble.err

set -euo pipefail

export STOR=/storage3/fs1/yvorobeychik/Active/dshrestha
export PROJ=$STOR/taskdecomp-gpt-oss
export WF=$STOR/workflow
export APIBANK=$STOR/external/DAMO-ConvAI/api-bank
export BR=$WF/benchmark_results

source "$PROJ/.venv-c2/bin/activate"

export HOME=$PROJ/.home-c2
export HF_HOME=$PROJ/.hf_cache
export TRANSFORMERS_CACHE=$PROJ/.hf_cache/transformers
export HF_DATASETS_CACHE=$PROJ/.hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=$PROJ/.hf_cache/hub
export XDG_CACHE_HOME=$PROJ/.cache-c2
export TMPDIR=$PROJ/tmp
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$WF/scripts:$APIBANK:${PYTHONPATH:-}

mkdir -p "$BR" "$PROJ/logs/compute2" "$TMPDIR"

LIMIT=${APIBANK_ENSEMBLE_LIMIT:-10}
OFFSET=${APIBANK_ENSEMBLE_OFFSET:-0}
TAG=${APIBANK_ENSEMBLE_TAG:-smoke${LIMIT}_offset${OFFSET}}
OUT="$BR/apibank_ensemble_level3_${TAG}_${SLURM_JOB_ID}.jsonl"
SCORE="$BR/apibank_ensemble_level3_${TAG}_${SLURM_JOB_ID}.scored.json"

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "LIMIT=$LIMIT OFFSET=$OFFSET"
echo "OUT=$OUT"
echo "SCORE=$SCORE"
nvidia-smi || true

python "$WF/scripts/eval_apibank_testdata_ensemble.py" \
  --api-bank-root "$APIBANK" \
  --input-json "$APIBANK/test-data/level-3-batch-inf.json" \
  --level3-gt "$APIBANK/test-data/level-3.json" \
  --output "$OUT" \
  --score-output "$SCORE" \
  --limit "$LIMIT" \
  --offset "$OFFSET" \
  --max-plan-tokens "${APIBANK_ENSEMBLE_PLAN_TOKENS:-256}" \
  --max-action-tokens "${APIBANK_ENSEMBLE_ACTION_TOKENS:-180}" \
  --max-recovery-tokens "${APIBANK_ENSEMBLE_RECOVERY_TOKENS:-220}" \
  --recovery-mode "${APIBANK_ENSEMBLE_RECOVERY_MODE:-always}" \
  --validator-mode "${APIBANK_ENSEMBLE_VALIDATOR_MODE:-always}" \
  --validator-base "${APIBANK_ENSEMBLE_VALIDATOR_BASE:-Team-ACE/ToolACE-8B}" \
  --validator-max-new-tokens "${APIBANK_ENSEMBLE_VALIDATOR_TOKENS:-192}" \
  --max-error-details "${APIBANK_ENSEMBLE_MAX_ERROR_DETAILS:-80}"

echo "DONE"
