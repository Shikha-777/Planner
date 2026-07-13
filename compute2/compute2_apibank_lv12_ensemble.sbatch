#!/bin/bash
#SBATCH --job-name=apibank_lv12_ensemble
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.apibank_lv12_ensemble.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.apibank_lv12_ensemble.err

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

DATA_DIR=${APIBANK_LV12_DATA_DIR:-lv1-lv2-samples/level-1-given-desc}
LIMIT=${APIBANK_LV12_LIMIT:-0}
TAG=${APIBANK_LV12_TAG:-lv12_ensemble}
OUT="$BR/apibank_${TAG}_${SLURM_JOB_ID}.jsonl"
SCORE="$BR/apibank_${TAG}_${SLURM_JOB_ID}.scored.json"

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_DIR=$DATA_DIR LIMIT=$LIMIT"
echo "OUT=$OUT"
echo "SCORE=$SCORE"
nvidia-smi || true

python "$WF/scripts/eval_apibank_lv12_ensemble.py" \
  --api-bank-root "$APIBANK" \
  --data-dir "$DATA_DIR" \
  --output "$OUT" \
  --score-output "$SCORE" \
  --limit "$LIMIT" \
  --max-plan-tokens "${APIBANK_LV12_PLAN_TOKENS:-220}" \
  --max-action-tokens "${APIBANK_LV12_ACTION_TOKENS:-160}" \
  --max-recovery-tokens "${APIBANK_LV12_RECOVERY_TOKENS:-180}" \
  --recovery-mode "${APIBANK_LV12_RECOVERY_MODE:-always}" \
  --toolace-mode "${APIBANK_LV12_TOOLACE_MODE:-always}" \
  --toolace-max-new-tokens "${APIBANK_LV12_TOOLACE_TOKENS:-160}" \
  --max-error-details "${APIBANK_LV12_MAX_ERROR_DETAILS:-80}"

echo "DONE"
