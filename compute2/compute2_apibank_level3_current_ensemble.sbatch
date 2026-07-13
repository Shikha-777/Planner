#!/bin/bash
# Run API-Bank level 3 through the current ToolACE + xLAM + APIGen + TaskBench ensemble.

#SBATCH --job-name=apibank_cur_l3
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.apibank_cur_l3.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.apibank_cur_l3.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
export WF="${WF:-$STOR/workflow}"
export APIBANK="${APIBANK:-$STOR/external/DAMO-ConvAI/api-bank}"
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
export PYTHONPATH="$WF/scripts:$APIBANK:${PYTHONPATH:-}"

mkdir -p "$BR" "$PROJ/logs/compute2" "$TMPDIR"

LIMIT="${APIBANK_CURRENT_L3_LIMIT:-0}"
OFFSET="${APIBANK_CURRENT_L3_OFFSET:-0}"
TAG="${APIBANK_CURRENT_L3_TAG:-current_level3}"
MAX_NEW_TOKENS="${APIBANK_CURRENT_MAX_NEW_TOKENS:-180}"
VERIFIER_MAX_NEW_TOKENS="${APIBANK_CURRENT_VERIFIER_MAX_NEW_TOKENS:-256}"
PRECISION="${APIBANK_CURRENT_PRECISION:-auto}"
DEVICE_MAP="${APIBANK_CURRENT_DEVICE_MAP:-auto}"
LOAD_4BIT="${APIBANK_CURRENT_LOAD_4BIT:-0}"

TOOLACE_MODEL="${APIBANK_CURRENT_TOOLACE_MODEL:-Team-ACE/ToolACE-8B}"
XLAM_MODEL="${APIBANK_CURRENT_XLAM_MODEL:-Salesforce/Llama-xLAM-2-8b-fc-r}"
VERIFIER_MODEL="${APIBANK_CURRENT_VERIFIER_MODEL:-$XLAM_MODEL}"
GPTOSS_BASE="${APIBANK_CURRENT_GPTOSS_BASE:-openai/gpt-oss-20b}"
APIGEN_ADAPTER="${APIBANK_CURRENT_APIGEN_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-xlam60k-lora-2xa40-noGC1024}"
TASKBENCH_ADAPTER="${APIBANK_CURRENT_TASKBENCH_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-taskbench-taskdecomp-lora}"

OUT="$BR/apibank_${TAG}_${SLURM_JOB_ID}.jsonl"
SCORE="$BR/apibank_${TAG}_${SLURM_JOB_ID}.scored.json"

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "LIMIT=$LIMIT OFFSET=$OFFSET TAG=$TAG"
echo "TOOLACE_MODEL=$TOOLACE_MODEL"
echo "XLAM_MODEL=$XLAM_MODEL"
echo "VERIFIER_MODEL=$VERIFIER_MODEL"
echo "APIGEN_ADAPTER=$APIGEN_ADAPTER"
echo "TASKBENCH_ADAPTER=$TASKBENCH_ADAPTER"
echo "OUT=$OUT"
echo "SCORE=$SCORE"
nvidia-smi || true

LIMIT_ARGS=()
if [ "$LIMIT" != "0" ]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi
LOAD_4BIT_ARGS=()
if [ "$LOAD_4BIT" = "1" ]; then
  LOAD_4BIT_ARGS=(--load-4bit)
fi

python "$WF/scripts/eval_apibank_level3_current_ensemble.py" \
  --api-bank-root "$APIBANK" \
  --input-json "$APIBANK/test-data/level-3-batch-inf.json" \
  --level3-gt "$APIBANK/test-data/level-3.json" \
  --output "$OUT" \
  --score-output "$SCORE" \
  --offset "$OFFSET" \
  --toolace-model "$TOOLACE_MODEL" \
  --xlam-model "$XLAM_MODEL" \
  --verifier-model "$VERIFIER_MODEL" \
  --gptoss-base "$GPTOSS_BASE" \
  --apigen-adapter "$APIGEN_ADAPTER" \
  --taskbench-adapter "$TASKBENCH_ADAPTER" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --verifier-max-new-tokens "$VERIFIER_MAX_NEW_TOKENS" \
  --precision "$PRECISION" \
  --device-map "$DEVICE_MAP" \
  --trust-remote-code \
  --max-error-details "${APIBANK_CURRENT_MAX_ERROR_DETAILS:-80}" \
  "${LOAD_4BIT_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"

echo "DONE"
