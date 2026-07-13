#!/bin/bash
# Official BFCL multi-turn runner for the current tool-calling ensemble.

#SBATCH --job-name=bfcl_mt_current
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.bfcl_mt_current.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.bfcl_mt_current.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
export WF="${WF:-$STOR/workflow}"
export BFCL_ROOT="${BFCL_ROOT:-$WF/external/gorilla/berkeley-function-call-leaderboard}"
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
export PYTHONPATH="$WF/scripts:$BFCL_ROOT:${PYTHONPATH:-}"

mkdir -p "$BR" "$PROJ/logs/compute2" "$TMPDIR"

CATEGORY="${BFCL_MT_CATEGORY:-multi_turn_base}"
LIMIT="${BFCL_MT_LIMIT:-1}"
OFFSET="${BFCL_MT_OFFSET:-0}"
TAG="${BFCL_MT_TAG:-current_ensemble_${CATEGORY}}"
SOURCES="${BFCL_MT_SOURCES:-toolace,xlam,gptoss_apigen,taskbench}"
MAX_NEW_TOKENS="${BFCL_MT_MAX_NEW_TOKENS:-256}"
VERIFIER_MAX_NEW_TOKENS="${BFCL_MT_VERIFIER_MAX_NEW_TOKENS:-192}"
MAX_STEPS_PER_TURN="${BFCL_MT_MAX_STEPS_PER_TURN:-4}"
DUPLICATE_PENALTY_EXEMPT_TOOLS="${BFCL_MT_DUPLICATE_PENALTY_EXEMPT_TOOLS:-cd}"
EMPTY_TURN_ACTION_PENALTY="${BFCL_MT_EMPTY_TURN_ACTION_PENALTY:-12}"
STOP_AFTER_MULTICALL_SUCCESS="${BFCL_MT_STOP_AFTER_MULTICALL_SUCCESS:-1}"
PRECISION="${BFCL_MT_PRECISION:-auto}"
DEVICE_MAP="${BFCL_MT_DEVICE_MAP:-auto}"
LOAD_4BIT="${BFCL_MT_LOAD_4BIT:-0}"
INCLUDE_LOGS="${BFCL_MT_INCLUDE_LOGS:-1}"

TOOLACE_MODEL="${BFCL_MT_TOOLACE_MODEL:-Team-ACE/ToolACE-8B}"
XLAM_MODEL="${BFCL_MT_XLAM_MODEL:-Salesforce/Llama-xLAM-2-8b-fc-r}"
VERIFIER_MODEL="${BFCL_MT_VERIFIER_MODEL:-$XLAM_MODEL}"
GPTOSS_BASE="${BFCL_MT_GPTOSS_BASE:-openai/gpt-oss-20b}"
APIGEN_ADAPTER="${BFCL_MT_APIGEN_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-xlam60k-lora-2xa40-noGC1024}"
TASKBENCH_ADAPTER="${BFCL_MT_TASKBENCH_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-taskbench-taskdecomp-lora}"

PREFIX="$BR/bfcl_mt_${TAG}_${SLURM_JOB_ID}"
PREDICTIONS="${PREFIX}.jsonl"
SCORE="${PREFIX}.official_multi_turn.scored.json"

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CATEGORY=$CATEGORY LIMIT=$LIMIT OFFSET=$OFFSET"
echo "SOURCES=$SOURCES"
echo "STOP_AFTER_MULTICALL_SUCCESS=$STOP_AFTER_MULTICALL_SUCCESS"
echo "VERIFIER_MODEL=$VERIFIER_MODEL"
echo "PREDICTIONS=$PREDICTIONS"
echo "SCORE=$SCORE"
nvidia-smi || true

ARGS=(
  --category "$CATEGORY"
  --output "$PREDICTIONS"
  --score-output "$SCORE"
  --sources "$SOURCES"
  --toolace-model "$TOOLACE_MODEL"
  --xlam-model "$XLAM_MODEL"
  --verifier-model "$VERIFIER_MODEL"
  --gptoss-base "$GPTOSS_BASE"
  --apigen-adapter "$APIGEN_ADAPTER"
  --taskbench-adapter "$TASKBENCH_ADAPTER"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --verifier-max-new-tokens "$VERIFIER_MAX_NEW_TOKENS"
  --max-steps-per-turn "$MAX_STEPS_PER_TURN"
  --duplicate-penalty-exempt-tools "$DUPLICATE_PENALTY_EXEMPT_TOOLS"
  --empty-turn-action-penalty "$EMPTY_TURN_ACTION_PENALTY"
  --precision "$PRECISION"
  --device-map "$DEVICE_MAP"
  --offset "$OFFSET"
)

if [ "$LIMIT" != "0" ]; then
  ARGS+=(--limit "$LIMIT")
fi
if [ "$LOAD_4BIT" = "1" ]; then
  ARGS+=(--load-4bit)
fi
if [ "$INCLUDE_LOGS" = "1" ]; then
  ARGS+=(--include-logs)
fi
if [ "$STOP_AFTER_MULTICALL_SUCCESS" = "1" ]; then
  ARGS+=(--stop-after-multicall-success)
else
  ARGS+=(--no-stop-after-multicall-success)
fi

python "$WF/scripts/eval_bfcl_multiturn_current_ensemble.py" "${ARGS[@]}"

echo "DONE"
