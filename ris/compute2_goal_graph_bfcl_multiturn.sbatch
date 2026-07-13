#!/bin/bash
# Official BFCL multi-turn runner for the goal-graph runtime.

#SBATCH --job-name=goal_graph_mt
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.goal_graph_mt.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.goal_graph_mt.err

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
export PYTHONPATH="src:compute2:scripts:.:$WF/scripts:$BFCL_ROOT:${PYTHONPATH:-}"

mkdir -p "$BR" "$BR/goal_graph_runtime" "$PROJ/logs/compute2" "$TMPDIR"

CATEGORY="${BFCL_MT_CATEGORY:-multi_turn_base}"
LIMIT="${BFCL_MT_LIMIT:-1}"
OFFSET="${BFCL_MT_OFFSET:-0}"
TAG="${BFCL_MT_TAG:-goal_graph_mt_${CATEGORY}_${SLURM_JOB_ID}}"
MODEL="${BFCL_MT_MODEL:-openai/gpt-oss-20b}"
MAX_NEW_TOKENS="${BFCL_MT_MAX_NEW_TOKENS:-700}"
REPAIR_ATTEMPTS="${BFCL_MT_REPAIR_ATTEMPTS:-0}"
PLANNER_MODE="${BFCL_MT_PLANNER_MODE:-stepwise}"
MAX_STEPS_PER_TURN="${BFCL_MT_MAX_STEPS_PER_TURN:-4}"
INCLUDE_LOGS="${BFCL_MT_INCLUDE_LOGS:-1}"
STRICT_READ_ONLY="${BFCL_MT_STRICT_READ_ONLY:-0}"
BINDER_FALLBACK="${BFCL_MT_BINDER_FALLBACK:-1}"
STOP_AFTER_MULTICALL_SUCCESS="${BFCL_MT_STOP_AFTER_MULTICALL_SUCCESS:-1}"

PREFIX="$BR/goal_graph_runtime/${TAG}"
PREDICTIONS="${PREFIX}.bfcl_multiturn.predictions.jsonl"
SCORE="${PREFIX}.bfcl_multiturn.official.scored.json"

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CATEGORY=$CATEGORY LIMIT=$LIMIT OFFSET=$OFFSET"
echo "MODEL=$MODEL PLANNER_MODE=$PLANNER_MODE"
echo "PREDICTIONS=$PREDICTIONS"
echo "SCORE=$SCORE"
nvidia-smi || true

ARGS=(
  --category "$CATEGORY"
  --output "$PREDICTIONS"
  --score-output "$SCORE"
  --model "$MODEL"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --repair-attempts "$REPAIR_ATTEMPTS"
  --planner-mode "$PLANNER_MODE"
  --max-steps-per-turn "$MAX_STEPS_PER_TURN"
  --offset "$OFFSET"
)

if [ "$LIMIT" != "0" ]; then
  ARGS+=(--limit "$LIMIT")
fi
if [ "$INCLUDE_LOGS" = "1" ]; then
  ARGS+=(--include-logs)
fi
if [ "$STRICT_READ_ONLY" = "1" ]; then
  ARGS+=(--strict-read-only)
fi
if [ "$BINDER_FALLBACK" != "1" ]; then
  ARGS+=(--no-binder-fallback)
fi
if [ "$STOP_AFTER_MULTICALL_SUCCESS" = "1" ]; then
  ARGS+=(--stop-after-multicall-success)
else
  ARGS+=(--no-stop-after-multicall-success)
fi

python scripts/eval_goal_graph_bfcl_multiturn.py "${ARGS[@]}"

echo "BFCL_MULTITURN_PREDICTIONS_PATH=$PREDICTIONS"
echo "BFCL_MULTITURN_SCORE_PATH=$SCORE"
echo "DONE"
