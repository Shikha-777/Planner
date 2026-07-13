#!/usr/bin/env bash
# Replay saved goal-graph planner states without tau's user simulator.

#SBATCH --job-name=tau_goal_graph_replay
#SBATCH --partition=general-gpu
#SBATCH --account=compute2-yvorobeychik
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/compute2/%j.tau_goal_graph_replay.out
#SBATCH --error=logs/compute2/%j.tau_goal_graph_replay.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
export WF="${WF:-$STOR/workflow}"
export BR="${BR:-$WF/benchmark_results}"

cd "$PROJ"
mkdir -p logs/compute2 "$BR" "$PROJ/tmp" "$PROJ/.home-c2" "$PROJ/.hf_cache" "$PROJ/.cache-c2"

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
export PYTHONPATH="$PROJ/src:$PROJ/scripts:$PROJ/compute2:${PYTHONPATH:-}"

source "$PROJ/.venv-c2/bin/activate"

DEBUG_JSONL="${DEBUG_JSONL:?Set DEBUG_JSONL to an instrumented tau debug JSONL file.}"
REPLAY_OUTPUT="${REPLAY_OUTPUT:?Set REPLAY_OUTPUT for the replay JSONL result.}"
REPLAY_STEPS="${REPLAY_STEPS:-}"
MODEL="${TAU_GOAL_GRAPH_MODEL:-openai/gpt-oss-20b}"
MAX_NEW_TOKENS="${TAU_GOAL_GRAPH_MAX_NEW_TOKENS:-900}"

ARGS=(
  "$PROJ/scripts/tau_goal_graph_replay.py"
  --debug-jsonl "$DEBUG_JSONL"
  --output "$REPLAY_OUTPUT"
  --model "$MODEL"
  --max-new-tokens "$MAX_NEW_TOKENS"
)
if [ -n "$REPLAY_STEPS" ]; then
  ARGS+=(--steps "$REPLAY_STEPS")
fi

echo "Replaying tau planner states: debug=$DEBUG_JSONL steps=${REPLAY_STEPS:-all} output=$REPLAY_OUTPUT"
python "${ARGS[@]}"
