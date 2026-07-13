#!/usr/bin/env bash
# Run goal-graph runtime smoke tests on BFCL and API-Bank.

#SBATCH --job-name=goal_graph_smoke
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.goal_graph_smoke.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.goal_graph_smoke.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
export WF="${WF:-$STOR/workflow}"
export APIBANK="${APIBANK:-$STOR/external/DAMO-ConvAI/api-bank}"
if [[ -z "${BFCL_ROOT:-}" ]]; then
  if [[ -d "$PROJ/external/gorilla/berkeley-function-call-leaderboard" ]]; then
    export BFCL_ROOT="$PROJ/external/gorilla/berkeley-function-call-leaderboard"
  else
    export BFCL_ROOT="$WF/external/gorilla/berkeley-function-call-leaderboard"
  fi
fi
export BR="${BR:-$WF/benchmark_results}"
cd "$PROJ"

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
export PYTHONPATH="src:compute2:scripts:.:$WF/scripts:$BFCL_ROOT:$APIBANK:${PYTHONPATH:-}"

mkdir -p "$BR" "$BR/goal_graph_runtime" "$PROJ/logs/compute2" "$TMPDIR" \
  "$HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME"

source "$PROJ/.venv-c2/bin/activate"

MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-20b}"
TAG="${TAG:-goal_graph_smoke_${SLURM_JOB_ID}}"
RUN_BFCL="${RUN_BFCL:-1}"
RUN_APIBANK="${RUN_APIBANK:-1}"

BFCL_DATA_DIR="${BFCL_DATA_DIR:-data/bfcl_v4_full}"
BFCL_CATEGORIES="${BFCL_CATEGORIES:-simple_python}"
BFCL_OFFSET="${BFCL_OFFSET:-0}"
BFCL_LIMIT="${BFCL_LIMIT:-20}"

APIBANK_LEVELS="${APIBANK_LEVELS:-level-1-api level-2-api}"
APIBANK_LIMIT="${APIBANK_LIMIT:-20}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1400}"
REPAIR_ATTEMPTS="${REPAIR_ATTEMPTS:-1}"
GOAL_GRAPH_PLANNER_MODE="${GOAL_GRAPH_PLANNER_MODE:-stepwise}"
BFCL_OFFICIAL_AST_SCORE="${BFCL_OFFICIAL_AST_SCORE:-0}"
if [[ -z "${BFCL_OFFICIAL_SCORER:-}" ]]; then
  if [[ -f "$PROJ/scripts/score_bfcl_official_ast.py" ]]; then
    BFCL_OFFICIAL_SCORER="$PROJ/scripts/score_bfcl_official_ast.py"
  else
    BFCL_OFFICIAL_SCORER="$WF/scripts/score_bfcl_official_ast.py"
  fi
fi

BFCL_OUTPUT="$BR/goal_graph_runtime/${TAG}.bfcl.metrics.json"
BFCL_REVIEW="$BR/goal_graph_runtime/${TAG}.bfcl.review.csv"
BFCL_PLANS="$BR/goal_graph_runtime/${TAG}.bfcl.plans.jsonl"
BFCL_OFFICIAL_PREDICTIONS="$BR/goal_graph_runtime/${TAG}.bfcl.official_predictions.jsonl"
APIBANK_OUTPUT="$BR/goal_graph_runtime/${TAG}.apibank.metrics.json"
APIBANK_REVIEW="$BR/goal_graph_runtime/${TAG}.apibank.review.csv"
APIBANK_PLANS="$BR/goal_graph_runtime/${TAG}.apibank.plans.jsonl"

hostname
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("device0", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), round(props.total_memory / 1024**3, 2), "GB")
PY

echo "MODEL_PATH=$MODEL_PATH"
echo "TAG=$TAG"
echo "RUN_BFCL=$RUN_BFCL RUN_APIBANK=$RUN_APIBANK"
echo "BFCL_DATA_DIR=$BFCL_DATA_DIR BFCL_CATEGORIES=$BFCL_CATEGORIES BFCL_OFFSET=$BFCL_OFFSET BFCL_LIMIT=$BFCL_LIMIT"
echo "APIBANK=$APIBANK APIBANK_LEVELS=$APIBANK_LEVELS APIBANK_LIMIT=$APIBANK_LIMIT"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS REPAIR_ATTEMPTS=$REPAIR_ATTEMPTS GOAL_GRAPH_PLANNER_MODE=$GOAL_GRAPH_PLANNER_MODE"
echo "BFCL_ROOT=$BFCL_ROOT BFCL_OFFICIAL_AST_SCORE=$BFCL_OFFICIAL_AST_SCORE BFCL_OFFICIAL_SCORER=$BFCL_OFFICIAL_SCORER"

if [[ "$RUN_BFCL" == "1" || "$RUN_BFCL" == "true" ]]; then
  BFCL_ARGS=(
    --model "$MODEL_PATH"
    --data-dir "$BFCL_DATA_DIR"
    --categories $BFCL_CATEGORIES
    --offset "$BFCL_OFFSET"
    --limit "$BFCL_LIMIT"
    --output "$BFCL_OUTPUT"
    --review-csv "$BFCL_REVIEW"
    --plans-jsonl "$BFCL_PLANS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --repair-attempts "$REPAIR_ATTEMPTS"
    --planner-mode "$GOAL_GRAPH_PLANNER_MODE"
  )
  if [[ "$BFCL_OFFICIAL_AST_SCORE" == "1" || "$BFCL_OFFICIAL_AST_SCORE" == "true" ]]; then
    BFCL_ARGS+=(--official-predictions-jsonl "$BFCL_OFFICIAL_PREDICTIONS")
  fi
  PYTHONPATH="$PYTHONPATH" python scripts/eval_goal_graph_bfcl.py "${BFCL_ARGS[@]}"
  echo "BFCL_METRICS_PATH=$BFCL_OUTPUT"
  echo "BFCL_REVIEW_PATH=$BFCL_REVIEW"
  echo "BFCL_PLANS_PATH=$BFCL_PLANS"
  if [[ "$BFCL_OFFICIAL_AST_SCORE" == "1" || "$BFCL_OFFICIAL_AST_SCORE" == "true" ]]; then
    echo "BFCL_OFFICIAL_PREDICTIONS_PATH=$BFCL_OFFICIAL_PREDICTIONS"
    for CATEGORY in $BFCL_CATEGORIES; do
      BFCL_INPUT="$BFCL_DATA_DIR/BFCL_v4_${CATEGORY}.json"
      BFCL_ANSWERS="$BFCL_DATA_DIR/possible_answer/BFCL_v4_${CATEGORY}.json"
      BFCL_SCORE="$BR/goal_graph_runtime/${TAG}.bfcl.${CATEGORY}.official_ast.scored.json"
      SCORE_ARGS=(
        --predictions "$BFCL_OFFICIAL_PREDICTIONS"
        --input "$BFCL_INPUT"
        --output "$BFCL_SCORE"
      )
      if [[ -f "$BFCL_ANSWERS" ]]; then
        SCORE_ARGS+=(--answers "$BFCL_ANSWERS")
      fi
      if [[ -f "$BFCL_OFFICIAL_SCORER" ]]; then
        PYTHONPATH="$PYTHONPATH" python "$BFCL_OFFICIAL_SCORER" "${SCORE_ARGS[@]}"
        echo "BFCL_OFFICIAL_AST_SCORE_PATH_${CATEGORY}=$BFCL_SCORE"
      else
        echo "Missing official BFCL scorer: $BFCL_OFFICIAL_SCORER" >&2
      fi
    done
  fi
fi

if [[ "$RUN_APIBANK" == "1" || "$RUN_APIBANK" == "true" ]]; then
  PYTHONPATH="$PYTHONPATH" python scripts/eval_goal_graph_apibank_routing.py \
    --model "$MODEL_PATH" \
    --api-bank-root "$APIBANK" \
    --levels $APIBANK_LEVELS \
    --limit "$APIBANK_LIMIT" \
    --output "$APIBANK_OUTPUT" \
    --review-csv "$APIBANK_REVIEW" \
    --plans-jsonl "$APIBANK_PLANS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --repair-attempts "$REPAIR_ATTEMPTS" \
    --planner-mode "$GOAL_GRAPH_PLANNER_MODE"
  echo "APIBANK_METRICS_PATH=$APIBANK_OUTPUT"
  echo "APIBANK_REVIEW_PATH=$APIBANK_REVIEW"
  echo "APIBANK_PLANS_PATH=$APIBANK_PLANS"
fi
