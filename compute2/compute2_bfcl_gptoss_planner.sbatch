#!/bin/bash
#SBATCH --job-name=bfcl_gptoss_planner
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.bfcl_gptoss_planner.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.bfcl_gptoss_planner.err

set -euo pipefail

export STOR=/storage3/fs1/yvorobeychik/Active/dshrestha
export PROJ=$STOR/taskdecomp-gpt-oss
export WF=$STOR/workflow
export BFCL_ROOT=$WF/external/gorilla/berkeley-function-call-leaderboard
export DATA=$BFCL_ROOT/bfcl_eval/data
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
export PYTHONPATH=$WF/scripts:$BFCL_ROOT:${PYTHONPATH:-}

mkdir -p "$BR" "$PROJ/logs/compute2" "$TMPDIR"

CATEGORY=${BFCL_PLANNER_CATEGORY:-simple_python}
LIMIT=${BFCL_PLANNER_LIMIT:-20}
TAG=${BFCL_PLANNER_TAG:-gptoss_planner_${CATEGORY}}
ADAPTER=${BFCL_PLANNER_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-taskdecomp-lora}
MAX_NEW_TOKENS=${BFCL_PLANNER_MAX_NEW_TOKENS:-512}
TEMPERATURE=${BFCL_PLANNER_TEMPERATURE:-0.0}
TOP_P=${BFCL_PLANNER_TOP_P:-1.0}
DEFAULT_SYSTEM_PROMPT="You are evaluating Berkeley Function Calling Leaderboard tasks. Do not answer the user directly. If one or more provided tools match the request, output only a JSON list of function calls inside <tool_call>...</tool_call>. Each call object must have name and arguments keys. If no provided tool is relevant or required arguments are missing, output <tool_call>[]</tool_call>. Do not include explanations, final answers, or natural language."
SYSTEM_PROMPT=${BFCL_PLANNER_SYSTEM_PROMPT:-$DEFAULT_SYSTEM_PROMPT}

INPUT=$DATA/BFCL_v4_${CATEGORY}.json
ANSWERS=$DATA/possible_answer/BFCL_v4_${CATEGORY}.json
OUT=$BR/bfcl_${TAG}_${SLURM_JOB_ID}.jsonl
SCORE=$BR/bfcl_${TAG}_${SLURM_JOB_ID}.official_ast.scored.json

if [ ! -f "$INPUT" ]; then
  echo "Missing BFCL input: $INPUT" >&2
  exit 1
fi

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CATEGORY=$CATEGORY LIMIT=$LIMIT"
echo "ADAPTER=$ADAPTER"
echo "INPUT=$INPUT"
echo "OUT=$OUT"
echo "SCORE=$SCORE"
nvidia-smi || true

EVAL_ARGS=(
  --workflow "$WF"
  --bfcl-root "$BFCL_ROOT"
  --input "$INPUT"
  --output "$OUT"
  --adapter "$ADAPTER"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
)
if [ "$LIMIT" != "0" ]; then
  EVAL_ARGS+=(--limit "$LIMIT")
fi
if [ -n "$SYSTEM_PROMPT" ]; then
  EVAL_ARGS+=(--system-prompt "$SYSTEM_PROMPT")
fi

python "$PROJ/scripts/eval_bfcl_gptoss_adapter.py" "${EVAL_ARGS[@]}"

SCORE_ARGS=(
  --predictions "$OUT"
  --input "$INPUT"
  --output "$SCORE"
)
if [ -f "$ANSWERS" ]; then
  SCORE_ARGS+=(--answers "$ANSWERS")
fi

python "$WF/scripts/score_bfcl_official_ast.py" "${SCORE_ARGS[@]}"

python - <<PY
import json
path = "$SCORE"
with open(path) as f:
    data = json.load(f)
print(json.dumps(data.get("summary", data), indent=2))
PY

echo "DONE"
