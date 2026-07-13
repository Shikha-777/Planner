#!/bin/bash
# Run BFCL through ToolACE + original xLAM + GPT-OSS APIGen/xLAM + TaskBench,
# then select with an optional xLAM verifier.

#SBATCH --job-name=bfcl_tool_ensemble
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.bfcl_tool_ensemble.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.bfcl_tool_ensemble.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
export WF="${WF:-$STOR/workflow}"
export BFCL_ROOT="${BFCL_ROOT:-$WF/external/gorilla/berkeley-function-call-leaderboard}"
export DATA="${DATA:-$BFCL_ROOT/bfcl_eval/data}"
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
export PYTHONPATH="$PROJ/src:$WF/scripts:$BFCL_ROOT:${PYTHONPATH:-}"

mkdir -p "$BR" "$PROJ/logs/compute2" "$TMPDIR"

CATEGORY="${BFCL_ENSEMBLE_CATEGORY:-simple_python}"
LIMIT="${BFCL_ENSEMBLE_LIMIT:-20}"
TAG="${BFCL_ENSEMBLE_TAG:-tool_ensemble_${CATEGORY}}"
MAX_NEW_TOKENS="${BFCL_ENSEMBLE_MAX_NEW_TOKENS:-512}"
VERIFIER_MAX_NEW_TOKENS="${BFCL_ENSEMBLE_VERIFIER_MAX_NEW_TOKENS:-256}"
PRECISION="${BFCL_ENSEMBLE_PRECISION:-auto}"
DEVICE_MAP="${BFCL_ENSEMBLE_DEVICE_MAP:-auto}"
LOAD_4BIT="${BFCL_ENSEMBLE_LOAD_4BIT:-0}"
CAPABILITY_PLANNER_MODE="${BFCL_ENSEMBLE_CAPABILITY_PLANNER:-auto}"

TOOLACE_MODEL="${BFCL_ENSEMBLE_TOOLACE_MODEL:-Team-ACE/ToolACE-8B}"
XLAM_MODEL="${BFCL_ENSEMBLE_XLAM_MODEL:-Salesforce/Llama-xLAM-2-8b-fc-r}"
VERIFIER_MODEL="${BFCL_ENSEMBLE_VERIFIER_MODEL:-$XLAM_MODEL}"
GPTOSS_BASE="${BFCL_ENSEMBLE_GPTOSS_BASE:-openai/gpt-oss-20b}"
APIGEN_ADAPTER="${BFCL_ENSEMBLE_APIGEN_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-xlam60k-lora-2xa40-noGC1024}"
TASKBENCH_ADAPTER="${BFCL_ENSEMBLE_TASKBENCH_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-taskbench-taskdecomp-lora}"

INPUT="$DATA/BFCL_v4_${CATEGORY}.json"
ANSWERS="$DATA/possible_answer/BFCL_v4_${CATEGORY}.json"
PREFIX="$BR/bfcl_${TAG}_${SLURM_JOB_ID}"

TOOLACE_OUT="${PREFIX}.toolace.jsonl"
XLAM_OUT="${PREFIX}.xlam.jsonl"
APIGEN_OUT="${PREFIX}.gptoss_apigen.jsonl"
TASKBENCH_OUT="${PREFIX}.taskbench.jsonl"
ENSEMBLE_OUT="${PREFIX}.ensemble.jsonl"
SCORE="${PREFIX}.ensemble.official_ast.scored.json"

if [ ! -f "$INPUT" ]; then
  echo "Missing BFCL input: $INPUT" >&2
  exit 1
fi

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CATEGORY=$CATEGORY LIMIT=$LIMIT"
echo "TOOLACE_MODEL=$TOOLACE_MODEL"
echo "XLAM_MODEL=$XLAM_MODEL"
echo "VERIFIER_MODEL=$VERIFIER_MODEL"
echo "APIGEN_ADAPTER=$APIGEN_ADAPTER"
echo "TASKBENCH_ADAPTER=$TASKBENCH_ADAPTER"
echo "INPUT=$INPUT"
echo "ENSEMBLE_OUT=$ENSEMBLE_OUT"
echo "SCORE=$SCORE"
nvidia-smi || true

COMMON_LIMIT=()
if [ "$LIMIT" != "0" ]; then
  COMMON_LIMIT=(--limit "$LIMIT")
fi
LOAD_4BIT_ARGS=()
if [ "$LOAD_4BIT" = "1" ]; then
  LOAD_4BIT_ARGS=(--load-4bit)
fi

echo "__TOOLACE__"
python "$WF/scripts/eval_bfcl_toolace_official.py" \
  --input "$INPUT" \
  --output "$TOOLACE_OUT" \
  --base "$TOOLACE_MODEL" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --precision "$PRECISION" \
  --device-map "$DEVICE_MAP" \
  "${LOAD_4BIT_ARGS[@]}" \
  "${COMMON_LIMIT[@]}"

echo "__ORIGINAL_XLAM__"
python "$WF/scripts/eval_bfcl_hf_adapter.py" \
  --input "$INPUT" \
  --output "$XLAM_OUT" \
  --base "$XLAM_MODEL" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --precision "$PRECISION" \
  --device-map "$DEVICE_MAP" \
  "${LOAD_4BIT_ARGS[@]}" \
  "${COMMON_LIMIT[@]}"

echo "__GPTOSS_APIGEN_XLAM_LORA__"
python "$WF/scripts/eval_bfcl_hf_adapter.py" \
  --input "$INPUT" \
  --output "$APIGEN_OUT" \
  --base "$GPTOSS_BASE" \
  --adapter "$APIGEN_ADAPTER" \
  --tokenizer "$APIGEN_ADAPTER" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --precision "$PRECISION" \
  --device-map "$DEVICE_MAP" \
  --trust-remote-code \
  "${LOAD_4BIT_ARGS[@]}" \
  "${COMMON_LIMIT[@]}"

echo "__GPTOSS_TASKBENCH_LORA__"
python "$WF/scripts/eval_bfcl_hf_adapter.py" \
  --input "$INPUT" \
  --output "$TASKBENCH_OUT" \
  --base "$GPTOSS_BASE" \
  --adapter "$TASKBENCH_ADAPTER" \
  --tokenizer "$TASKBENCH_ADAPTER" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --precision "$PRECISION" \
  --device-map "$DEVICE_MAP" \
  --trust-remote-code \
  "${LOAD_4BIT_ARGS[@]}" \
  "${COMMON_LIMIT[@]}"

echo "__SELECT__"
SELECT_ARGS=(
  --input "$INPUT"
  --output "$ENSEMBLE_OUT"
  --candidate "toolace:$TOOLACE_OUT"
  --candidate "xlam:$XLAM_OUT"
  --candidate "gptoss_apigen:$APIGEN_OUT"
  --candidate "taskbench:$TASKBENCH_OUT"
  --precision "$PRECISION"
  --device-map "$DEVICE_MAP"
  --verifier-max-new-tokens "$VERIFIER_MAX_NEW_TOKENS"
  --capability-planner "$CAPABILITY_PLANNER_MODE"
)
if [ "$LIMIT" != "0" ]; then
  SELECT_ARGS+=(--limit "$LIMIT")
fi
if [ -n "$VERIFIER_MODEL" ] && [ "$VERIFIER_MODEL" != "none" ]; then
  SELECT_ARGS+=(--verifier-model "$VERIFIER_MODEL")
fi
python "$WF/scripts/eval_bfcl_ensemble_select.py" "${SELECT_ARGS[@]}"

SCORE_ARGS=(
  --predictions "$ENSEMBLE_OUT"
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
