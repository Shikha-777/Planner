#!/usr/bin/env bash
#SBATCH --job-name=toolace8b_apibank_api
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.toolace8b_apibank_api.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.toolace8b_apibank_api.err
set -euo pipefail

export STOR=/storage3/fs1/yvorobeychik/Active/dshrestha
export PROJ=$STOR/taskdecomp-gpt-oss
export WF=$STOR/workflow
export APIBANK=$STOR/external/DAMO-ConvAI/api-bank
export BR=$WF/benchmark_results
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

mkdir -p "$HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" "$BR"
. "$PROJ/.venv-c2/bin/activate"
cd "$WF"

INPUT_JSON=${INPUT_JSON:?set INPUT_JSON}
TAG=${TAG:?set TAG}
LEVEL3_GT=${LEVEL3_GT:-}
LIMIT=${LIMIT:-0}
PRED=$BR/toolace8b_apibank_${TAG}.jsonl
SCORED=$BR/toolace8b_apibank_${TAG}.scored.json
rm -f "$PRED" "$SCORED"

ARGS=(
  --api-bank-root "$APIBANK"
  --input-json "$INPUT_JSON"
  --output "$PRED"
  --score-output "$SCORED"
  --base Team-ACE/ToolACE-8B
  --load-4bit
  --precision bf16
  --max-new-tokens 160
)
if [ "$LIMIT" != "0" ]; then
  ARGS+=(--limit "$LIMIT")
fi
if [ -n "$LEVEL3_GT" ]; then
  ARGS+=(--level3-gt "$LEVEL3_GT")
fi

python scripts/eval_apibank_testdata_toolace.py "${ARGS[@]}"
