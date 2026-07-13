#!/usr/bin/env bash
#SBATCH --job-name=toolace8b_apibank_l1
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.toolace8b_apibank_l1.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.toolace8b_apibank_l1.err
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
export PIP_CACHE_DIR=$PROJ/.cache-c2/pip
export TMPDIR=$PROJ/tmp
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$WF/scripts:$APIBANK:${PYTHONPATH:-}

mkdir -p "$HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$BR"
. "$PROJ/.venv-c2/bin/activate"
cd "$WF"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

LIMIT=${LIMIT:-0}
TAG=${TAG:-full}
DATA_DIR=${DATA_DIR:-lv1-lv2-samples/level-1-given-desc}
SLUG=${SLUG:-$(basename "$DATA_DIR" | tr '-' '_')}
PRED=$BR/toolace8b_apibank_${SLUG}_${TAG}.jsonl
SCORED=$BR/toolace8b_apibank_${SLUG}_${TAG}.scored.json
rm -f "$PRED" "$SCORED"

ARGS=(
  --api-bank-root "$APIBANK"
  --data-dir "$DATA_DIR"
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

python scripts/eval_apibank_toolace_official.py "${ARGS[@]}"
python - <<PY
import json
path = "$SCORED"
with open(path) as handle:
    data = json.load(handle)
print(json.dumps({key: data[key] for key in ("paper_metric", "paper_ability", "total_api_calls", "correct_api_calls", "accuracy", "by_filename_level", "errors")}, indent=2))
PY
