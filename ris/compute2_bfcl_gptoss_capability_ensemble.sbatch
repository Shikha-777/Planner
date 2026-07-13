#!/usr/bin/env bash
#SBATCH --job-name=bfcl_gptoss_cap_ens
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.bfcl_gptoss_cap_ens.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.bfcl_gptoss_cap_ens.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
export WF="${WF:-$STOR/workflow}"
export APIBANK="${APIBANK:-$STOR/external/DAMO-ConvAI/api-bank}"
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
export PYTHONPATH="$WF/scripts:$APIBANK:${PYTHONPATH:-}"

mkdir -p logs/compute2 "$BR" "$BR/bfcl_tool_binding" "$HOME" "$HF_HOME" \
  "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" \
  "$XDG_CACHE_HOME" "$TMPDIR"

source "$PROJ/.venv-c2/bin/activate"

MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-20b}"
DATA_DIR="${DATA_DIR:-data/bfcl_v4_full}"
CATEGORIES="${CATEGORIES:-simple_python multiple parallel parallel_multiple irrelevance}"
OFFSET="${OFFSET:-0}"
LIMIT="${LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1200}"
PLANNER_MODE="${PLANNER_MODE:-ensemble}"
ENSEMBLE_VARIANTS="${ENSEMBLE_VARIANTS:-chunked:baseline,chunked:lexical,chunked:slot_grounded,one_shot:semantic,rules_first:deterministic}"
SEMANTIC_SLOT_FRAME="${SEMANTIC_SLOT_FRAME:-0}"
TAG="${TAG:-bfcl_v4_gptoss_capability_ensemble_${SLURM_JOB_ID}}"
SHARD_MANIFEST="${SHARD_MANIFEST:-}"

if [[ -n "$SHARD_MANIFEST" ]]; then
  shard_index="${SLURM_ARRAY_TASK_ID:-0}"
  shard_line="$(sed -n "$((shard_index + 1))p" "$SHARD_MANIFEST")"
  if [[ -z "$shard_line" ]]; then
    echo "No shard line found for SLURM_ARRAY_TASK_ID=$shard_index in $SHARD_MANIFEST" >&2
    exit 2
  fi
  read -r SHARD_CATEGORY SHARD_OFFSET SHARD_LIMIT SHARD_TAG <<< "$shard_line"
  CATEGORIES="$SHARD_CATEGORY"
  OFFSET="$SHARD_OFFSET"
  LIMIT="$SHARD_LIMIT"
  TAG="$SHARD_TAG"
fi

OUTPUT="${OUTPUT:-$BR/bfcl_tool_binding/${TAG}.metrics.json}"
REVIEW_CSV="${REVIEW_CSV:-$BR/bfcl_tool_binding/${TAG}.review.csv}"
PLANS_JSONL="${PLANS_JSONL:-$BR/bfcl_tool_binding/${TAG}.plans.jsonl}"

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
echo "PROJ=$PROJ"
echo "WF=$WF"
echo "APIBANK=$APIBANK"
echo "BR=$BR"
echo "DATA_DIR=$DATA_DIR"
echo "CATEGORIES=$CATEGORIES"
echo "OFFSET=$OFFSET"
echo "LIMIT=$LIMIT"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "PLANNER_MODE=$PLANNER_MODE"
echo "ENSEMBLE_VARIANTS=$ENSEMBLE_VARIANTS"
echo "SEMANTIC_SLOT_FRAME=$SEMANTIC_SLOT_FRAME"
echo "SHARD_MANIFEST=$SHARD_MANIFEST"
echo "OUTPUT=$OUTPUT"
echo "REVIEW_CSV=$REVIEW_CSV"
echo "PLANS_JSONL=$PLANS_JSONL"

ARGS=(
  --model "$MODEL_PATH"
  --data-dir "$DATA_DIR"
  --categories $CATEGORIES
  --offset "$OFFSET"
  --limit "$LIMIT"
  --output "$OUTPUT"
  --review-csv "$REVIEW_CSV"
  --plans-jsonl "$PLANS_JSONL"
  --planner-mode "$PLANNER_MODE"
  --ensemble-variants "$ENSEMBLE_VARIANTS"
  --max-new-tokens "$MAX_NEW_TOKENS"
)

if [[ "$SEMANTIC_SLOT_FRAME" == "1" || "$SEMANTIC_SLOT_FRAME" == "true" ]]; then
  ARGS+=(--semantic-slot-frame)
fi

PYTHONPATH=src:compute2:.:$PYTHONPATH python scripts/eval_bfcl_gptoss_capability_ensemble.py "${ARGS[@]}"

echo "METRICS_PATH=$OUTPUT"
echo "REVIEW_PATH=$REVIEW_CSV"
echo "PLANS_PATH=$PLANS_JSONL"
