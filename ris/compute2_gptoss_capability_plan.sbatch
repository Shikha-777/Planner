#!/usr/bin/env bash
#SBATCH --job-name=gptoss_cap_plan
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.gptoss_cap_plan.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.gptoss_cap_plan.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
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

mkdir -p logs/compute2 results "$HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

source "$PROJ/.venv-c2/bin/activate"

MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-20b}"
GOLD="${GOLD:-data/capability_planning/gold_40.jsonl}"
INPUT="${INPUT:-$GOLD}"
OUTPUT="${OUTPUT:-results/gptoss_capability_plan_gold40_${SLURM_JOB_ID}.jsonl}"
METRICS_OUTPUT="${METRICS_OUTPUT:-results/gptoss_capability_plan_gold40_${SLURM_JOB_ID}.metrics.json}"
CSV_OUTPUT="${CSV_OUTPUT:-results/gptoss_capability_plan_gold40_${SLURM_JOB_ID}.review.csv}"
PLANNER_MODE="${PLANNER_MODE:-chunked}"
if [ -z "${MAX_NEW_TOKENS:-}" ]; then
  if [ "$PLANNER_MODE" = "one_shot" ]; then
    MAX_NEW_TOKENS=4200
  else
    MAX_NEW_TOKENS=1200
  fi
fi
LIMIT="${LIMIT:-}"

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
echo "GOLD=$GOLD"
echo "INPUT=$INPUT"
echo "OUTPUT=$OUTPUT"
echo "METRICS_OUTPUT=$METRICS_OUTPUT"
echo "CSV_OUTPUT=$CSV_OUTPUT"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "PLANNER_MODE=$PLANNER_MODE"

ARGS=(
  --model "$MODEL_PATH"
  --input "$INPUT"
  --output "$OUTPUT"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --planner-mode "$PLANNER_MODE"
)

if [ -n "$LIMIT" ]; then
  ARGS+=(--limit "$LIMIT")
fi

PYTHONPATH=src python scripts/run_gptoss_capability_plan.py "${ARGS[@]}"

EVAL_ARGS=(
  --gold "$GOLD"
  --predictions "$OUTPUT"
  --metrics-out "$METRICS_OUTPUT"
  --csv-out "$CSV_OUTPUT"
)

if [ -n "$LIMIT" ]; then
  EVAL_ARGS+=(--predicted-only)
fi

PYTHONPATH=src python scripts/eval_capability_plans.py "${EVAL_ARGS[@]}"

echo "RESULT_PATH=$OUTPUT"
echo "METRICS_PATH=$METRICS_OUTPUT"
echo "CSV_PATH=$CSV_OUTPUT"
