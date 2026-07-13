#!/usr/bin/env bash
#SBATCH --job-name=gptoss_trace_infer
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.gptoss_trace_infer.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.gptoss_trace_infer.err

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

mkdir -p logs/compute2 results "$HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

source "$PROJ/.venv-c2/bin/activate"

MODEL_PATH="${MODEL_PATH:-checkpoints/gpt-oss-20b-trace-decomp-lora-v1}"
TASK="${TASK:-make this essay better}"
CONTEXT="${CONTEXT:-}"
OUT="${OUT:-results/gptoss_trace_decomp_infer_${SLURM_JOB_ID}.json}"

hostname
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

echo "MODEL_PATH=$MODEL_PATH"
echo "TASK=$TASK"
echo "OUT=$OUT"

PYTHONPATH=src python scripts/run_gptoss_decomp_once.py \
  --model "$MODEL_PATH" \
  --task "$TASK" \
  --context "$CONTEXT" \
  --out "$OUT" \
  --max-new-tokens "${MAX_NEW_TOKENS:-420}"

echo "RESULT_PATH=$OUT"
