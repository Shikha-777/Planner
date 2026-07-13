#!/usr/bin/env bash
#SBATCH --job-name=gptoss_trace_decomp
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.gptoss_trace_decomp.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.gptoss_trace_decomp.err

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
export PIP_CACHE_DIR="$PROJ/.cache-c2/pip"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

mkdir -p logs/compute2 checkpoints results "$HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

source "$PROJ/.venv-c2/bin/activate"

hostname
nvidia-smi
python - <<'PY'
import json
from pathlib import Path
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print("device", index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index), round(props.total_memory / 1024**3, 2), "GB")

summary = Path("data/trace_decomp/summary.json")
if summary.exists():
    print("trace_decomp_summary", json.dumps(json.loads(summary.read_text()).get("counts", {}), sort_keys=True))
PY

TRAIN_FILE="${TRAIN_FILE:-data/trace_decomp/train.sft.jsonl}"
VALIDATION_FILE="${VALIDATION_FILE:-data/trace_decomp/validation.sft.jsonl}"
OUT_DIR="${OUT_DIR:-checkpoints/gpt-oss-20b-trace-decomp-lora-v1}"
BASE_MODEL="${BASE_MODEL:-openai/gpt-oss-20b}"
INIT_ADAPTER="${INIT_ADAPTER:-checkpoints/gpt-oss-20b-taskdecomp-lora}"

INIT_ARGS=()
if [ -n "$INIT_ADAPTER" ] && [ -d "$INIT_ADAPTER" ]; then
  INIT_ARGS=(--init-adapter "$INIT_ADAPTER")
  echo "Continuing from adapter: $INIT_ADAPTER"
else
  echo "No init adapter found; training a fresh LoRA"
fi

python scripts/train_sft.py \
  --model "$BASE_MODEL" \
  "${INIT_ARGS[@]}" \
  --train "$TRAIN_FILE" \
  --validation "$VALIDATION_FILE" \
  --output-dir "$OUT_DIR" \
  --max-seq-length "${MAX_SEQ_LENGTH:-3072}" \
  --truncation-side left \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "${GRAD_ACCUM:-8}" \
  --learning-rate "${LEARNING_RATE:-1e-5}" \
  --weight-decay 0.0 \
  --warmup-ratio 0.03 \
  --lr-scheduler-type cosine \
  --num-train-epochs "${EPOCHS:-3.0}" \
  --eval-steps "${EVAL_STEPS:-10}" \
  --save-steps "${SAVE_STEPS:-10}" \
  --save-total-limit 3 \
  --logging-steps 1 \
  --dequantize-mxfp4 \
  --no-load-in-4bit \
  --gradient-checkpointing \
  --output-router-logits

echo "DONE: $OUT_DIR"
