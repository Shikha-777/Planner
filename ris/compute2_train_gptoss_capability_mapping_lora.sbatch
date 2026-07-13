#!/usr/bin/env bash
#SBATCH --job-name=gptoss_capmap_lora
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.gptoss_capmap_lora.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.gptoss_capmap_lora.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
export WF="${WF:-$STOR/workflow}"
export APIBANK="${APIBANK:-$STOR/external/DAMO-ConvAI/api-bank}"

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
export PYTHONPATH="$WF/scripts:$APIBANK:${PYTHONPATH:-}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

mkdir -p logs/compute2 checkpoints results "$HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

if [ ! -x "$PROJ/.venv-c2/bin/python" ]; then
  VENV_DIR=.venv-c2 PYTHON_BIN="${PYTHON_BIN:-python3}" bash scripts/bootstrap_env.sh
fi
source "$PROJ/.venv-c2/bin/activate"

python --version
hostname
nvidia-smi
python - <<'PY'
import json
from pathlib import Path
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("device0", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), round(props.total_memory / 1024**3, 2), "GB")

summary = Path("data/gpt55_capability_mapping_sft_v3/summary.json")
if summary.exists():
    data = json.loads(summary.read_text())
    print("capmap_summary", json.dumps({k: data.get(k) for k in ("total_records", "train_records", "validation_records", "family_count", "operation_count")}, sort_keys=True))
PY

DATA_DIR="${DATA_DIR:-data/gpt55_capability_mapping_sft_v3}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.sft.jsonl}"
VALIDATION_FILE="${VALIDATION_FILE:-$DATA_DIR/validation.sft.jsonl}"
OUT_DIR="${OUT_DIR:-checkpoints/gpt-oss-20b-capability-mapping-lora-v3}"
BASE_MODEL="${BASE_MODEL:-openai/gpt-oss-20b}"
INIT_ADAPTER="${INIT_ADAPTER:-checkpoints/gpt-oss-20b-taskdecomp-lora}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
EPOCHS="${EPOCHS:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_VALIDATION_SAMPLES="${MAX_VALIDATION_SAMPLES:-1000}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

if [ ! -s "$TRAIN_FILE" ]; then
  echo "Missing train file: $TRAIN_FILE" >&2
  exit 2
fi
if [ ! -s "$VALIDATION_FILE" ]; then
  echo "Missing validation file: $VALIDATION_FILE" >&2
  exit 2
fi

INIT_ARGS=()
if [ -n "$INIT_ADAPTER" ] && [ -d "$INIT_ADAPTER" ]; then
  INIT_ARGS=(--init-adapter "$INIT_ADAPTER")
  echo "Continuing from adapter: $INIT_ADAPTER"
else
  echo "No init adapter found; training a fresh LoRA"
fi

EXTRA_SAMPLE_ARGS=()
if [ -n "$MAX_TRAIN_SAMPLES" ]; then
  EXTRA_SAMPLE_ARGS+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [ -n "$MAX_VALIDATION_SAMPLES" ]; then
  EXTRA_SAMPLE_ARGS+=(--max-validation-samples "$MAX_VALIDATION_SAMPLES")
fi

echo "Training GPT-OSS capability planner LoRA"
echo "train=$TRAIN_FILE"
echo "validation=$VALIDATION_FILE"
echo "out=$OUT_DIR"
echo "max_seq_length=$MAX_SEQ_LENGTH epochs=$EPOCHS lr=$LEARNING_RATE grad_accum=$GRAD_ACCUM lora_r=$LORA_R"

python scripts/train_sft.py \
  --model "$BASE_MODEL" \
  "${INIT_ARGS[@]}" \
  --train "$TRAIN_FILE" \
  --validation "$VALIDATION_FILE" \
  --output-dir "$OUT_DIR" \
  --tuning-method lora \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --truncation-side right \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay 0.0 \
  --warmup-ratio 0.03 \
  --lr-scheduler-type cosine \
  --num-train-epochs "$EPOCHS" \
  --eval-steps "$EVAL_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit 3 \
  --logging-steps 10 \
  --dequantize-mxfp4 \
  --no-load-in-4bit \
  --gradient-checkpointing \
  --output-router-logits \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  "${EXTRA_SAMPLE_ARGS[@]}"

echo "DONE: $OUT_DIR"
