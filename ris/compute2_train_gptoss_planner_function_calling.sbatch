#!/usr/bin/env bash
#SBATCH --job-name=gptoss_planner_fc
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.gptoss_planner_fc.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.gptoss_planner_fc.err

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
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

DATA_DIR="${DATA_DIR:-data/planner_function_calling_mix_v1}"
OUT_DIR="${OUT_DIR:-checkpoints/gpt-oss-20b-taskdecomp-function-calling-lora-v1}"
BASE_MODEL="${BASE_MODEL:-openai/gpt-oss-20b}"
INIT_ADAPTER="${INIT_ADAPTER:-checkpoints/gpt-oss-20b-taskdecomp-lora}"

python scripts/mix_sft_jsonl.py \
  --out-dir "$DATA_DIR" \
  --seed "${MIX_SEED:-23}" \
  --train-source "data/xlam_function_calling_60k/train.jsonl:${XLAM_TRAIN_LIMIT:-30000}" \
  --train-source "data/nemotron_agentic_bfcl/train.jsonl:${NEMOTRON_TRAIN_LIMIT:-12000}" \
  --train-source "data/processed/train.sft.jsonl:${TASKDECOMP_TRAIN_LIMIT:-8000}" \
  --train-source "data/processed/taskbench.train.sft.jsonl:${TASKBENCH_TRAIN_LIMIT:-8000}" \
  --validation-source "data/xlam_function_calling_60k/validation.jsonl:${XLAM_VAL_LIMIT:-400}" \
  --validation-source "data/nemotron_agentic_bfcl/validation.jsonl:${NEMOTRON_VAL_LIMIT:-200}" \
  --validation-source "data/processed/validation.sft.jsonl:${TASKDECOMP_VAL_LIMIT:-200}" \
  --validation-source "data/processed/taskbench.validation.sft.jsonl:${TASKBENCH_VAL_LIMIT:-200}"

python scripts/train_sft.py \
  --model "$BASE_MODEL" \
  --init-adapter "$INIT_ADAPTER" \
  --train "$DATA_DIR/train.jsonl" \
  --validation "$DATA_DIR/validation.jsonl" \
  --output-dir "$OUT_DIR" \
  --max-seq-length "${MAX_SEQ_LENGTH:-2048}" \
  --truncation-side right \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "${GRAD_ACCUM:-64}" \
  --learning-rate "${LEARNING_RATE:-2e-5}" \
  --weight-decay 0.0 \
  --warmup-ratio 0.03 \
  --lr-scheduler-type cosine \
  --num-train-epochs "${EPOCHS:-1.0}" \
  --eval-steps "${EVAL_STEPS:-250}" \
  --save-steps "${SAVE_STEPS:-250}" \
  --save-total-limit 2 \
  --logging-steps 10 \
  --dequantize-mxfp4 \
  --no-load-in-4bit \
  --gradient-checkpointing \
  --output-router-logits

echo "DONE"
