#!/usr/bin/env bash
#SBATCH --job-name=gptoss_agent_full
#SBATCH --partition=general-gpu
#SBATCH --account=compute2-yvorobeychik
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=768G
#SBATCH --gres=gpu:H100:4
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/compute2/%j.gptoss_agentinstruct_full.out
#SBATCH --error=logs/compute2/%j.gptoss_agentinstruct_full.err

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
export PIP_CACHE_DIR="$PROJ/.cache-c2/pip"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$WF/scripts:$APIBANK:${PYTHONPATH:-}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

mkdir -p \
  logs/compute2 checkpoints results "$HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

if [ ! -x "$PROJ/.venv-c2/bin/python" ]; then
  VENV_DIR=.venv-c2 PYTHON_BIN="${PYTHON_BIN:-python3}" bash scripts/bootstrap_env.sh
fi
source "$PROJ/.venv-c2/bin/activate"
python --version
python - <<'PY' || python -m pip install "importlib-metadata>=6,<9"
import importlib_metadata  # noqa: F401
PY

hostname
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print("device", index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index), round(props.total_memory / 1024**3, 2), "GB")
PY

DATA_DIR="${DATA_DIR:-data/agentinstruct_sft}"
OUT_DIR="${OUT_DIR:-checkpoints/gpt-oss-20b-agentinstruct-full-fsdp}"
BASE_MODEL="${BASE_MODEL:-openai/gpt-oss-20b}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
EPOCHS="${EPOCHS:-3.0}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EVAL_STEPS="${EVAL_STEPS:-25}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

python scripts/prepare_agentinstruct_sft.py \
  --dataset zai-org/AgentInstruct \
  --out-dir "$DATA_DIR" \
  --validation-ratio 0.05 \
  --loss-policy true-only

torchrun --standalone --nproc_per_node="$GPUS_PER_NODE" scripts/train_sft.py \
  --model "$BASE_MODEL" \
  --train "$DATA_DIR/train.jsonl" \
  --validation "$DATA_DIR/validation.jsonl" \
  --output-dir "$OUT_DIR" \
  --tuning-method full \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --truncation-side right \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay 0.1 \
  --warmup-ratio 0.05 \
  --lr-scheduler-type cosine \
  --num-train-epochs "$EPOCHS" \
  --eval-steps "$EVAL_STEPS" \
  --save-steps "$EVAL_STEPS" \
  --save-total-limit 3 \
  --optim paged_adamw_8bit \
  --dequantize-mxfp4 \
  --no-load-in-4bit \
  --no-gradient-checkpointing \
  --output-router-logits \
  --fsdp "full_shard auto_wrap" \
  --fsdp-transformer-layer-cls-to-wrap GptOssDecoderLayer \
  --fsdp-activation-checkpointing
