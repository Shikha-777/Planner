#!/usr/bin/env bash
#SBATCH --job-name=gptoss_chat_once
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.gptoss_chat_once.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.gptoss_chat_once.err

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

MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-20b}"
PROMPT="${PROMPT:-what is 1+2+log930 + 4}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"
OUT="${OUT:-results/gptoss_chat_once_${SLURM_JOB_ID}.json}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

hostname
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

echo "MODEL_PATH=$MODEL_PATH"
echo "PROMPT=$PROMPT"
echo "OUT=$OUT"

PYTHONPATH=src python scripts/run_gptoss_chat_once.py \
  --model "$MODEL_PATH" \
  --prompt "$PROMPT" \
  --system "$SYSTEM_PROMPT" \
  --out "$OUT" \
  --max-new-tokens "$MAX_NEW_TOKENS"

echo "RESULT_PATH=$OUT"
