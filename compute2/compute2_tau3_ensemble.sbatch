#!/usr/bin/env bash
# Run the local GPT-OSS ensemble agent on tau3-bench/tau2-bench.

#SBATCH --job-name=tau3_ensemble
#SBATCH --partition=general-gpu
#SBATCH --account=compute2-yvorobeychik
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/compute2/%j.tau3_ensemble.out
#SBATCH --error=logs/compute2/%j.tau3_ensemble.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
export WF="${WF:-$STOR/workflow}"
export TAU3_ROOT="${TAU3_ROOT:-$WF/external/tau2-bench}"
export BR="${BR:-$WF/benchmark_results}"

cd "$PROJ"
mkdir -p logs/compute2 "$BR" "$PROJ/tmp" "$PROJ/.home-c2" "$PROJ/.hf_cache" "$PROJ/.cache-c2"

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
export PATH="$STOR/bin:$PATH"
export PYTHONPATH="$WF/scripts:$TAU3_ROOT/src:${PYTHONPATH:-}"

JOB_ID="${SLURM_JOB_ID:-manual}"
DOMAIN="${DOMAIN:-retail}"
TASK_SPLIT="${TASK_SPLIT:-base}"
NUM_TASKS="${NUM_TASKS:-1}"
TRIALS="${TRIALS:-1}"
MAX_STEPS="${MAX_STEPS:-80}"
MAX_ERRORS="${MAX_ERRORS:-10}"
RUN_NAME="${RUN_NAME:-tau3_ensemble_${DOMAIN}_${TASK_SPLIT}_${NUM_TASKS}_${JOB_ID}}"

USER_MODEL="${USER_MODEL:-ollama_chat/qwen2.5:14b}"
USER_ARGS="${USER_ARGS:-{\"temperature\":0}}"
AGENT_ARGS="${AGENT_ARGS:-{\"temperature\":0}}"

echo "HOST=$(hostname)"
echo "DOMAIN=$DOMAIN TASK_SPLIT=$TASK_SPLIT NUM_TASKS=$NUM_TASKS TRIALS=$TRIALS"
echo "TAU3_ROOT=$TAU3_ROOT RUN_NAME=$RUN_NAME"
nvidia-smi || true

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found at $STOR/bin/uv" >&2
  exit 2
fi

ALLOCATED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=, read -r -a CUDA_IDS <<< "$ALLOCATED_CUDA_VISIBLE_DEVICES"
USER_DEFAULT_CUDA="${CUDA_IDS[0]}"
if [ "${#CUDA_IDS[@]}" -gt 1 ]; then
  AGENT_DEFAULT_CUDA="$(IFS=,; echo "${CUDA_IDS[*]:1}")"
else
  AGENT_DEFAULT_CUDA="$USER_DEFAULT_CUDA"
fi
USER_CUDA_VISIBLE_DEVICES="${USER_CUDA_VISIBLE_DEVICES:-$USER_DEFAULT_CUDA}"
AGENT_CUDA_VISIBLE_DEVICES="${AGENT_CUDA_VISIBLE_DEVICES:-$AGENT_DEFAULT_CUDA}"
echo "Allocated CUDA_VISIBLE_DEVICES=$ALLOCATED_CUDA_VISIBLE_DEVICES"
echo "User simulator CUDA_VISIBLE_DEVICES=$USER_CUDA_VISIBLE_DEVICES"
echo "Agent CUDA_VISIBLE_DEVICES=$AGENT_CUDA_VISIBLE_DEVICES"

if [[ "$USER_MODEL" == ollama* ]]; then
  PORT="${OLLAMA_PORT:-$((23000 + (JOB_ID % 10000)))}"
  export OLLAMA_HOST="127.0.0.1:$PORT"
  export OLLAMA_API_BASE="http://127.0.0.1:$PORT"
  export OLLAMA_MODELS="${OLLAMA_MODELS:-$STOR/ollama}"
  export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30s}"
  export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
  export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
  OLLAMA_BIN="${OLLAMA_BIN:-$STOR/bin/ollama}"

  echo "Starting Ollama user simulator on $OLLAMA_HOST model=$USER_MODEL"
  CUDA_VISIBLE_DEVICES="$USER_CUDA_VISIBLE_DEVICES" "$OLLAMA_BIN" serve >"$PROJ/logs/compute2/${JOB_ID}.tau3_ollama.log" 2>&1 &
  OLLAMA_PID=$!
  trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

  for i in $(seq 1 90); do
    if "$OLLAMA_BIN" list >/dev/null 2>&1; then
      echo "Ollama ready after $((i * 2))s"
      break
    fi
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
      echo "Ollama exited before readiness" >&2
      tail -80 "$PROJ/logs/compute2/${JOB_ID}.tau3_ollama.log" >&2 || true
      exit 1
    fi
    sleep 2
  done

  if [ "${SKIP_OLLAMA_PULL:-1}" != "1" ]; then
    "$OLLAMA_BIN" pull "${USER_MODEL#ollama_chat/}"
  fi
fi

export TAU_ENSEMBLE_EXECUTOR_ADAPTER="${TAU_ENSEMBLE_EXECUTOR_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-nemotron-agentic-bfcl-lora-2xa40-noGC1024}"
export TAU_ENSEMBLE_PLANNER_ADAPTER="${TAU_ENSEMBLE_PLANNER_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-taskdecomp-lora}"
export TAU_ENSEMBLE_RECOVERY_ADAPTER="${TAU_ENSEMBLE_RECOVERY_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-taskbench-taskdecomp-lora}"
export TAU3_ENSEMBLE_DEBUG_DIR="${TAU3_ENSEMBLE_DEBUG_DIR:-$BR/tau3_ensemble_debug_${JOB_ID}}"

cd "$TAU3_ROOT"
echo "Preparing tau3 uv environment"
uv sync
uv pip install -q \
  "torch==2.6.0" \
  "transformers>=4.55.0,<5" \
  "huggingface-hub>=0.34.0,<1" \
  "peft>=0.17.0,<0.20" \
  "accelerate>=1.0.0,<2" \
  "sentencepiece"

echo "Registered agents/domains:"
uv run --no-sync python - <<'PY'
from tau2.run import get_options
info = get_options()
print("agents", info.agents)
print("domains", info.domains)
PY

echo "Running tau3 ensemble"
CUDA_VISIBLE_DEVICES="$AGENT_CUDA_VISIBLE_DEVICES" uv run --no-sync tau2 run \
  --domain "$DOMAIN" \
  --agent ensemble_agent \
  --agent-llm local-gptoss-ensemble \
  --agent-llm-args "$AGENT_ARGS" \
  --user user_simulator \
  --user-llm "$USER_MODEL" \
  --user-llm-args "$USER_ARGS" \
  --task-split-name "$TASK_SPLIT" \
  --num-tasks "$NUM_TASKS" \
  --num-trials "$TRIALS" \
  --max-steps "$MAX_STEPS" \
  --max-errors "$MAX_ERRORS" \
  --max-concurrency 1 \
  --save-to "$RUN_NAME" \
  --log-level INFO

RESULT_DIR="$TAU3_ROOT/data/simulations/$RUN_NAME"
DEST_DIR="$BR/$RUN_NAME"
mkdir -p "$DEST_DIR"
rsync -a "$RESULT_DIR/" "$DEST_DIR/"
echo "RESULT_DIR=$RESULT_DIR"
echo "COPIED_RESULT_DIR=$DEST_DIR"
echo "DONE"
