#!/usr/bin/env bash
# Run the local GPT-OSS planner/executor + ToolACE validator ensemble on tau-bench.
# Defaults are a small smoke. Override START/END/SPLIT/TRIALS for larger runs.

#SBATCH --job-name=tau_ensemble
#SBATCH --partition=general-gpu
#SBATCH --account=compute2-yvorobeychik
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/compute2/%j.tau_ensemble.out
#SBATCH --error=logs/compute2/%j.tau_ensemble.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
export WF="${WF:-$STOR/workflow}"
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
export PYTHONPATH="$WF/scripts:$WF/external/tau-bench:${PYTHONPATH:-}"

source "$PROJ/.venv-c2/bin/activate"

python -m pip install -q \
  "litellm>=1.41.0" \
  "openai>=1.13.3" \
  "pydantic>=2" \
  "termcolor>=2.4.0" \
  "tenacity>=8.3.0" \
  "anthropic>=0.26.1" \
  "mistralai>=0.4.0" \
  "google-generativeai>=0.5.4"
python -m pip install -q -e "$WF/external/tau-bench"

JOB_ID="${SLURM_JOB_ID:-manual}"
USER_MODEL="${USER_MODEL:-qwen2.5:14b}"
USER_MODEL_PROVIDER="${USER_MODEL_PROVIDER:-ollama_chat}"
USER_STRATEGY="${USER_STRATEGY:-llm}"

if [ -n "${GOOGLE_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
  export GEMINI_API_KEY="$GOOGLE_API_KEY"
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
echo "User simulator provider=$USER_MODEL_PROVIDER model=$USER_MODEL strategy=$USER_STRATEGY"

if [ "$USER_MODEL_PROVIDER" = "ollama_chat" ] || [ "$USER_MODEL_PROVIDER" = "ollama" ]; then
  PORT="${OLLAMA_PORT:-$((22000 + (JOB_ID % 10000)))}"
  export OLLAMA_HOST="127.0.0.1:$PORT"
  export OLLAMA_API_BASE="http://127.0.0.1:$PORT"
  export OLLAMA_MODELS="${OLLAMA_MODELS:-$STOR/ollama}"
  export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30s}"
  export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
  export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"

  OLLAMA_BIN="${OLLAMA_BIN:-$STOR/bin/ollama}"
  if [ ! -x "$OLLAMA_BIN" ]; then
    OLLAMA_BIN="$(command -v ollama)"
  fi

  echo "Starting Ollama on port=$PORT"
  CUDA_VISIBLE_DEVICES="$USER_CUDA_VISIBLE_DEVICES" "$OLLAMA_BIN" serve >"$PROJ/logs/compute2/${JOB_ID}.tau_ensemble_ollama.log" 2>&1 &
  OLLAMA_PID=$!
  trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

  for i in $(seq 1 90); do
    if "$OLLAMA_BIN" list >/dev/null 2>&1; then
      echo "Ollama ready after $((i * 2))s"
      break
    fi
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
      echo "Ollama exited before readiness" >&2
      tail -80 "$PROJ/logs/compute2/${JOB_ID}.tau_ensemble_ollama.log" >&2 || true
      exit 1
    fi
    sleep 2
  done

  if [ "${SKIP_OLLAMA_PULL:-0}" != "1" ]; then
    "$OLLAMA_BIN" pull "$USER_MODEL"
  fi
else
  trap - EXIT
  if [ "$USER_MODEL_PROVIDER" = "gemini" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "GEMINI_API_KEY is required for USER_MODEL_PROVIDER=gemini" >&2
    exit 2
  fi
fi

export TAU_ENSEMBLE_EXECUTOR_ADAPTER="${TAU_ENSEMBLE_EXECUTOR_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-nemotron-agentic-bfcl-lora-2xa40-noGC1024}"
export TAU_ENSEMBLE_PLANNER_ADAPTER="${TAU_ENSEMBLE_PLANNER_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-taskdecomp-lora}"
export TAU_ENSEMBLE_RECOVERY_ADAPTER="${TAU_ENSEMBLE_RECOVERY_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-taskbench-taskdecomp-lora}"
export TAU_ENSEMBLE_VALIDATOR_MODEL="${TAU_ENSEMBLE_VALIDATOR_MODEL:-Team-ACE/ToolACE-8B}"
export TAU_ENSEMBLE_ENABLE_TOOLACE="${TAU_ENSEMBLE_ENABLE_TOOLACE:-1}"
export TAU_ENSEMBLE_RECOVERY_MODE="${TAU_ENSEMBLE_RECOVERY_MODE:-risky}"
export TAU_ENSEMBLE_VALIDATOR_MODE="${TAU_ENSEMBLE_VALIDATOR_MODE:-always}"
export TAU_ENSEMBLE_TRUST_VALIDATOR="${TAU_ENSEMBLE_TRUST_VALIDATOR:-0}"
export TAU_ENSEMBLE_DEBUG_DIR="${TAU_ENSEMBLE_DEBUG_DIR:-$BR/tau_ensemble_debug_${JOB_ID}}"

SPLIT="${SPLIT:-test}"
START="${START:-0}"
END="${END:-2}"
TRIALS="${TRIALS:-1}"
TAU_ENV="${TAU_ENV:-retail}"
LOG_DIR="${LOG_DIR:-$BR/tau_ensemble_${TAU_ENV}_${SPLIT}_${START}_${END}_${JOB_ID}}"

echo "Running tau ensemble: env=$TAU_ENV split=$SPLIT start=$START end=$END trials=$TRIALS"
echo "Logs/results: $LOG_DIR"
cd "$WF/external/tau-bench"
CUDA_VISIBLE_DEVICES="$AGENT_CUDA_VISIBLE_DEVICES" python "$WF/scripts/tau_run.py" \
  --agent-strategy ensemble \
  --env "$TAU_ENV" \
  --model local-gptoss-ensemble \
  --model-provider openai \
  --user-model "$USER_MODEL" \
  --user-model-provider "$USER_MODEL_PROVIDER" \
  --user-strategy "$USER_STRATEGY" \
  --task-split "$SPLIT" \
  --start-index "$START" \
  --end-index "$END" \
  --num-trials "$TRIALS" \
  --max-concurrency 1 \
  --temperature "${TEMPERATURE:-0}" \
  --log-dir "$LOG_DIR"
