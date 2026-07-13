#!/usr/bin/env bash
# Run tau-bench with the direct goal-graph runtime agent.
# This intentionally bypasses tau_ensemble_agent.py.

#SBATCH --job-name=tau_goal_graph
#SBATCH --partition=general-gpu
#SBATCH --account=compute2-yvorobeychik
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/compute2/%j.tau_goal_graph.out
#SBATCH --error=logs/compute2/%j.tau_goal_graph.err

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
export PYTHONPATH="$PROJ/src:$PROJ/scripts:$PROJ/compute2:$WF/scripts:$WF/external/tau-bench:${PYTHONPATH:-}"
find "$PROJ/compute2" -path '*/__pycache__/tau_goal_graph_agent*.pyc' -delete 2>/dev/null || true

source "$PROJ/.venv-c2/bin/activate"

# The shared Compute2 virtualenv is provisioned ahead of time.  Installing
# into it inside every batch job races with concurrent evaluations and can
# temporarily uninstall tau_bench underneath a running process.
python - <<'PY'
import tau_bench  # noqa: F401
PY

JOB_ID="${SLURM_JOB_ID:-manual}"
USER_MODEL="${USER_MODEL:-qwen2.5:14b}"
USER_MODEL_PROVIDER="${USER_MODEL_PROVIDER:-ollama_chat}"
USER_STRATEGY="${USER_STRATEGY:-llm}"
export TAU_GOAL_GRAPH_MODEL="${TAU_GOAL_GRAPH_MODEL:-openai/gpt-oss-20b}"
# These are capacity declarations, rather than partition-specific assumptions.
# Set either value to 0 for an intentionally smaller model deployment.
TAU_GOAL_GRAPH_AGENT_MIN_VRAM_MIB="${TAU_GOAL_GRAPH_AGENT_MIN_VRAM_MIB:-32768}"
TAU_USER_SIMULATOR_MIN_VRAM_MIB="${TAU_USER_SIMULATOR_MIN_VRAM_MIB:-12288}"

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
echo "Goal-graph agent CUDA_VISIBLE_DEVICES=$AGENT_CUDA_VISIBLE_DEVICES"
echo "User simulator provider=$USER_MODEL_PROVIDER model=$USER_MODEL strategy=$USER_STRATEGY"

visible_vram_mib() {
  local devices="$1"
  CUDA_VISIBLE_DEVICES="$devices" python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(sum(torch.cuda.get_device_properties(index).total_memory // (1024 * 1024)
          for index in range(torch.cuda.device_count())))
PY
}

agent_vram_mib="$(visible_vram_mib "$AGENT_CUDA_VISIBLE_DEVICES")"
user_vram_mib="$(visible_vram_mib "$USER_CUDA_VISIBLE_DEVICES")"
if [ "$AGENT_CUDA_VISIBLE_DEVICES" = "$USER_CUDA_VISIBLE_DEVICES" ]; then
  required_vram_mib=$((TAU_GOAL_GRAPH_AGENT_MIN_VRAM_MIB + TAU_USER_SIMULATOR_MIN_VRAM_MIB))
  echo "Co-located agent/user VRAM: available=${agent_vram_mib}MiB required=${required_vram_mib}MiB"
  if [ "$agent_vram_mib" -lt "$required_vram_mib" ]; then
    echo "Allocated GPU capacity is insufficient for the configured agent and user simulator. Request a full GPU or lower the declared model requirements." >&2
    exit 3
  fi
else
  echo "Agent VRAM: available=${agent_vram_mib}MiB required=${TAU_GOAL_GRAPH_AGENT_MIN_VRAM_MIB}MiB"
  echo "User VRAM: available=${user_vram_mib}MiB required=${TAU_USER_SIMULATOR_MIN_VRAM_MIB}MiB"
  if [ "$agent_vram_mib" -lt "$TAU_GOAL_GRAPH_AGENT_MIN_VRAM_MIB" ] || [ "$user_vram_mib" -lt "$TAU_USER_SIMULATOR_MIN_VRAM_MIB" ]; then
    echo "Allocated GPU capacity is insufficient for the configured agent or user simulator. Request larger or separate GPUs, or lower the declared model requirements." >&2
    exit 3
  fi
fi

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
  CUDA_VISIBLE_DEVICES="$USER_CUDA_VISIBLE_DEVICES" "$OLLAMA_BIN" serve >"$PROJ/logs/compute2/${JOB_ID}.tau_goal_graph_ollama.log" 2>&1 &
  OLLAMA_PID=$!
  trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

  for i in $(seq 1 90); do
    if "$OLLAMA_BIN" list >/dev/null 2>&1; then
      echo "Ollama ready after $((i * 2))s"
      break
    fi
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
      echo "Ollama exited before readiness" >&2
      tail -80 "$PROJ/logs/compute2/${JOB_ID}.tau_goal_graph_ollama.log" >&2 || true
      exit 1
    fi
    sleep 2
  done

  if [ "${SKIP_OLLAMA_PULL:-1}" != "1" ]; then
    "$OLLAMA_BIN" pull "$USER_MODEL"
  else
    echo "Skipping Ollama pull for $USER_MODEL (set SKIP_OLLAMA_PULL=0 to pull)"
  fi
else
  trap - EXIT
  if [ "$USER_MODEL_PROVIDER" = "gemini" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "GEMINI_API_KEY is required for USER_MODEL_PROVIDER=gemini" >&2
    exit 2
  fi
fi

export TAU_GOAL_GRAPH_MAX_NEW_TOKENS="${TAU_GOAL_GRAPH_MAX_NEW_TOKENS:-900}"
export TAU_GOAL_GRAPH_REPAIR_ATTEMPTS="${TAU_GOAL_GRAPH_REPAIR_ATTEMPTS:-1}"
export TAU_GOAL_GRAPH_MAX_STEPS="${TAU_GOAL_GRAPH_MAX_STEPS:-30}"
export TAU_GOAL_GRAPH_PLANNER_MODE="${TAU_GOAL_GRAPH_PLANNER_MODE:-stepwise}"
export TAU_GOAL_GRAPH_ALLOW_SIDE_EFFECTS="${TAU_GOAL_GRAPH_ALLOW_SIDE_EFFECTS:-1}"
export TAU_GOAL_GRAPH_POLICY_REPAIR="${TAU_GOAL_GRAPH_POLICY_REPAIR:-0}"
# The default runtime is schema/evidence-driven.  Set this to 1 only when
# comparing against the retired tau-specific heuristic baseline.
export TAU_GOAL_GRAPH_LEGACY_HEURISTICS="${TAU_GOAL_GRAPH_LEGACY_HEURISTICS:-0}"
export TAU_GOAL_GRAPH_DEBUG_DIR="${TAU_GOAL_GRAPH_DEBUG_DIR:-$BR/tau_goal_graph_debug_${JOB_ID}}"

SPLIT="${SPLIT:-dev}"
START="${START:-0}"
END="${END:-2}"
TASK_IDS="${TASK_IDS:-}"
TRIALS="${TRIALS:-1}"
TAU_ENV="${TAU_ENV:-retail}"
if [ -n "$TASK_IDS" ]; then
  TASK_LABEL="$(echo "$TASK_IDS" | tr '[:space:],:' '____' | tr -s '_' | sed 's/^_//;s/_$//')"
  LOG_DIR="${LOG_DIR:-$BR/tau_goal_graph_${TAU_ENV}_${SPLIT}_ids_${TASK_LABEL}_${JOB_ID}}"
else
  LOG_DIR="${LOG_DIR:-$BR/tau_goal_graph_${TAU_ENV}_${SPLIT}_${START}_${END}_${JOB_ID}}"
fi

echo "Running tau goal-graph: env=$TAU_ENV split=$SPLIT start=$START end=$END task_ids=${TASK_IDS:-none} trials=$TRIALS"
echo "Goal-graph model=$TAU_GOAL_GRAPH_MODEL planner=$TAU_GOAL_GRAPH_PLANNER_MODE"
echo "Logs/results: $LOG_DIR"

RUN_ARGS=(
  --agent-strategy goal-graph
  --env "$TAU_ENV"
  --model "$TAU_GOAL_GRAPH_MODEL"
  --model-provider openai
  --user-model "$USER_MODEL"
  --user-model-provider "$USER_MODEL_PROVIDER"
  --user-strategy "$USER_STRATEGY"
  --task-split "$SPLIT"
  --num-trials "$TRIALS"
  --max-concurrency 1
  --temperature "${TEMPERATURE:-0}"
  --log-dir "$LOG_DIR"
)
if [ -n "$TASK_IDS" ]; then
  read -r -a TASK_ID_ARGS <<< "$TASK_IDS"
  RUN_ARGS+=(--task-ids "${TASK_ID_ARGS[@]}")
else
  RUN_ARGS+=(--start-index "$START" --end-index "$END")
fi

cd "$WF/external/tau-bench"
CUDA_VISIBLE_DEVICES="$AGENT_CUDA_VISIBLE_DEVICES" python "$PROJ/compute2/tau_run.py" "${RUN_ARGS[@]}"
