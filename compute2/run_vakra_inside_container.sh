#!/usr/bin/env bash
# Run VAKRA with the current ensemble from inside the VAKRA image.

set -euo pipefail

CAPABILITY_ID="${CAPABILITY_ID:-1}"
DOMAIN="${DOMAIN:-beer_factory}"
AGENT="${AGENT:-ensemble}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
MAX_STEPS="${MAX_STEPS:-8}"
JOB_ID="${SLURM_JOB_ID:-manual}"

PROJECT_MOUNT="${PROJECT_MOUNT:-/workspace/project}"
VAKRA_MOUNT="${VAKRA_MOUNT:-/workspace/vakra}"
RESULTS_MOUNT="${RESULTS_MOUNT:-/workspace/results}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_MOUNT/vakra_${DOMAIN}_cap${CAPABILITY_ID}_${JOB_ID}}"

export HOME="${HOME:-$PROJECT_MOUNT/.home-c2}"
export HF_HOME="${HF_HOME:-$PROJECT_MOUNT/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PROJECT_MOUNT/.hf_cache/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_MOUNT/.hf_cache/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PROJECT_MOUNT/.hf_cache/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_MOUNT/.cache-c2}"
export TMPDIR="${TMPDIR:-$PROJECT_MOUNT/tmp}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PROJECT_MOUNT/compute2:/workspace/scripts:/app:${PYTHONPATH:-}"

export TAU_ENSEMBLE_EXECUTOR_ADAPTER="${TAU_ENSEMBLE_EXECUTOR_ADAPTER:-$PROJECT_MOUNT/checkpoints/gpt-oss-20b-nemotron-agentic-bfcl-lora-2xa40-noGC1024}"
export TAU_ENSEMBLE_PLANNER_ADAPTER="${TAU_ENSEMBLE_PLANNER_ADAPTER:-$PROJECT_MOUNT/checkpoints/gpt-oss-20b-taskdecomp-lora}"
export TAU_ENSEMBLE_RECOVERY_ADAPTER="${TAU_ENSEMBLE_RECOVERY_ADAPTER:-$PROJECT_MOUNT/checkpoints/gpt-oss-20b-taskbench-taskdecomp-lora}"
export TAU_ENSEMBLE_DEVICE_MAP="${TAU_ENSEMBLE_DEVICE_MAP:-single}"
export VAKRA_MCP_LAUNCHER=local
export VAKRA_MCP_COMMAND="${VAKRA_MCP_COMMAND:-python}"
export VAKRA_MCP_ARGS="${VAKRA_MCP_ARGS:-/app/mcp_dispatch.py}"

mkdir -p "$HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$OUTPUT_DIR"

echo "HOST=$(hostname)"
echo "CAPABILITY_ID=$CAPABILITY_ID DOMAIN=$DOMAIN AGENT=$AGENT MAX_SAMPLES=$MAX_SAMPLES"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "PROJECT_MOUNT=$PROJECT_MOUNT"
echo "VAKRA_MOUNT=$VAKRA_MOUNT"
echo "TAU_ENSEMBLE_EXECUTOR_ADAPTER=$TAU_ENSEMBLE_EXECUTOR_ADAPTER"
python --version
nvidia-smi || true

SERVICE_PID=""
if [ "$CAPABILITY_ID" != "1" ]; then
  echo "Starting VAKRA background services for capability $CAPABILITY_ID"
  /app/entrypoint.sh >"$OUTPUT_DIR/vakra_services.log" 2>&1 &
  SERVICE_PID=$!
  trap 'if [ -n "$SERVICE_PID" ]; then kill "$SERVICE_PID" 2>/dev/null || true; fi' EXIT
  for i in $(seq 1 180); do
    if grep -q "All services running" "$OUTPUT_DIR/vakra_services.log" 2>/dev/null; then
      echo "VAKRA services ready"
      break
    fi
    if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
      echo "VAKRA services exited early" >&2
      tail -120 "$OUTPUT_DIR/vakra_services.log" >&2 || true
      exit 1
    fi
    sleep 2
  done
fi

ARGS=(
  --vakra-root "$VAKRA_MOUNT"
  --capability-id "$CAPABILITY_ID"
  --domain "$DOMAIN"
  --agent "$AGENT"
  --max-steps "$MAX_STEPS"
  --output-dir "$OUTPUT_DIR"
  --mcp-launcher local
  --mcp-command "$VAKRA_MCP_COMMAND"
  --mcp-args $VAKRA_MCP_ARGS
  --mcp-db-root /app/db
)

if [ "$MAX_SAMPLES" != "0" ]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi

python "$PROJECT_MOUNT/compute2/vakra_mcp_ensemble_runner.py" "${ARGS[@]}"
python "$VAKRA_MOUNT/validate_output.py" "$OUTPUT_DIR"

echo "DONE"
