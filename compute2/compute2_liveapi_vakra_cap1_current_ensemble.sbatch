#!/usr/bin/env bash
# Run the current GPT-OSS ensemble on the official executable
# Live API-Bench/VAKRA capability-1 SLOT/SEL data subset.

#SBATCH --job-name=liveapi_cap1_current
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.liveapi_cap1_current.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.liveapi_cap1_current.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
export WF="${WF:-$STOR/workflow}"
export BR="${BR:-$WF/benchmark_results}"
export VAKRA_ROOT="${VAKRA_ROOT:-$WF/external/vakra}"
export JOB_ID="${SLURM_JOB_ID:-manual}"

export LIVEAPI_SPLIT="${LIVEAPI_SPLIT:-train}"
export DOMAIN="${DOMAIN:-beer_factory}"
export DOMAINS="${DOMAINS:-$DOMAIN}"
export AGENT="${AGENT:-ensemble}"
export MAX_SAMPLES="${MAX_SAMPLES:-3}"
export MAX_STEPS="${MAX_STEPS:-8}"
export OUTPUT_DIR="${OUTPUT_DIR:-$BR/liveapi_vakra_cap1_${LIVEAPI_SPLIT}_${DOMAIN}_${AGENT}_${JOB_ID}}"

export HOME="$PROJ/.home-c2"
export HF_HOME="$PROJ/.hf_cache"
export TRANSFORMERS_CACHE="$PROJ/.hf_cache/transformers"
export HF_DATASETS_CACHE="$PROJ/.hf_cache/datasets"
export HUGGINGFACE_HUB_CACHE="$PROJ/.hf_cache/hub"
export XDG_CACHE_HOME="$PROJ/.cache-c2"
export TMPDIR="$PROJ/tmp"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PROJ/compute2:$WF/scripts:/app:${PYTHONPATH:-}"

export TAU_ENSEMBLE_EXECUTOR_ADAPTER="${TAU_ENSEMBLE_EXECUTOR_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-nemotron-agentic-bfcl-lora-2xa40-noGC1024}"
export TAU_ENSEMBLE_PLANNER_ADAPTER="${TAU_ENSEMBLE_PLANNER_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-taskdecomp-lora}"
export TAU_ENSEMBLE_RECOVERY_ADAPTER="${TAU_ENSEMBLE_RECOVERY_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-taskbench-taskdecomp-lora}"
export TAU_ENSEMBLE_DEVICE_MAP="${TAU_ENSEMBLE_DEVICE_MAP:-single}"
export VAKRA_ENSEMBLE_MAX_NEW_TOKENS="${VAKRA_ENSEMBLE_MAX_NEW_TOKENS:-700}"
export VAKRA_ENSEMBLE_TEMPERATURE="${VAKRA_ENSEMBLE_TEMPERATURE:-0.0}"

mkdir -p "$PROJ/logs/compute2" "$BR" "$HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$OUTPUT_DIR"

echo "HOST=$(hostname)"
echo "LIVEAPI_SPLIT=$LIVEAPI_SPLIT DOMAINS=$DOMAINS MAX_SAMPLES=$MAX_SAMPLES MAX_STEPS=$MAX_STEPS"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "VAKRA_ROOT=$VAKRA_ROOT"
nvidia-smi || true

if [ ! -d "$VAKRA_ROOT" ]; then
  echo "Missing VAKRA_ROOT=$VAKRA_ROOT" >&2
  exit 2
fi

if ! command -v singularity >/dev/null 2>&1; then
  echo "singularity is required for the Python 3.11 MCP subprocess." >&2
  exit 2
fi

export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$PROJ/.singularity-cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/tmp/${USER:-dshrestha}/singularity-${JOB_ID}}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

VAKRA_RUNTIME_IMAGE="${VAKRA_RUNTIME_IMAGE:-docker://pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime}"
VAKRA_VENV="${VAKRA_VENV:-$PROJ/.venv-vakra-py311}"
BIND_ARG="$VAKRA_ROOT:/app,$PROJ:$PROJ,$WF:$WF"

if [ ! -x "$VAKRA_VENV/bin/python" ]; then
  echo "Creating VAKRA Python 3.11 runtime venv at $VAKRA_VENV"
  singularity exec --nv --bind "$BIND_ARG" "$VAKRA_RUNTIME_IMAGE" python -m venv --system-site-packages "$VAKRA_VENV"
  singularity exec --nv --bind "$BIND_ARG" "$VAKRA_RUNTIME_IMAGE" "$VAKRA_VENV/bin/python" -m pip install --upgrade pip
fi

singularity exec --nv --bind "$BIND_ARG" "$VAKRA_RUNTIME_IMAGE" "$VAKRA_VENV/bin/python" -m pip install \
  "huggingface_hub>=0.24.0" \
  "mcp>=1.0.0" \
  "accelerate>=1.0.0" \
  "peft>=0.17.0" \
  "transformers>=4.55.0" \
  "pandas>=2.0.0" \
  "pyyaml>=6.0.0" \
  -r /app/environment/m3/python_tools/requirements.txt

read -r -a DOMAIN_LIST <<< "$DOMAINS"

singularity exec --nv \
  --env "HF_HOME=$HF_HOME" \
  --env "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE" \
  --bind "$BIND_ARG" \
  "$VAKRA_RUNTIME_IMAGE" \
  "$VAKRA_VENV/bin/python" "$PROJ/compute2/prepare_vakra_liveapi_subset.py" \
    --vakra-root "$VAKRA_ROOT" \
    --split "$LIVEAPI_SPLIT" \
    --domains "${DOMAIN_LIST[@]}"

for RUN_DOMAIN in "${DOMAIN_LIST[@]}"; do
  RUN_OUT="$OUTPUT_DIR/$RUN_DOMAIN"
  mkdir -p "$RUN_OUT"
  RUNNER_ARGS=(
    --vakra-root "$VAKRA_ROOT"
    --data-root "$VAKRA_ROOT/data/$LIVEAPI_SPLIT"
    --capability-id 1
    --domain "$RUN_DOMAIN"
    --agent "$AGENT"
    --max-steps "$MAX_STEPS"
    --output-dir "$RUN_OUT"
    --mcp-launcher local
    --mcp-command "$VAKRA_VENV/bin/python"
    --mcp-args /app/docker/mcp_dispatch.py
    --mcp-db-root /app/data/databases
  )
  if [ "$MAX_SAMPLES" != "0" ]; then
    RUNNER_ARGS+=(--max-samples "$MAX_SAMPLES")
  fi

  singularity exec --nv \
    --env "PYTHONPATH=$PYTHONPATH" \
    --env "HOME=$HOME" \
    --env "HF_HOME=$HF_HOME" \
    --env "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE" \
    --env "HF_DATASETS_CACHE=$HF_DATASETS_CACHE" \
    --env "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE" \
    --env "XDG_CACHE_HOME=$XDG_CACHE_HOME" \
    --env "TMPDIR=$TMPDIR" \
    --env "TAU_ENSEMBLE_EXECUTOR_ADAPTER=$TAU_ENSEMBLE_EXECUTOR_ADAPTER" \
    --env "TAU_ENSEMBLE_PLANNER_ADAPTER=$TAU_ENSEMBLE_PLANNER_ADAPTER" \
    --env "TAU_ENSEMBLE_RECOVERY_ADAPTER=$TAU_ENSEMBLE_RECOVERY_ADAPTER" \
    --env "TAU_ENSEMBLE_DEVICE_MAP=$TAU_ENSEMBLE_DEVICE_MAP" \
    --env "VAKRA_ENSEMBLE_MAX_NEW_TOKENS=$VAKRA_ENSEMBLE_MAX_NEW_TOKENS" \
    --env "VAKRA_ENSEMBLE_TEMPERATURE=$VAKRA_ENSEMBLE_TEMPERATURE" \
    --bind "$BIND_ARG" \
    "$VAKRA_RUNTIME_IMAGE" \
    "$VAKRA_VENV/bin/python" "$PROJ/compute2/vakra_mcp_ensemble_runner.py" "${RUNNER_ARGS[@]}"

  singularity exec --nv --bind "$BIND_ARG" "$VAKRA_RUNTIME_IMAGE" "$VAKRA_VENV/bin/python" \
    "$VAKRA_ROOT/validate_output.py" "$RUN_OUT"

  singularity exec --nv --bind "$BIND_ARG" "$VAKRA_RUNTIME_IMAGE" "$VAKRA_VENV/bin/python" \
    "$PROJ/compute2/score_vakra_cap1_exact.py" \
      --gold "$VAKRA_ROOT/data/$LIVEAPI_SPLIT/capability_1_bi_apis/output/$RUN_DOMAIN.json" \
      --pred "$RUN_OUT/$RUN_DOMAIN.json" \
      --out "$RUN_OUT/${RUN_DOMAIN}.exact_score.json"
done

echo "DONE"
