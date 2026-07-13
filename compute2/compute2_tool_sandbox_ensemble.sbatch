#!/usr/bin/env bash
# Run Apple ToolSandbox smoke scenarios with the local GPT-OSS ensemble.

#SBATCH --job-name=toolsandbox_ensemble
#SBATCH --partition=general-gpu
#SBATCH --account=compute2-yvorobeychik
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:3
#SBATCH --time=06:00:00
#SBATCH --output=logs/compute2/%j.toolsandbox_ensemble.out
#SBATCH --error=logs/compute2/%j.toolsandbox_ensemble.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
export WF="${WF:-$STOR/workflow}"
export BR="${BR:-$WF/benchmark_results}"
export TOOLSANDBOX_DIR="${TOOLSANDBOX_DIR:-$WF/external/ToolSandbox}"

mkdir -p "$WF/external" "$BR" "$PROJ/logs/compute2" "$PROJ/tmp" "$PROJ/.home-c2" "$PROJ/.hf_cache" "$PROJ/.cache-c2"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$WF/scripts:$TOOLSANDBOX_DIR:${PYTHONPATH:-}"

source "$PROJ/.venv-c2/bin/activate"

if [ ! -d "$TOOLSANDBOX_DIR/.git" ]; then
  git clone https://github.com/apple/ToolSandbox.git "$TOOLSANDBOX_DIR"
fi

python -m pip install -q -e "$TOOLSANDBOX_DIR" --no-deps
python -m pip install -q --upgrade \
  "decorator==5.1.1" \
  "langchain-core==0.1.23" \
  "polars==0.20.31" \
  "StrEnum==0.4.15" \
  "ccy==1.3.1" \
  "geopy==2.4.1" \
  "holidays==0.51" \
  "jsonschema==4.19.2" \
  "phonenumbers==8.13.39" \
  "pint>=0.24.4" \
  "rapidfuzz==3.9.3" \
  "rouge-score==0.1.2" \
  "tree-sitter==0.22.3" \
  "tree-sitter-languages==1.10.2"

export TAU_ENSEMBLE_EXECUTOR_ADAPTER="${TAU_ENSEMBLE_EXECUTOR_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-nemotron-agentic-bfcl-lora-2xa40-noGC1024}"
export TAU_ENSEMBLE_PLANNER_ADAPTER="${TAU_ENSEMBLE_PLANNER_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-taskdecomp-lora}"
export TAU_ENSEMBLE_RECOVERY_ADAPTER="${TAU_ENSEMBLE_RECOVERY_ADAPTER:-$PROJ/checkpoints/gpt-oss-20b-taskbench-taskdecomp-lora}"

JOB_ID="${SLURM_JOB_ID:-manual}"
SCENARIOS="${SCENARIOS:-wifi_off}"
LOG_DIR="${LOG_DIR:-$BR/toolsandbox_ensemble_${JOB_ID}}"
mkdir -p "$LOG_DIR"

echo "Running ToolSandbox ensemble smoke"
echo "Scenarios: $SCENARIOS"
echo "Results: $LOG_DIR"

python "$WF/scripts/tool_sandbox_ensemble_runner.py" \
  --output-dir "$LOG_DIR" \
  --scenarios $SCENARIOS
