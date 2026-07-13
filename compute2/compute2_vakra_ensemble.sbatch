#!/usr/bin/env bash
# Run the current ensemble on VAKRA inside a single container job.
# Defaults are a one-sample capability-1 smoke. Override DOMAIN/CAPABILITY_ID
# and MAX_SAMPLES for larger runs.

#SBATCH --job-name=vakra_ensemble
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.vakra_ensemble.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.vakra_ensemble.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
export WF="${WF:-$STOR/workflow}"
export BR="${BR:-$WF/benchmark_results}"
export VAKRA_ROOT="${VAKRA_ROOT:-$WF/external/vakra}"
export VAKRA_DB_ROOT="${VAKRA_DB_ROOT:-$VAKRA_ROOT/data/databases}"
export VAKRA_CONFIG_ROOT="${VAKRA_CONFIG_ROOT:-$VAKRA_ROOT/environment/configs}"
export VAKRA_CHROMA_ROOT="${VAKRA_CHROMA_ROOT:-$VAKRA_ROOT/data/retrievers/chroma_data}"
export VAKRA_QUERIES_ROOT="${VAKRA_QUERIES_ROOT:-$VAKRA_ROOT/data/retrievers/queries}"

mkdir -p "$PROJ/logs/compute2" "$BR" "$PROJ/tmp" "$PROJ/.home-c2" "$PROJ/.hf_cache" "$PROJ/.cache-c2"

export CAPABILITY_ID="${CAPABILITY_ID:-1}"
export DOMAIN="${DOMAIN:-beer_factory}"
export AGENT="${AGENT:-ensemble}"
export MAX_SAMPLES="${MAX_SAMPLES:-1}"
export MAX_STEPS="${MAX_STEPS:-8}"
export JOB_ID="${SLURM_JOB_ID:-manual}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/results/vakra_${DOMAIN}_cap${CAPABILITY_ID}_${JOB_ID}}"

# Registry image built by compute2/build_vakra_ensemble_image.sh.
export VAKRA_IMAGE="${VAKRA_IMAGE:-ghcr.io/REPLACE_ME/vakra-ensemble:amd64}"

echo "HOST=$(hostname)"
echo "VAKRA_IMAGE=$VAKRA_IMAGE"
echo "CAPABILITY_ID=$CAPABILITY_ID DOMAIN=$DOMAIN MAX_SAMPLES=$MAX_SAMPLES AGENT=$AGENT"
echo "PROJ=$PROJ"
echo "VAKRA_ROOT=$VAKRA_ROOT"
echo "VAKRA_DB_ROOT=$VAKRA_DB_ROOT"
nvidia-smi || true

if [[ "$VAKRA_IMAGE" == *REPLACE_ME* ]]; then
  echo "Set VAKRA_IMAGE to the pushed amd64 image, e.g. ghcr.io/<user>/vakra-ensemble:amd64" >&2
  exit 2
fi

if [ ! -d "$VAKRA_ROOT" ]; then
  echo "VAKRA_ROOT does not exist: $VAKRA_ROOT" >&2
  exit 2
fi

if [ ! -d "$VAKRA_DB_ROOT" ]; then
  echo "VAKRA_DB_ROOT does not exist: $VAKRA_DB_ROOT" >&2
  exit 2
fi

CONTAINER_CMD=(bash /workspace/scripts/run_vakra_inside_container.sh)
RUNNER_KIND="${RUNNER_KIND:-auto}"

if [ "$RUNNER_KIND" = "auto" ]; then
  if command -v apptainer >/dev/null 2>&1; then
    RUNNER_KIND="apptainer"
  elif command -v singularity >/dev/null 2>&1; then
    RUNNER_KIND="singularity"
  elif command -v docker >/dev/null 2>&1; then
    RUNNER_KIND="docker"
  else
    RUNNER_KIND="direct"
  fi
fi

echo "RUNNER_KIND=$RUNNER_KIND"

case "$RUNNER_KIND" in
  apptainer|singularity)
    RUNTIME="$RUNNER_KIND"
    IMAGE_REF="$VAKRA_IMAGE"
    if [[ "$IMAGE_REF" != docker://* ]] && [[ "$IMAGE_REF" != /* ]] && [[ "$IMAGE_REF" != *.sif ]]; then
      IMAGE_REF="docker://$IMAGE_REF"
    fi
    BINDS=(
      "$PROJ:/workspace/project"
      "$VAKRA_ROOT:/workspace/vakra"
      "$BR:/workspace/results"
      "$VAKRA_DB_ROOT:/app/db:ro"
      "$VAKRA_CONFIG_ROOT:/app/environment/configs:ro"
    )
    if [ -d "$VAKRA_CHROMA_ROOT" ]; then
      BINDS+=("$VAKRA_CHROMA_ROOT:/app/retrievers/chroma_data:ro")
    fi
    if [ -d "$VAKRA_QUERIES_ROOT" ]; then
      BINDS+=("$VAKRA_QUERIES_ROOT:/app/retrievers/queries:ro")
    fi
    BIND_ARG="$(IFS=,; echo "${BINDS[*]}")"
    "$RUNTIME" exec --nv --bind "$BIND_ARG" "$IMAGE_REF" "${CONTAINER_CMD[@]}"
    ;;
  docker)
    DOCKER_IMAGE="${VAKRA_IMAGE#docker://}"
    DOCKER_ARGS=(
      run --rm --gpus all
      --user "$(id -u):$(id -g)"
      -e "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
      -e "CAPABILITY_ID=$CAPABILITY_ID"
      -e "DOMAIN=$DOMAIN"
      -e "AGENT=$AGENT"
      -e "MAX_SAMPLES=$MAX_SAMPLES"
      -e "MAX_STEPS=$MAX_STEPS"
      -e "OUTPUT_DIR=$OUTPUT_DIR"
      -v "$PROJ:/workspace/project"
      -v "$VAKRA_ROOT:/workspace/vakra"
      -v "$BR:/workspace/results"
      -v "$VAKRA_DB_ROOT:/app/db:ro"
      -v "$VAKRA_CONFIG_ROOT:/app/environment/configs:ro"
    )
    if [ -d "$VAKRA_CHROMA_ROOT" ]; then
      DOCKER_ARGS+=(-v "$VAKRA_CHROMA_ROOT:/app/retrievers/chroma_data:ro")
    fi
    if [ -d "$VAKRA_QUERIES_ROOT" ]; then
      DOCKER_ARGS+=(-v "$VAKRA_QUERIES_ROOT:/app/retrievers/queries:ro")
    fi
    docker "${DOCKER_ARGS[@]}" "$DOCKER_IMAGE" "${CONTAINER_CMD[@]}"
    ;;
  direct)
    echo "No direct container runtime command was found. If RIS launches this script inside the image, set RUNNER_KIND=direct." >&2
    bash "$PROJ/compute2/run_vakra_inside_container.sh"
    ;;
  *)
    echo "Unknown RUNNER_KIND=$RUNNER_KIND; expected auto, apptainer, singularity, docker, or direct." >&2
    exit 2
    ;;
esac

echo "DONE"
