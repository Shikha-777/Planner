#!/usr/bin/env bash
# Probe what container/runtime support is visible from an allocated Compute2 job.

#SBATCH --job-name=c2_container_probe
#SBATCH --account=compute2-yvorobeychik
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.container_probe.out
#SBATCH --error=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/%j.container_probe.err

set -euo pipefail

export STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
export PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
mkdir -p "$PROJ/logs/compute2"

echo "=== host ==="
hostname
date
id
pwd

echo "=== slurm ==="
env | sort | grep -E '^(SLURM|CUDA|NVIDIA|APPTAINER|SINGULARITY|CONTAINER|DOCKER|PODMAN)_' || true
srun --version || true

echo "=== python ==="
command -v python || true
python --version || true
command -v python3 || true
python3 --version || true

echo "=== gpu ==="
command -v nvidia-smi || true
nvidia-smi || true

echo "=== modules ==="
if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
type module || true
module --version 2>&1 || true
module avail 2>&1 | sed -n '1,220p' || true
module spider apptainer 2>&1 || true
module spider singularity 2>&1 || true
module spider docker 2>&1 || true

echo "=== runtimes ==="
for exe in docker podman apptainer singularity enroot; do
  echo "--- $exe ---"
  command -v "$exe" || true
  "$exe" --version 2>&1 || true
done

echo "=== optional container smoke ==="
PROBE_IMAGE="${PROBE_IMAGE:-docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime}"
if command -v apptainer >/dev/null 2>&1; then
  apptainer exec --nv "docker://$PROBE_IMAGE" python - <<'PY' || true
import torch
print("apptainer torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
PY
elif command -v singularity >/dev/null 2>&1; then
  singularity exec --nv "docker://$PROBE_IMAGE" python - <<'PY' || true
import torch
print("singularity torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
PY
elif command -v docker >/dev/null 2>&1; then
  docker run --rm --gpus all "$PROBE_IMAGE" python - <<'PY' || true
import torch
print("docker torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
PY
else
  echo "No direct container runtime command was visible inside this allocation."
fi

echo "DONE"
