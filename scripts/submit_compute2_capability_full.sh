#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-dshrestha@c2-login-001.ris.wustl.edu}"
STOR="${STOR:-/storage3/fs1/yvorobeychik/Active/dshrestha}"
PROJ="${PROJ:-$STOR/taskdecomp-gpt-oss}"
DATA_DIR="${DATA_DIR:-data/gpt55_capability_mapping_sft_v3}"
JOB_FILE="${JOB_FILE:-ris/compute2_train_gptoss_capability_mapping_full.sbatch}"

ssh -o BatchMode=yes "$REMOTE" "mkdir -p '$PROJ/scripts' '$PROJ/ris' '$PROJ/$DATA_DIR' '$PROJ/logs/compute2'"

rsync -av \
  scripts/train_sft.py \
  scripts/generate_gpt55_capability_mapping_sft.py \
  "$REMOTE:$PROJ/scripts/"

rsync -av "$JOB_FILE" "$REMOTE:$PROJ/ris/"
rsync -av "$DATA_DIR/" "$REMOTE:$PROJ/$DATA_DIR/"

ssh -o BatchMode=yes "$REMOTE" "cd '$PROJ' && sbatch '$JOB_FILE'"
