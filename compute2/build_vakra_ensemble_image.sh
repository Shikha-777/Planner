#!/usr/bin/env bash
# Build an amd64 VAKRA image with the current ensemble runner included.
#
# Examples:
#   IMAGE=ghcr.io/<user>/vakra-ensemble:amd64 \
#   BASE_IMAGE=ghcr.io/<user>/vakra-base:amd64 \
#   PUSH=1 compute2/build_vakra_ensemble_image.sh
#
#   IMAGE=vakra-ensemble:amd64 LOAD=1 compute2/build_vakra_ensemble_image.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLATFORM="${PLATFORM:-linux/amd64}"
IMAGE="${IMAGE:-vakra-ensemble:amd64}"
BASE_IMAGE="${BASE_IMAGE:-vakra-base:amd64}"
PUSH="${PUSH:-0}"
LOAD="${LOAD:-1}"

if [ "$PUSH" = "1" ] && [[ "$IMAGE" != */* ]]; then
  echo "PUSH=1 requires IMAGE to include a registry/repository, e.g. ghcr.io/user/vakra-ensemble:amd64" >&2
  exit 2
fi

if [ "$PUSH" = "1" ] && [[ "$BASE_IMAGE" != */* ]]; then
  echo "PUSH=1 requires BASE_IMAGE to include a registry/repository, e.g. ghcr.io/user/vakra-base:amd64" >&2
  exit 2
fi

if ! docker buildx version >/dev/null 2>&1; then
  echo "docker buildx is required for --platform $PLATFORM builds." >&2
  exit 2
fi

OUTPUT_FLAG=()
if [ "$PUSH" = "1" ]; then
  OUTPUT_FLAG=(--push)
elif [ "$LOAD" = "1" ]; then
  OUTPUT_FLAG=(--load)
fi

cd "$ROOT"

echo "Building VAKRA base image: $BASE_IMAGE ($PLATFORM)"
docker buildx build \
  --platform "$PLATFORM" \
  -t "$BASE_IMAGE" \
  -f external/vakra/docker/Dockerfile.unified \
  "${OUTPUT_FLAG[@]}" \
  external/vakra

echo "Building VAKRA ensemble image: $IMAGE ($PLATFORM)"
docker buildx build \
  --platform "$PLATFORM" \
  --build-arg "VAKRA_BASE_IMAGE=$BASE_IMAGE" \
  -t "$IMAGE" \
  -f compute2/Dockerfile.vakra_ensemble \
  "${OUTPUT_FLAG[@]}" \
  .

cat <<EOF

Built:
  base:     $BASE_IMAGE
  ensemble: $IMAGE

For compute2, push both images to a registry RIS can pull from:
  IMAGE=<registry>/vakra-ensemble:amd64 BASE_IMAGE=<registry>/vakra-base:amd64 PUSH=1 $0
EOF
