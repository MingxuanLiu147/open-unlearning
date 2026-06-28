#!/bin/bash
set -e

REGISTRY="210.75.240.150:30003"
USER="liumingxuan"
IMAGE_NAME="open-unlearning"
TAG="${1:-latest}"

FULL_IMAGE="${REGISTRY}/${USER}/${IMAGE_NAME}:${TAG}"

echo "=== Building Docker image: ${FULL_IMAGE} ==="
cd "$(dirname "$0")/../.."

docker build -t "${FULL_IMAGE}" -f Dockerfile .

echo ""
echo "=== Pushing to registry ==="
docker push "${FULL_IMAGE}"

echo ""
echo "=== Done ==="
echo "Image: ${FULL_IMAGE}"
echo "Use this in your PAI job config."
