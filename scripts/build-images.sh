#!/usr/bin/env bash
# Build every image THIS tree's stacks run, under THIS tree's tags.
#
#   ./scripts/build-images.sh                                   # main: heco-*:latest + heco-{persons,faces,embed}:cuda
#   HECO_IMAGE_TAG=shm HECO_CUDA_TAG=shmcuda ./scripts/build-images.sh   # the shm tree (shm-up.sh build does this)
#
# WHY THE TAGS ARE PARAMETERS. Code is baked into the service images, and both
# trees on the .94 box used to build the SAME tags (heco-runner, heco-persons:cuda,
# ...). Whichever tree built last therefore supplied the code for ALL THREE
# stacks on their next recreate — the camera stacks could be running the
# shared-memory branch, or the shm candidate main's code, with nothing on screen
# to say so. Each tree now builds, and its stacks now run, its own tags.
#
# ORDER IS LOAD-BEARING. The :cuda images are built FROM the plain ones
# (docker/persons-cuda.Dockerfile adds the GPU wheels on top), so the plain
# build comes first. Never build through docker-compose.gpumax.yml — its
# header says why (it would tag a CPU image as the CUDA one).
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=${HECO_IMAGE_TAG:-latest}
CUDA=${HECO_CUDA_TAG:-cuda}
export HECO_IMAGE_TAG=$TAG

# The dependency base carries no repo code, so every tree shares it; build it
# only when it is missing.
docker image inspect heco-pipeline-base:latest >/dev/null 2>&1 \
  || docker build -f docker/base.Dockerfile -t heco-pipeline-base:latest .

docker compose -f docker-compose.yml build
for svc in persons faces embed; do
  docker build -f docker/persons-cuda.Dockerfile \
    --build-arg BASE="heco-$svc:$TAG" -t "heco-$svc:$CUDA" .
done
echo "built heco-*:$TAG and heco-{persons,faces,embed}:$CUDA from $(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty')"
