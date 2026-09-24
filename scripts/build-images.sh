#!/usr/bin/env bash
# Build every image THIS tree's stacks run, under THIS tree's tags.
#
#   ./scripts/build-images.sh                                   # main: heco-*:latest + heco-{persons,faces,embed}:cuda
#   HECO_IMAGE_TAG=shm HECO_CUDA_TAG=shmcuda ./scripts/build-images.sh   # the shm tree (shm-up.sh build does this)
#   HECO_TRT=1 ./scripts/build-images.sh      # with TensorRT baked in (+8 GB per GPU image) — only for the TRT lever
#   HECO_HWDEC=1 ./scripts/build-images.sh    # also the NVDEC ingest image (heco-ingest:hwdec)
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
TRT=${HECO_TRT:-0}
export HECO_IMAGE_TAG=$TAG

# DISK GUARD, before anything is written. On 2026-09-24 the .94 box's WSL
# crashed mid-build (bus errors, then the VM gone) because the WINDOWS drive
# under WSL's virtual disk was full: 175 GB of build cache and six 12.9 GB
# TensorRT images had grown ext4.vhdx to 232 GB, and the next layer had
# nowhere to go. Linux reported plenty of room right up to the crash — the
# drive that matters is the one holding the .vhdx, so both are checked.
MIN_FREE_GB=${HECO_BUILD_MIN_FREE_GB:-40}
HOST_DRIVE=${HECO_BUILD_HOST_DRIVE:-/mnt/c}
check_free() {
  local path=$1 label=$2 avail
  [ -d "$path" ] || return 0
  avail=$(df -BG --output=avail "$path" | tail -1 | tr -dc '0-9')
  if [ "${avail:-0}" -lt "$MIN_FREE_GB" ]; then
    echo "REFUSING TO BUILD: $label ($path) has ${avail} GB free, under ${MIN_FREE_GB} GB." >&2
    echo "  Free space first (docker builder prune -a -f; docker image prune -f), and on WSL" >&2
    echo "  compact the .vhdx from Windows — see docs/LEVERS-2026-09-25.md." >&2
    exit 3
  fi
}
check_free / "the Linux disk"
check_free "$HOST_DRIVE" "the Windows drive holding WSL's virtual disk"

# The dependency base carries no repo code, so every tree shares it; build it
# only when it is missing.
docker image inspect heco-pipeline-base:latest >/dev/null 2>&1 \
  || docker build -f docker/base.Dockerfile -t heco-pipeline-base:latest .

docker compose -f docker-compose.yml build
for svc in persons faces embed; do
  docker build -f docker/persons-cuda.Dockerfile \
    --build-arg BASE="heco-$svc:$TAG" --build-arg WITH_TENSORRT="$TRT" -t "heco-$svc:$CUDA" .
done
# The hardware-decode ingest image, only when that lever is being built for.
if [ "${HECO_HWDEC:-0}" = "1" ]; then
  docker build -f docker/ingest-hwdec.Dockerfile \
    --build-arg BASE="heco-ingest:$TAG" -t "heco-ingest:${HECO_HWDEC_TAG:-hwdec}" .
fi
echo "built heco-*:$TAG and heco-{persons,faces,embed}:$CUDA (TensorRT: $TRT) from $(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty')"

# CLEAN UP AFTER EVERY BUILD (operator's standing ask, 2026-09-24). Dangling
# images are the previous build's layers now that the tags moved on — the
# prune never touches an image a container still uses. The build cache is
# trimmed to HECO_BUILD_CACHE_KEEP (default 10 GB, most-recently-used kept),
# which keeps the pip wheel cache that makes the next rebuild minutes, not a
# re-download. HECO_BUILD_CACHE_KEEP=0 clears it completely.
KEEP=${HECO_BUILD_CACHE_KEEP:-10GB}
docker image prune -f >/dev/null
if [ "$KEEP" = "0" ]; then
  docker builder prune -a -f >/dev/null
else
  docker builder prune -a -f --reserved-space "$KEEP" >/dev/null 2>&1 \
    || docker builder prune -a -f --keep-storage "$KEEP" >/dev/null
fi
echo "after cleanup:"
docker system df --format '  {{.Type}}: {{.Size}} ({{.Reclaimable}} reclaimable)'
df -h / "$HOST_DRIVE" 2>/dev/null | sed 's/^/  /'

