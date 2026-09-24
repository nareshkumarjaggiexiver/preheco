#!/usr/bin/env bash
# The shared-memory candidate, on ports 7300-7306, beside the camera stacks.
#
#   ./scripts/shm-up.sh            ref alongside JPEG  (the measured 1.32x)
#   ./scripts/shm-up.sh ref-only   also drop the JPEG  (NOT yet trusted)
#   ./scripts/shm-up.sh status     health + DEVICE TRUTH + tmpfs
#   ./scripts/shm-up.sh down
#
# CHECK `status` BEFORE BELIEVING ANY NUMBER FROM THIS STACK. Three stacks
# share one 8 GB card, and on 2026-09-24 this one's persons/faces/embed came
# up reporting `CUDA -> CPUExecutionProvider` while the camera stacks stayed
# on the GPU. The ref-only throughput measured that day (2.58 fps against
# 5.14) was a CPU pipeline compared with a GPU one and means nothing. The
# status output flags it.
set -euo pipefail
cd "$(dirname "$0")/.."

export HECO_CPUSET=
export HECO_DEVICE=CUDA
export PERSONS_MODEL=yolox_s.onnx
export FACES_MODEL=scrfd_10g_kps.onnx
export FACES_SCRFD_SCORE_MIN=0.7
export HECO_FACES_WHOLE_FRAME=1
export EMBED_MODEL=models/arcface_w600k_r50.onnx
export HECO_EMBEDDING_DIM=512
export HECO_EMBEDDER_ID=arcface-w600k-r50
export HECO_MATCH_NEARMISS_FLOOR=0
export HECO_REVIEW_FLOOR=${HECO_REVIEW_FLOOR:-0.28}
# The measured feature-norm floor, same as demo-up.sh: on the Sharon re-run
# it removes exactly p00047 (a half face behind a pillar) and no real guest.
export HECO_QUALITY_MIN_FEAT_NORM=${HECO_QUALITY_MIN_FEAT_NORM:-18}
export PLANNER_URL=${PLANNER_URL:-http://192.168.1.55:8787}
# THIS TREE'S IMAGES. Main's stacks run heco-*:latest / :cuda; this candidate
# runs :shm / :shmcuda, so building one tree can never swap the other's code
# (it did, until 2026-09-24 — see scripts/build-images.sh).
export HECO_IMAGE_TAG=${HECO_IMAGE_TAG:-shm}
export HECO_CUDA_TAG=${HECO_CUDA_TAG:-shmcuda}

C=(-p heco-shm
   -f docker-compose.yml
   -f docker-compose.gpumax.yml
   -f docker-compose.shm.yml
   -f docker-compose.shmports.yml)

status() {
  printf 'shm: '
  for p in 7300 7301 7302 7303 7304 7305 7306; do
    printf ':%s=%s ' "$p" "$(curl -s -m 5 "http://localhost:$p/health" -o /dev/null -w '%{http_code}')"
  done
  echo
  echo '--- device truth (a stage on CPU makes every number from this stack meaningless) ---'
  for p in 7302 7304 7305; do
    PORT=$p python3 - <<'PY' 2>/dev/null
import json, os, urllib.request
port = os.environ["PORT"]
try:
    d = json.load(urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5))
except Exception:
    raise SystemExit
dev = d.get("device") or {}
active = (dev.get("active") or ["?"])[0]
flag = "" if active.startswith("CUDA") else "   <-- ON CPU, DO NOT BENCHMARK"
print(f"  :{port} {d.get('model','?'):34s} {dev.get('requested')} -> {active}{flag}")
PY
  done
  echo '--- shared frames ---'
  docker exec heco-shm-ingest-1 sh -c 'df -h /frames | tail -1; echo "  frames held: $(ls /frames 2>/dev/null | wc -l)"' 2>/dev/null \
    || echo '  (ingest not up)'
}

case "${1:-both}" in
  build)    ./scripts/build-images.sh; exit 0 ;;
  ref-only) export HECO_FRAMES_REF_ONLY=1; docker compose "${C[@]}" up -d ;;
  both|up)  docker compose "${C[@]}" up -d ;;
  down)     docker compose "${C[@]}" down; exit 0 ;;
  status)   status; exit 0 ;;
  *) echo "usage: $0 [build|up|ref-only|down|status]" >&2; exit 2 ;;
esac

sleep 25
status
