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
#
# THE THROUGHPUT LEVERS (docs/LEVERS-2026-09-25.md) are OFF unless the caller
# sets them, exactly as in demo-up.sh — nothing here exports one:
#
#   HECO_PIPELINE_OVERLAP=1 HECO_PARALLEL_DETECT=1 ./scripts/shm-up.sh
#   HECO_TRT=1 FACES_SCRFD_INPUT=1472x832 ./scripts/shm-up.sh ref-only
#   HECO_HWDEC=1 INGEST_DECODER=nvdec INGEST_CV_THREADS=1 ./scripts/shm-up.sh
#
# HECO_TRT=1 appends docker-compose.trt.yml after the shm overlays (the
# :shmcuda images must carry TensorRT — rebuild with `build`); HECO_HWDEC=1
# appends docker-compose.hwdec.yml, running THIS tree's ffmpeg ingest image,
# heco-ingest:shmhwdec, built by hand FROM heco-ingest:shm:
#   docker build -f docker/ingest-hwdec.Dockerfile --build-arg BASE=heco-ingest:shm \
#     -t heco-ingest:shmhwdec .
# Every lever carries the shared frame the way it carries the JPEG (see
# docs/SHARED-MEMORY-BRANCH.md): frames are written to tmpfs when taken.
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
export HECO_HWDEC_TAG=${HECO_HWDEC_TAG:-shmhwdec}

C=(-p heco-shm
   -f docker-compose.yml
   -f docker-compose.gpumax.yml
   -f docker-compose.shm.yml
   -f docker-compose.shmports.yml)
# The lever overlays, after the shm ones (TensorRT asks to be last).
if [ "${HECO_HWDEC:-0}" = 1 ]; then C+=(-f docker-compose.hwdec.yml); fi
if [ "${HECO_TRT:-0}" = 1 ]; then C+=(-f docker-compose.trt.yml); fi
case "${INGEST_DECODER:-cpu}" in
  cpu|'') ;;
  *) [ "${HECO_HWDEC:-0}" = 1 ] || echo "note: INGEST_DECODER=${INGEST_DECODER} without HECO_HWDEC=1 runs the plain ingest image, which has no ffmpeg — it will fall back to cpu (status shows it)" >&2 ;;
esac

# The knobs the RUNNING containers hold (· = the service default), as in
# demo-up.sh: the shell's exports say nothing about a stack started earlier.
show() {
  local svc=$1; shift
  local env k v
  printf '  %-7s ' "$svc:"
  if env=$(docker compose "${C[@]}" exec -T "$svc" printenv 2>/dev/null); then
    for k in "$@"; do
      v=$(printf '%s\n' "$env" | sed -n "s/^$k=//p")
      printf '%s=%s ' "${k#HECO_}" "${v:-·}"
    done
    echo
  else
    echo "($svc not up)"
  fi
}

status() {
  printf 'shm: '
  for p in 7300 7301 7302 7303 7304 7305 7306; do
    printf ':%s=%s ' "$p" "$(curl -s -m 5 "http://localhost:$p/health" -o /dev/null -w '%{http_code}')"
  done
  echo
  echo '--- device truth (a stage on CPU makes every number from this stack meaningless) ---'
  for p in 7301 7302 7304 7305; do
    PORT=$p python3 - <<'PY' 2>/dev/null
import json, os, urllib.request
port = os.environ["PORT"]
try:
    d = json.load(urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5))
except Exception:
    raise SystemExit
dev = d.get("device") or {}
if port == "7301":
    # ingest's device block is the DECODER (L6), null until a source is open.
    err = f"   <-- FELL BACK: {dev['error']}" if dev.get("error") else ""
    print(f"  :{port} ingest decoder {dev.get('requested')} -> {dev.get('active')}{err}")
    raise SystemExit
active = (dev.get("active") or ["?"])[0]
requested = str(dev.get("requested"))
flag = "" if active.startswith(("CUDA", "Tensorrt")) else "   <-- ON CPU, DO NOT BENCHMARK"
if requested.upper() in ("TRT", "TENSORRT") and not active.startswith("Tensorrt"):
    flag += "   <-- TensorRT requested, not running"
print(f"  :{port} {d.get('model','?'):34s} {requested} -> {active}{flag}")
PY
  done
  echo '--- lever knobs as the containers hold them (· = the service default: every lever off) ---'
  show runner HECO_FRAMES_REF_ONLY HECO_PIPELINE_OVERLAP HECO_PARALLEL_DETECT HECO_FACE_CADENCE \
       HECO_FACE_CADENCE_MAX_GAP_S HECO_FACE_REVERIFY_INTERVAL_S HECO_APPEARANCE_WB
  show ingest HECO_FRAMES_KEEP INGEST_MOTION_GATE INGEST_MOTION_MIN_FRAC INGEST_MOTION_PIXEL_THR \
       INGEST_MOTION_KEEPALIVE_S INGEST_BUFFER_S INGEST_BUFFER_MB INGEST_DECODER INGEST_CV_THREADS
  show faces HECO_DEVICE FACES_MODEL FACES_SCRFD_INPUT HECO_TRT_CACHE
  show embed HECO_DEVICE EMBED_BATCH HECO_TRT_CACHE
  show match HECO_REVIEW_FLOOR HECO_REVIEW_CLOTHES_CLASH HECO_REVIEW_HEAD_CLASH HECO_REVIEW_BEARD_MIN_N
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
