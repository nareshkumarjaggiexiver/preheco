#!/usr/bin/env bash
# Bring up the demo configuration on the CUDA box: one stack per camera, every
# model stage on the GPU. One place for the env, because the deploy line is
# fifteen variables long and a demo is the worst time to retype it.
#
#   ./scripts/demo-up.sh          both cameras
#   ./scripts/demo-up.sh a        camera A only (ports 7100-7106)
#   ./scripts/demo-up.sh status   what is actually up, with device truth
#
# THE SETTINGS, AND WHY EACH ONE:
#
#   HECO_CPUSET=            the base compose pins the even CPUs of a 64-thread
#                           T440, a list running to cpu 62. Any smaller box has
#                           docker refuse the container outright.
#   HECO_DEVICE=CUDA        persons, faces and embed all on the 4060. Verify it
#                           in `status` — a silent CPU fallback is the failure
#                           this project's device-truth checks exist to catch.
#   FACES_SCRFD_SCORE_MIN   0.7, not the family default 0.5. At 0.5 the detector
#                           invents faces on the back of a head, with plausible
#                           landmarks, and every geometric gate agrees with it.
#   HECO_FACES_WHOLE_FRAME  one inference per frame instead of one per person:
#                           4x faster and MORE faces over the 112 px gate,
#                           because it does not need the person detector to have
#                           found somebody first.
#   HECO_MATCH_NEARMISS_FLOOR=0
#                           the near-miss bands are SFace's numbers (0.29/0.15)
#                           and this runs ArcFace. Left armed they suggest a
#                           child and an adult are the same guest, on clothing
#                           similarity alone. Off until eval/sweep.py calibrates.
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
export PLANNER_URL=${PLANNER_URL:-http://192.168.1.55:8787}

A=(-f docker-compose.yml -f docker-compose.gpumax.yml)
B=("${A[@]}" -f docker-compose.camB.yml)

status() {
  for pair in "A 7100 7101 7102 7103 7104 7105 7106" "B 7200 7201 7202 7203 7204 7205 7206"; do
    set -- $pair; name=$1; shift
    printf 'camera %s: ' "$name"
    for p in "$@"; do printf ':%s=%s ' "$p" "$(curl -s -m 5 "http://localhost:$p/health" -o /dev/null -w '%{http_code}')"; done
    echo
  done
  echo '--- device truth (requested vs ACTIVE; CPU here means the GPU is not being used) ---'
  for p in 7102 7104 7105 7202 7204 7205; do
    PORT=$p python3 - <<'PY' 2>/dev/null
import json, os, urllib.request
port = os.environ["PORT"]
try:
    d = json.load(urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5))
except Exception:
    raise SystemExit
dev = d.get("device") or {}
active = (dev.get("active") or ["?"])[0]
flag = "" if active.startswith("CUDA") or dev.get("requested") in (None, "CPU") else "   <-- NOT ON THE GPU"
print(f"  :{port} {d.get('model','?'):34s} ok={str(d.get('ok')):5s} "
      f"{dev.get('requested')} -> {active}{flag}")
PY
  done
}

case "${1:-both}" in
  a)      docker compose "${A[@]}" up -d ;;
  b)      docker compose -p heco-pipeline-b "${B[@]}" up -d ;;
  both)   docker compose "${A[@]}" up -d
          docker compose -p heco-pipeline-b "${B[@]}" up -d ;;
  down)   docker compose "${A[@]}" down || true
          docker compose -p heco-pipeline-b "${B[@]}" down || true; exit 0 ;;
  status) status; exit 0 ;;
  *) echo "usage: $0 [both|a|b|down|status]" >&2; exit 2 ;;
esac

sleep 20
status
