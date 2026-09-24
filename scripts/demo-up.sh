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
#   HECO_REVIEW_FLOOR=0.28  unless already set. The duplicate-review band's
#                           floor is 0.15 in the service — SFace's number — and
#                           on run f0bfc5 (ArcFace, 74 guests) that queued 500
#                           pairs, 468 of them under 0.28: a man beside an
#                           elderly woman, a girl beside a woman. At 0.28 the
#                           same run queues 32. The camera stacks were meant to
#                           carry this all along; the shm stack shipped without
#                           it, which is why the default now lives HERE and not
#                           only in a shell history.
#
# THE REVIEW EXCLUSIONS (gender, age, stature, clothes, head, beard) and the
# track-presence split are NOT exported here: their defaults live in the
# services (match config.py, runner config.py) and are on out of the box.
# `status` prints what the running containers actually hold, so an operator
# can see a knob that was set — or one that was not — without reading a
# compose file.
#
# THE THROUGHPUT LEVERS (docs/LEVERS-2026-09-25.md) are OFF unless the caller
# sets them: nothing below exports one, and compose passes an unset knob
# through as the service's default, which for every lever is off. Set them on
# the command line, per camera if they should differ:
#
#   HECO_PIPELINE_OVERLAP=1 HECO_PARALLEL_DETECT=1 ./scripts/demo-up.sh a
#   HECO_FACE_CADENCE=1 HECO_FACE_REVERIFY_INTERVAL_S=2.5 ./scripts/demo-up.sh b
#   HECO_HWDEC=1 INGEST_DECODER=nvdec INGEST_CV_THREADS=1 ./scripts/demo-up.sh both
#   HECO_TRT=1 FACES_SCRFD_INPUT=1472x832 ./scripts/demo-up.sh both
#
# Two levers need an overlay as well as a variable; these two switches append
# it (and are off by default like the rest):
#
#   HECO_TRT=1     docker-compose.trt.yml, LAST: persons/faces/embed on
#                  TensorRT fp16. The :cuda images must carry TensorRT —
#                  build them with HECO_TRT=1 ./scripts/build-images.sh; a plain
#                  build leaves it out (f1b2f07) — and each model's
#                  first start builds its engine: YOLOX-s ~157 s, SCRFD ~34 s,
#                  ArcFace ~35 s, genderage ~20 s — warm them before a count.
#   HECO_HWDEC=1   docker-compose.hwdec.yml: the ffmpeg ingest image
#                  (heco-ingest:${HECO_HWDEC_TAG:-hwdec}, built by hand from
#                  docker/ingest-hwdec.Dockerfile) and the WSL GPU wiring, for
#                  INGEST_DECODER=nvdec. Without it a hardware decoder falls
#                  back to cpu, loudly — `status` shows which one ran.
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
# Raw ArcFace feature-norm floor, measured on the Sharon re-run c84098: at 18
# exactly one identity loses every template — p00047, a girl half hidden
# behind a pillar, the half-face the operator asked to be ignored — and no
# real guest does (the two single-sighting guests a 0.77 detector-confidence
# floor would have erased read 22+). 2.8% of templates dropped, all from
# identities that keep better ones. 0 turns it off.
export HECO_QUALITY_MIN_FEAT_NORM=${HECO_QUALITY_MIN_FEAT_NORM:-18}
# Half-balance floor, measured on run f0bfc5's sightings: p00002 — a girl with
# a dark railing across her face, which the norm floor passes (23.7) — reads
# 0.27; the lowest genuine face 0.42, the 5th percentile 0.51. 0 turns it off.
export HECO_QUALITY_MIN_BALANCE=${HECO_QUALITY_MIN_BALANCE:-0.33}
export PLANNER_URL=${PLANNER_URL:-http://192.168.1.55:8787}

A=(-f docker-compose.yml -f docker-compose.gpumax.yml)
B=("${A[@]}" -f docker-compose.camB.yml)
# The lever overlays, appended to both stacks and after camB: TensorRT's header
# asks to be layered last, over everything that names the model services.
EXTRA=()
if [ "${HECO_HWDEC:-0}" = 1 ]; then EXTRA+=(-f docker-compose.hwdec.yml); fi
if [ "${HECO_TRT:-0}" = 1 ]; then EXTRA+=(-f docker-compose.trt.yml); fi
if [ ${#EXTRA[@]} -gt 0 ]; then A+=("${EXTRA[@]}"); B+=("${EXTRA[@]}"); fi
case "${INGEST_DECODER:-cpu}" in
  cpu|'') ;;
  *) [ "${HECO_HWDEC:-0}" = 1 ] || echo "note: INGEST_DECODER=${INGEST_DECODER} without HECO_HWDEC=1 runs the plain ingest image, which has no ffmpeg — it will fall back to cpu (status shows it)" >&2 ;;
esac

# The knobs a RUNNING stack holds, read out of its containers and not out of
# this shell: `status` runs the same exports `up` does, so the shell would
# say HECO_REVIEW_FLOOR=0.28 about a stack that was started last week at
# 0.15. Empty (·) means the service's own default (listed once, below).
show() {
  local name=$1 svc=$2; shift 2
  local vars=() env k v
  while [ "$1" != -- ]; do vars+=("$1"); shift; done
  shift
  printf '  camera %s  %-7s ' "$name" "$svc:"
  if env=$(docker compose "$@" exec -T "$svc" printenv 2>/dev/null); then
    for k in "${vars[@]}"; do
      v=$(printf '%s\n' "$env" | sed -n "s/^$k=//p")
      printf '%s=%s ' "${k#HECO_}" "${v:-·}"
    done
    echo
  else
    echo "($svc not up)"
  fi
}

knobs() {
  local name=$1; shift
  show "$name" match HECO_REVIEW_FLOOR HECO_REVIEW_GENDER_MIN_P HECO_REVIEW_AGE_CHILD_MAX \
       HECO_REVIEW_AGE_ADULT_MIN HECO_REVIEW_STATURE_GAP HECO_REVIEW_STATURE_MIN_N \
       HECO_STATURE_ADULT_M HECO_REVIEW_CLOTHES_CLASH HECO_REVIEW_CLOTHES_MIN_N \
       HECO_REVIEW_CLOTHES_SELF_MIN HECO_REVIEW_HEAD_CLASH HECO_REVIEW_BEARD_MIN_N -- "$@"
  show "$name" runner HECO_PRESENCE_SPLIT HECO_COPRESENCE_SPLIT HECO_QUALITY_MIN_FEAT_NORM HECO_QUALITY_MIN_BALANCE \
       HECO_APPEARANCE_WB HECO_PIPELINE_OVERLAP HECO_PARALLEL_DETECT HECO_FACE_CADENCE \
       HECO_FACE_CADENCE_MAX_GAP_S HECO_FACE_REVERIFY_INTERVAL_S -- "$@"
  show "$name" ingest INGEST_MOTION_GATE INGEST_MOTION_MIN_FRAC INGEST_MOTION_PIXEL_THR \
       INGEST_MOTION_KEEPALIVE_S INGEST_BUFFER_S INGEST_BUFFER_MB INGEST_DECODER \
       INGEST_CV_THREADS INGEST_LIVE_TIMEOUT_S -- "$@"
  show "$name" faces HECO_DEVICE FACES_MODEL FACES_SCRFD_INPUT HECO_TRT_CACHE -- "$@"
  show "$name" embed HECO_DEVICE EMBED_BATCH HECO_TRT_CACHE -- "$@"
}

status() {
  for pair in "A 7100 7101 7102 7103 7104 7105 7106" "B 7200 7201 7202 7203 7204 7205 7206"; do
    set -- $pair; name=$1; shift
    printf 'camera %s: ' "$name"
    for p in "$@"; do printf ':%s=%s ' "$p" "$(curl -s -m 5 "http://localhost:$p/health" -o /dev/null -w '%{http_code}')"; done
    echo
  done
  echo '--- device truth (requested vs ACTIVE; CPU here means the GPU is not being used) ---'
  for p in 7101 7102 7104 7105 7201 7202 7204 7205; do
    PORT=$p python3 - <<'PY' 2>/dev/null
import json, os, urllib.request
port = os.environ["PORT"]
try:
    d = json.load(urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5))
except Exception:
    raise SystemExit
dev = d.get("device") or {}
if port.endswith("01"):
    # INGEST's device block is the DECODER (L6): requested vs what the open
    # capture decodes with (null until a source is open), and why not.
    err = dev.get("error") or ""
    if err:
        down = err.startswith("live source down")
        err = f"   <-- {'SOURCE DOWN' if down else 'FELL BACK'}: {err}"
    print(f"  :{port} ingest decoder {dev.get('requested')} -> {dev.get('active')}{err}")
    raise SystemExit
active = (dev.get("active") or ["?"])[0]
requested = str(dev.get("requested"))
on_gpu = active.startswith(("CUDA", "Tensorrt"))
flag = "" if on_gpu or requested in ("None", "CPU") else "   <-- NOT ON THE GPU"
if requested.upper() in ("TRT", "TENSORRT") and not active.startswith("Tensorrt"):
    flag += "   <-- TensorRT requested, not running"
print(f"  :{port} {d.get('model','?'):34s} ok={str(d.get('ok')):5s} "
      f"{requested} -> {active}{flag}")
if dev.get("trt"):
    print(f"        tensorrt: {dev['trt']}")
if dev.get("attributes"):  # embed under TRT: the gender/age pass's own truth
    att = dev["attributes"]
    print(f"        attribute pass: {att.get('requested')} -> {(att.get('active') or ['?'])[0]}"
          f"   tensorrt: {att.get('trt')}")
# The attribute head is the embed service's second model and its own truth:
# absent, every review pair reads "not measured" for gender and age and the
# gender/age exclusions never fire. Say so here rather than in a queue.
if port.endswith("05"):
    if "attrModel" not in d:
        print(f"        attribute model: not reported — this embed image predates the attribute head")
    else:
        print(f"        attribute model: {d['attrModel'] or 'NONE — gender/age will read not measured'}")
PY
  done
  echo '--- review, presence and lever knobs as the containers hold them (· = the service default) ---'
  knobs A "${A[@]}"
  knobs B -p heco-pipeline-b "${B[@]}"
  echo '    defaults: REVIEW_FLOOR 0.15 in the service (0.28 from this script) · GENDER_MIN_P 0.8 ·'
  echo '    AGE_CHILD_MAX 12 / AGE_ADULT_MIN 20 · STATURE_GAP 0.2 · STATURE_MIN_N 8 · STATURE_ADULT_M 1.75 ·'
  echo '    CLOTHES_CLASH 0.35 · CLOTHES_MIN_N 3 · CLOTHES_SELF_MIN 0.6 · HEAD_CLASH 0.45 · BEARD_MIN_N 3 ·'
  echo '    PRESENCE_SPLIT 1 · COPRESENCE_SPLIT 1 · QUALITY_MIN_FEAT_NORM 0 in the service (18 from this script) ·'
  echo '    QUALITY_MIN_BALANCE 0 in the service (0.33 from this script). 0 turns a signal off.'
  echo '    levers, all OFF by default: APPEARANCE_WB 0 · PIPELINE_OVERLAP 0 · PARALLEL_DETECT 0 · FACE_CADENCE 0'
  echo '    (MAX_GAP_S 1.0; skips nothing while FACE_REVERIFY_INTERVAL_S is 0) · INGEST_MOTION_GATE 0 (MIN_FRAC 0.002,'
  echo '    PIXEL_THR 0.08, KEEPALIVE_S 1.0) · INGEST_BUFFER_S 0 (MB 2048) · INGEST_DECODER cpu · INGEST_CV_THREADS unset ·'
  echo '    INGEST_LIVE_TIMEOUT_S 0 (OpenCV 30 s / ffmpeg own timeouts; 10 recommended for live cameras) ·'
  echo '    EMBED_BATCH 0 · FACES_SCRFD_INPUT 640 · TensorRT only with HECO_TRT=1 (device truth above).'
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
