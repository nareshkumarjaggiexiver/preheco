#!/usr/bin/env bash
# Launch the seven heco-pipeline services natively (no docker) on the contract
# ports, with the persons stage on CUDA. Logs land in $LOGDIR; PIDs in $PIDFILE.
#   run-native-cuda.sh start | stop | status
set -euo pipefail
ROOT=/home/naresh/projects/heco/apps/preheco
LOGDIR=${LOGDIR:-$ROOT/.native-logs}
PIDFILE=$LOGDIR/pids
PLANNER_URL=${PLANNER_URL:-http://localhost:8787}
HECO_DEVICE=${HECO_DEVICE:-CUDA}
PERSONS_MODEL=${PERSONS_MODEL:-yolox_s.onnx}

SP=$ROOT/services/persons/.venv/lib/python3.12/site-packages/nvidia
CUDA_LD=$SP/cuda_runtime/lib:$SP/cublas/lib:$SP/cudnn/lib:$SP/cufft/lib:$SP/curand/lib:$SP/nvjitlink/lib:$SP/cuda_nvrtc/lib:/usr/lib/wsl/lib

declare -A PORT=([ingest]=7101 [persons]=7102 [tracker]=7103 [faces]=7104 [embed]=7105 [match]=7106 [runner]=7100)
ORDER=(ingest persons tracker faces embed match runner)

start_one() {
  local svc=$1 port=${PORT[$1]}
  local env=(PORT="$port")
  case $svc in
    persons) env+=(HECO_DEVICE="$HECO_DEVICE" PERSONS_MODEL="$PERSONS_MODEL" LD_LIBRARY_PATH="$CUDA_LD") ;;
    runner)  env+=(PLANNER_URL="$PLANNER_URL"
                   HECO_INGEST_URL=http://localhost:7101 HECO_PERSONS_URL=http://localhost:7102
                   HECO_TRACKER_URL=http://localhost:7103 HECO_FACES_URL=http://localhost:7104
                   HECO_EMBED_URL=http://localhost:7105 HECO_MATCH_URL=http://localhost:7106
                   HECO_TOKEN_CACHE="$LOGDIR/runner-token.json") ;;
  esac
  ( cd "$ROOT/services/$svc" && env "${env[@]}" nohup .venv/bin/python -m uvicorn app.main:app \
      --host 127.0.0.1 --port "$port" >"$LOGDIR/$svc.log" 2>&1 & echo "$svc $!" >>"$PIDFILE" )
}

case ${1:-start} in
  start)
    mkdir -p "$LOGDIR"; : >"$PIDFILE"
    for s in "${ORDER[@]}"; do start_one "$s"; done
    echo "started — logs in $LOGDIR"; sleep 6; "$0" status ;;
  stop)
    [ -f "$PIDFILE" ] && while read -r s p; do kill "$p" 2>/dev/null && echo "stopped $s ($p)"; done <"$PIDFILE"
    rm -f "$PIDFILE" ;;
  status)
    for s in "${ORDER[@]}"; do
      printf '%-8s :%s  ' "$s" "${PORT[$s]}"
      curl -s -m 3 "http://127.0.0.1:${PORT[$s]}/health" || printf 'DOWN'; echo
    done ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
