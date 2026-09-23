#!/usr/bin/env bash
# Restart any heco service that stops answering /health. Logs every action.
#   nohup scripts/watchdog.sh >/dev/null 2>&1 &
ROOT=/home/naresh/projects/heco/apps/preheco
LOG=$ROOT/.native-logs/watchdog.log
export PATH="$HOME/.local/bin:$PATH"
declare -A PORT=([ingest]=7101 [persons]=7102 [tracker]=7103 [faces]=7104 [embed]=7105 [match]=7106 [runner]=7100)
SP=$ROOT/services/persons/.venv/lib/python3.12/site-packages/nvidia
CUDA_LD=$SP/cuda_runtime/lib:$SP/cublas/lib:$SP/cudnn/lib:$SP/cufft/lib:$SP/curand/lib:$SP/nvjitlink/lib:$SP/cuda_nvrtc/lib:/usr/lib/wsl/lib
while true; do
  for svc in "${!PORT[@]}"; do
    p=${PORT[$svc]}
    if ! curl -s -m 5 "http://127.0.0.1:$p/health" -o /dev/null; then
      echo "$(date '+%F %T') $svc (:$p) DOWN — restarting" >>"$LOG"
      env_args=(PORT="$p")
      [ "$svc" = persons ] && env_args+=(HECO_DEVICE=CUDA PERSONS_MODEL=yolox_s.onnx LD_LIBRARY_PATH="$CUDA_LD")
      [ "$svc" = runner ] && env_args+=(PLANNER_URL=http://localhost:8787 \
          HECO_INGEST_URL=http://localhost:7101 HECO_PERSONS_URL=http://localhost:7102 \
          HECO_TRACKER_URL=http://localhost:7103 HECO_FACES_URL=http://localhost:7104 \
          HECO_EMBED_URL=http://localhost:7105 HECO_MATCH_URL=http://localhost:7106)
      ( cd "$ROOT/services/$svc" && env "${env_args[@]}" nohup .venv/bin/python -m uvicorn app.main:app \
          --host 127.0.0.1 --port "$p" >>"$ROOT/.native-logs/$svc.log" 2>&1 & )
    fi
  done
  sleep 15
done
