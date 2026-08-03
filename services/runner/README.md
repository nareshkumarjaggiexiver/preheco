# runner — the pipeline conductor (port 7100)

**What.** Owns the run lifecycle: creates the planner-side run record
(`heco_common.planner.PlannerClient`), resets the match gallery and the
tracker state for the run, opens the source on ingest, then drives the loop —
`frame → persons → tracker → faces (within tracked boxes) → quality gate →
embed → match → unique count` — timing every stage. Aggregates
(count/min/mean/max) and sampled raw rows are POSTed to the site-planner every
2 s; the run ends with a `PUT … {status: ended, notes}` carrying the unique
count, frame count, and sub-canon share. Pure HTTP orchestration: no OpenCV,
no models, frames stay opaque base64.

The quality gate lives here (not in a separate service): faces below **56 px**
width never reach the embedder; **56–79 px** pass but are flagged *sub-canon*
(POC geometry: 2.8 mm camera at 2.0 m, faces ~64–85 px — see CONTRACTS.md).

## API

| method | path | body | returns |
| --- | --- | --- | --- |
| GET | `/health` | — | `{ok, model, version}` |
| POST | `/runs` | `{eventId, placementId?, source:{url\|path}, plannerUrl?, label?}` | `{runId, state}` |
| GET | `/runs/{runId}` | — | live local status (frames, unique, subCanonShare, state, error) |
| POST | `/runs/{runId}/stop` | — | ends the run after the current frame (RTSP sources never end alone) |

End-of-source: ingest serves the LATEST frame with an increasing `seq` and
never signals EOF explicitly — the runner treats a **stalled seq** (no new
frame for `HECO_SOURCE_STALL_S`) as the end. Stub-style explicit ends
(`ended: true`, missing `imageB64`, HTTP 204/404/410) also end the run.

## Run

```sh
make venv     # python3.12 -m venv .venv && pip install -r requirements.txt
make run      # uvicorn on 0.0.0.0:7100
```

## Test

```sh
make test     # pytest — the loop against fake stage services (httpx.MockTransport)
make lint     # ruff
```

Tests cover orchestration order, aggregation maths, sample-batch capping,
end-of-source and stop handling, and stage-failure settlement. No network.

## Tune

| env | default | meaning |
| --- | --- | --- |
| `HECO_INGEST_URL` … `HECO_MATCH_URL` | `http://<service>:<port>` | Stage service URLs (compose DNS names; point at localhost off-compose) |
| `PLANNER_URL` | `http://host.docker.internal:8787` | The site-planner app |
| `HECO_QUALITY_MIN_PX` | `56` | Quality-gate floor: narrower faces never embed |
| `HECO_QUALITY_CANON_PX` | `80` | Below this, matched faces are flagged sub-canon |
| `HECO_FLUSH_INTERVAL_S` | `2.0` | Planner stats/samples cadence |
| `HECO_REQUEST_TIMEOUT_S` | `30` | Per-stage HTTP timeout |
| `HECO_SOURCE_POLL_S` | `0.02` | Poll interval while ingest's `seq` is unchanged |
| `HECO_SOURCE_STALL_S` | `5.0` | Stalled-seq duration treated as end-of-source |

## Notes

- The planner write side is `heco_common.planner.PlannerClient` (shared
  `common/` package, installed editable); the runner adapts it onto its own
  httpx client so tests can fake the planner with `httpx.MockTransport`.
- The loop polls ingest as fast as the slowest stage allows — no pacing at POC
  scale. Add pacing only with profiling evidence.
- No compiled rewrite: the runner is I/O-bound HTTP glue; OpenCV/onnxruntime
  in the stage services are already native code. Revisit only if profiling
  shows the conductor itself is the bottleneck.
