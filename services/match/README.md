# match — the count-once decision (port 7106)

**What.** Stage 6 of the pipeline: given a face embedding, decide whether this
person was already seen in this run (re-sighting — unique count unchanged) or
is new (unique count +1). The gallery is a SQLite file per run
(`data/gallery-<runId>.db`); matching is an exact brute-force cosine scan with
numpy — at POC scale (hundreds of people) this costs well under a millisecond,
so no vector index until profiling says otherwise. No model weights: this
service is pure geometry + storage.

## API

| method | path | body | returns |
| --- | --- | --- | --- |
| GET | `/health` | — | `{ok, model, version, threshold, canonPx}` |
| POST | `/reset` | `{runId}` | `{ok, runId}` — wipes the run's gallery |
| POST | `/match` | `{runId, embedding, quality}` | `{personKey, isNew, cosine, galleryN, subCanon}` |

`quality` is the face box width in pixels. `cosine` is the best similarity
against the pre-existing gallery (`null` on the very first face of a run).
`galleryN` is the distinct-person count after the call.

## Run

```sh
make venv     # python3.12 -m venv .venv && pip install -r requirements.txt
make run      # uvicorn on 0.0.0.0:7106
```

## Test

```sh
make test     # pytest — synthetic clustered-gaussian identities, no network
make lint     # ruff
```

## Tune

| env | default | meaning |
| --- | --- | --- |
| `HECO_MATCH_THRESHOLD` | `0.363` | Cosine "same person" threshold. The SFace paper's operating point — a 1:1 verification number. The POC runs open-set 1:N against a growing gallery, so expect to tune this per bench session; production needs threshold = f(gallery size) per `docs/planning/04-identity-pipeline.md`. |
| `HECO_MATCH_CANON_PX` | `80` | Faces narrower than this are tagged `subCanon` (matched normally). POC geometry expects 64–85 px faces — below the production 80/100 px canon, accepted deliberately. |
| `HECO_MATCH_DATA_DIR` | `data` | Where per-run gallery databases live. |

## Known POC limits

- One template per person (the first sighting). Multi-template (3–5 diverse
  crops per person) is the planned accuracy win; add when the bench shows
  duplicate-rate evidence.
- Threshold is fixed per run; gallery-size-aware thresholding is a product
  requirement, not a POC one.
