# match — the count-once decision + staff whitelist (port 7106)

**What.** Stage 6 of the pipeline: given a face embedding, decide whether this
person was already seen in this run (re-sighting — unique count unchanged) or
is new (unique count +1). Two SQLite files back it, both the same brute-force
cosine `VectorStore` (`app/store.py`):

- **Guest gallery** — one file per run (`data/gallery-<runId>.db`), wiped at run
  start so the unique count begins at zero.
- **Staff whitelist** — one persistent file per site
  (`data/staff-<siteId>.db`): venue/catering staff enrolled once and checked
  **first**, so a staff hit is tagged and kept out of the guest count.

Matching is an exact brute-force cosine scan with numpy — at POC scale
(hundreds of guests, a roster of a few thousand) this costs well under a
millisecond. No model weights: this service is pure geometry + storage.

## The store, and how it scales

`VectorStore` stores each template as a `float32` BLOB and answers the nearest
neighbour with one matrix-vector multiply. Its docstring names the drop-in
growth path: **sqlite-vec**'s `vec0` virtual table stores the same BLOB and
answers `k`-NN with an ANN index, so past ~5k vectors the migration is "shadow
the `vec` column into a `vec0` table, swap the scan for a `MATCH` query" — the
row schema, the encoding and every caller stay put. Do that only on profiling
evidence (project hard rule: measure first).

## API

| method | path | body | returns |
| --- | --- | --- | --- |
| GET | `/health` | — | `{ok, model, version, threshold, canonPx}` |
| POST | `/reset` | `{runId}` | `{ok, runId}` — wipes the run's gallery |
| POST | `/match` | `{runId, embedding, quality?, siteId?}` | `{personKey, isNew, cosine, galleryN, subCanon, isStaff, staffId}` |
| POST | `/staff/enrol` | `{siteId, staffId, samples:[{embedding, quality?, subCanon?}]}` | `{staffId, sampleCount}` |
| POST | `/merge` | `{runId, keep, drop}` | `{merged, galleryN}` — *duplicate* correction |
| POST | `/split` | `{runId, a, b}` | `{ok, galleryN}` — *false-match* correction |
| POST | `/mark-staff` | `{runId, personKey, siteId, staffId?}` | `{moved, galleryN, staffKey}` — **400 without a siteId** |
| POST | `/count/manual` | `{runId, note?}` | `{personKey, galleryN, manual:true}` — *missed* correction |
| POST | `/staff/purge` | `{siteId, staffIds[]}` | `{siteId, removed}` — consent erasure |
| POST | `/gallery/sweep` | `{maxAgeS?}` | `{swept:[runId], maxAgeS}` — retention backstop |

`quality` is the face box width in pixels. `cosine` is the best similarity
against the pre-existing gallery (`null` on the very first face of a run).
`galleryN` is the distinct-guest count after the call.

- **Staff first.** With a `siteId`, `/match` checks the site staff store before
  the guest gallery. A hit returns `isStaff:true` with the `staffId` and leaves
  the guest count untouched — the runner counts it as a *staffCrossing*. Staff
  stay tracked upstream; they are only excluded from the guest **count**.
- **Corrections** back the runner's feedback loop. `/merge` folds `drop` into
  `keep` (count −1) unless a `/split` cannot-link constraint refuses it;
  `/split` records that constraint ("raise the pair's internal distance, no
  auto-merge later"); `/mark-staff` lifts a guest out of the gallery (count −1)
  and re-homes their templates in the staff store (under `staffId`, or a fresh
  `anon#####` key). `/mark-staff` **refuses without a siteId**: the templates
  are the only record of who that person is, so removing them with nowhere to
  put them destroys them and the person is counted as a brand-new guest at
  their next crossing — the correction silently undoing itself.
- **`/count/manual` is the only lever that moves the count UP.** Under-counting
  is the dominant failure mode (open-set 1:N at a 1:1 verification threshold),
  so a *missed* correction mints an `m#####` person with NO embedding: counted,
  never matchable, and permanently distinguishable from a detected `p#####`.
- **Re-enrolment supersedes.** `/staff/enrol` replaces a member's prior samples
  with the new best-N, so `sampleCount` is exact rather than growing across
  walk-throughs.
- **Gallery files have a lifecycle.** `/reset` deletes a run's gallery (the
  runner calls it at start AND at end of run); `/gallery/sweep` deletes files
  older than `maxAgeS` for runs that died without releasing. Each file holds
  real guests' face embeddings, so this is a retention control. Staff stores
  are never swept — they persist by design.
- **Open once, scan in memory.** Every store is opened once per process and
  its `(n, dim)` float32 matrix stays resident, updated write-through on each
  mutation (SQLite in WAL mode remains the durable record). A full match call
  (staff check + gallery match) at a 500-person gallery went from ~1414 µs to
  ~86 µs for a re-sighting and ~4340 µs to ~144 µs for a new person — and, more
  importantly, stopped growing with gallery size.

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

Tests cover the store (add/search/threshold, monotonic keys, merge/split/remove,
persistence), the staff whitelist (enrol → staff-first match, exclusion from the
guest count, mark-staff), and the original match/reset/sub-canon behaviour.

## Tune

| env | default | meaning |
| --- | --- | --- |
| `HECO_MATCH_THRESHOLD` | `0.363` | Cosine "same person" threshold (staff use the same). The SFace paper's 1:1 operating point — expect to tune per bench session; production needs threshold = f(gallery size) per `docs/planning/04-identity-pipeline.md`. |
| `HECO_MATCH_CANON_PX` | `80` | Faces narrower than this are tagged `subCanon` (matched normally). POC geometry expects 64–85 px faces. |
| `HECO_MATCH_DATA_DIR` | `data` | Where the per-run gallery and per-site staff databases live. |

## Known POC limits

- One template per new guest (the first sighting); merges and enrolment are the
  only ways a key grows to multi-template. Diverse multi-crop enrolment for
  guests is the planned accuracy win — add on duplicate-rate evidence.
- Brute-force scan, not an index (see sqlite-vec above).
- Threshold is fixed per run; gallery-size-aware thresholding is a product
  requirement, not a POC one.
