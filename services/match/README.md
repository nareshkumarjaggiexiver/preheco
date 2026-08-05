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
| POST | `/match` | `{runId, embedding, quality?, siteId?}` | `{personKey, isNew, cosine, galleryN, subCanon, isStaff, staffId, templateN, templateAdded}` |
| POST | `/staff/enrol` | `{siteId, staffId, samples:[{embedding, quality?, subCanon?}]}` | `{staffId, sampleCount}` |
| POST | `/merge` | `{runId, keep, drop}` | `{merged, galleryN}` — *duplicate* correction |
| POST | `/split` | `{runId, a, b}` | `{ok, galleryN}` — *false-match* correction |
| POST | `/mark-staff` | `{runId, personKey, siteId, staffId?}` | `{moved, galleryN, staffKey}` — **400 without a siteId** |
| POST | `/count/manual` | `{runId, note?}` | `{personKey, galleryN, manual:true}` — *missed* correction |
| POST | `/staff/purge` | `{siteId, staffIds[]}` | `{siteId, removed}` — consent erasure |
| POST | `/gallery/sweep` | `{maxAgeS?}` | `{swept:[runId], maxAgeS}` — retention backstop |

`quality` is the face box width in pixels. `cosine` is the best similarity
against the pre-existing gallery (`null` on the very first face of a run).
`galleryN` is the distinct-guest count after the call. `templateN` is how many
views the matched identity now holds (`null` for a staff hit); `templateAdded`
says whether this sighting became one of them.

- **Multiple templates per guest (M1).** A guest is represented by up to five
  views, not by whichever frame happened to be first. See below.
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

## Multiple templates per guest (M1)

**The problem it fixes.** A guest used to be stored once, on their first
sighting, and every later face was compared against that single arbitrary view.
The corridor bench — ground truth **one man walking** — produced **three**
gallery identities for him. Their pairwise cosines were **0.347 / 0.307 /
0.296** against the 0.363 threshold: every pair a near miss, the closest by
0.016. Those views were not unrecognisable (before the quality gate was armed
the same measurement was mean 0.172, spread −0.055 to 0.337); they were simply
never compared with anything except one frame.

**What changed.** On a match, the sighting may be kept as an *additional* view
of that identity, capped at `HECO_MATCH_TEMPLATES_PER_PERSON` (5) and evicting
the lowest capture quality — so a guest accumulates their **best** views, not
their most recent. Matching needs no new machinery: `VectorStore.search` is an
argmax over rows, and max-over-rows equals max-over-per-identity-maxima, so the
moment several rows share a key the returned cosine already *is* that
identity's best template score (pinned by `test_search_is_per_identity_best`).

**The threshold did not move, and must not.** A sighting still has to clear
0.363 against an already-stored template to count as a re-sighting at all. What
widens is the accept *region* — that is the whole point: it is how a profile
view gets attached to the frontal view that minted the key. Lowering the
threshold is M2 and is blocked on impostor data we do not have.

**Two different bars, and confusing them will mislead you.** 0.363 decides
*matched or not*. **0.413** (0.363 + `TEMPLATE_CONFIDENCE`) decides *learned
from or not*. Between them lies a **dead band**: the sighting is counted as the
same guest but is never kept, so the identity does not grow towards it and the
chain does not extend. Measured on a steady pose sweep — adjacent cosine 0.420
→ 1 identity (4 enrolled); 0.410 → 3 identities (1 enrolled); 0.380 → 3
identities (0 enrolled). In practice the chain only extends on sightings
roughly 65° of pose apart or closer. If a bench shows fragmentation *and* most
matches landing in 0.363–0.413, the dead band is the reason and
`HECO_MATCH_TEMPLATE_CONFIDENCE` is the knob — not the threshold.

**When a matched sighting is kept** (all four must hold — `gallery._should_enrol`):

| gate | default | why |
| --- | --- | --- |
| clears the threshold by `TEMPLATE_CONFIDENCE` | +0.05 → 0.413 | A bare-minimum match is the least certain evidence we have; promoting it would let an identity annex the region around a point we are unsure of, and the error would compound template by template. The drift brake. |
| beats the nearest **rival** identity by `TEMPLATE_MARGIN` | 0.05 | A view sitting between two people would become a *bridge*, and the next probe near it merges two paying guests. Ambiguity is resolved by not learning. |
| is below `TEMPLATE_MAX_COSINE` | 0.90 | Above this it is a near-duplicate: no new pose coverage, but it spends a capped slot and evicts a genuinely different view, leaving the identity **narrower** than before. Most frames of a walking guest land here, which is also what keeps the write path quiet. |
| beats the worst held view on quality, once at cap | — | Otherwise the insert is undone by the eviction on the next line. A sighting with no recorded face width counts as the worst possible. |

**The honest boundary.** M1 collapses a crossing only when the crossing supplied
the *intermediate* poses that chain its extremes together. Fed only tonight's
three anchor views and nothing between them, the gallery still reports three
people — and is right to: nothing can join two vectors at cosine 0.307 while
the threshold is 0.363 except a lower threshold.
`test_three_anchor_views_alone_still_do_not_collapse` keeps that boundary
executable so nobody later "fixes" it quietly.

**Cost.** A 500-guest gallery now holds up to ~2 500 templates. A re-sighting
costs ~104 µs and a new person ~133 µs (was ~86 / ~144 µs at 500 templates) —
the scan grew with the row count, everything else stayed flat. `count_for` was
moved onto the `idx_vectors_key` index because the verdict now reports
`templateN` on every call and the old key-list scan cost 27 µs at that size.

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
guest count, mark-staff), the original match/reset/sub-canon behaviour, and
`tests/test_templates.py` — multi-template, built on tonight's exact corridor
geometry (Cholesky of the measured Gram matrix, so the cosines are the
measurement rather than an approximation of it).

## Tune

| env | default | meaning |
| --- | --- | --- |
| `HECO_MATCH_THRESHOLD` | `0.363` | Cosine "same person" threshold (staff use the same). The SFace paper's 1:1 operating point — expect to tune per bench session; production needs threshold = f(gallery size) per `docs/planning/04-identity-pipeline.md`. |
| `HECO_MATCH_CANON_PX` | `80` | Faces narrower than this are tagged `subCanon` (matched normally). POC geometry expects 64–85 px faces. |
| `HECO_MATCH_DATA_DIR` | `data` | Where the per-run gallery and per-site staff databases live. |
| `HECO_MATCH_TEMPLATES_PER_PERSON` | `5` | Max views one guest may hold. Same N as the staff walk-through, and roughly the distinct views a crossing yields (frontal, two three-quarters, two profiles). **Set to `1` to restore the pre-M1 gallery exactly** — the off switch, and the way to A/B a bench run. |
| `HECO_MATCH_TEMPLATE_CONFIDENCE` | `0.05` | How far above the threshold a match must land to be kept as a template. |
| `HECO_MATCH_TEMPLATE_MARGIN` | `0.05` | How far ahead of the nearest rival identity it must land. |
| `HECO_MATCH_TEMPLATE_MAX_COSINE` | `0.90` | Near-duplicate ceiling: above this the view adds no coverage. |

Staff enrolment is unaffected by all four: staff templates come only from the
operator-supervised walk-through, never from a crossing.

## Known POC limits

- **Multi-template widens the accept region, and that cuts both ways.** It is
  the cure for splitting one guest across several keys; the matching risk is
  the opposite error, two guests chained into one, which is invisible in the
  count. The confidence + rival-margin gates and the cap of 5 bound it, and
  `test_two_near_miss_people_do_not_merge` holds a 0.30-apart pair apart, but
  the real number needs impostor pairs from the venue.
- **The gates' defaults are reasoned, not measured.** 0.05 / 0.05 / 0.90 come
  from the geometry above, not from a labelled bench set; they are env-tunable
  precisely because the first impostor data should move them.
- Brute-force scan, not an index (see sqlite-vec above).
- Threshold is fixed per run; gallery-size-aware thresholding is a product
  requirement, not a POC one.
