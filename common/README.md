# heco-common

**What.** The shared library every pipeline service installs *editable* into
its own venv (`-e ../../common` in each service's `requirements.txt`). It owns
the wire contract as code:

- `heco_common.schemas` — pydantic v2 models for **every** inter-service
  message in [CONTRACTS.md](../CONTRACTS.md): boxes, frames, faces, tracks,
  embeddings, match results, health, and the planner ingest shapes
  (runs / stats / samples, stage-name literal included).
- `heco_common.imaging` — base64 JPEG encode/decode (frames travel as base64
  JPEG in JSON at POC scale) and aspect-preserving resize.
- `heco_common.config` — typed config-from-env helpers (`env_str`, `env_int`,
  `env_float`, `env_bool`).
- `heco_common.planner` — `PlannerClient`: `create_run` / `end_run` /
  `post_stats` / `post_samples` against the site-planner write side, with
  retry + sample batching (≤ 200 per POST) and an **injectable transport** so
  tests never touch the network. The v1 debug/feedback surface rides on the
  same client, all **best-effort** (single-shot, swallow failures — a planner
  hiccup never blocks or crashes the run loop, per CONTRACTS.md):
  - `post_tap(stage, payload)` — one stage's structured output (boxes, track
    ids+ages, face quality flags, match verdicts, counters).
  - `post_frame(stage, jpeg)` — the stage's annotated JPEG, **multipart** via a
    second injectable `file_transport` (omit it and framing is a clean no-op).
  - `poll_feedback(since?)` / `resolve_feedback(id, status)` — the operator
    correction loop.
  - `report_enrolment(staffId, enrolledAt, sampleCount)` — confirm a staff
    enrolment (`PUT /api/staff/:id`); this one retries (the samples are already
    stored, so the report is worth a retry).
  - `durable=True` (opt-in; the runner turns it on): what the planner is away
    for — taps, frames, face cards, frame records, a failed run end — is kept
    in a bounded in-memory `Outbox` and delivered in order by one background
    thread instead of dropped. The first attempt is unchanged. Each post has
    an `offer_*` twin answering `accepted` / `queued` / `dropped` (a caller
    that re-sends on failure must not re-send a queued write), every durable
    write carries `X-Heco-Delivery` / `X-Heco-At`, and `outbox_stats()` says
    what was delivered, dropped and is held. CONTRACTS.md: "a planner restart
    loses nothing the runner posted".

**Run.** It is a library — nothing to run. Field names are deliberately
camelCase to mirror the JSON wire format byte-for-byte.

**Test.**

```sh
make venv   # one-off: .venv + editable install + pytest/ruff
make test   # offline unit tests
make lint   # ruff, root ruff.toml
```

**Tune.** Nothing here reads env itself except through `heco_common.config`;
services own their env names. `PlannerClient` retry/batch knobs are
constructor arguments (`retries`, `backoff_s`, `batch_size`, and for the
outbox `durable`, `outbox_max_bytes`, `outbox_stale_frame_s`,
`drain_timeout_s`) so the runner can expose them as env without this library
guessing names.
