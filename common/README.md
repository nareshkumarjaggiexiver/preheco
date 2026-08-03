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
  tests never touch the network.

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
constructor arguments (`retries`, `backoff_s`, `batch_size`) so the runner can
expose them as env without this library guessing names.
