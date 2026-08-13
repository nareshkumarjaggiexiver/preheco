# `counting/` — the decisions, separated from the machinery

This package holds the part of the pipeline that decides **how many people
walked through**: the quality gate, the exclusion zones, face-to-body and
face-to-track association, the heal and lock folds, the co-presence
assertion, and the ledger that owns the unique count.

It holds none of the machinery around those decisions — no HTTP client, no
FastAPI app, no camera, no planner. Those belong to whichever process is
hosting the decisions, and `tests/test_boundaries.py` enforces it: importing
this package in a fresh interpreter must pull in no transport and no
framework, and no module here may import its host.

It does depend on `heco-common`, for one module: `heco_common.geometry`, the
pure stdlib box and polygon maths that `faces/` and the runner already share.
Duplicating it would let the exclusion-zone polygon test drift away from the
one that draws the polygons. The direction is what matters — counting depends
on common, never the reverse — so counting stays off the **match** service's
import path and the count keeps exactly one writer.

## Why it is not in `common/`

`heco_common` is installed into all seven services (`-e ../../common` in every
`requirements.txt`) and copied into every image. Putting the counting
decisions there would put the mint ledger, the heal and the co-presence
assertion on the import path of the **match** service — the process that owns
the gallery.

The architecture rule is that the count has exactly one writer. A separate
distribution makes that structural rather than aspirational: the match image
never installs `heco-counting`, so it cannot accidentally grow a second
opinion about the number.

## Who hosts it

Today: the runner (`services/runner`), unchanged in behaviour.

Next: a per-camera worker holds the per-camera pieces in-process, and a
per-gate fusion process holds the per-gate pieces as the single writer. That
split runs along the scope annotations below, which is why they are recorded
here rather than discovered later.

| scope | what lives there | why |
| --- | --- | --- |
| **per-camera** | gate, zones, association, track memory, folds, co-presence | Every one of these reasons about *this camera's* geometry and *this camera's* tracker. Fold authority in particular must never cross cameras: cross-camera association will swap people at a crowded gate, and a fold amplifies that into a silent under-count. |
| **per-gate** | the identity ledger — unique, mints, retirements, the template census | One gallery per gate, one writer. |
| **per-venue** | staff crossings | A waiter is a waiter at every gate; debouncing them per-camera would count them twice. |

## The two rules the types encode

1. **No `runId` in any port signature.** Gallery identity is bound when the
   port is constructed, not passed per call — so a caller cannot address
   another run's gallery by accident.
2. **`CameraCounter` takes a ledger it does not own.** Several cameras share
   one ledger; none of them may create one. That is the single-writer rule
   expressed as a constructor argument.

## Guard rail

Nothing in here may change behaviour without showing up in
`eval/golden.py`. The extraction that created this package is held to a
byte-identical decision diff on real footage — see `eval/README.md`.
