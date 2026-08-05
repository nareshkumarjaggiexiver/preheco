"""The accuracy evaluation harness — the thing that says whether a change helped.

WHY THIS PACKAGE EXISTS. The product's promise is a unique count, and that
count is an INVOICE FIGURE. Until now the repo had no way to falsify a tuning
change: every "tune on pilot footage" caveat in the accuracy R&D report
(docs/planning/pipeline/accuracy-and-tuning.md, §4 step 5) was unfalsifiable,
and the only number anybody actually watched was frames per second.

That is not a theoretical risk. Setting ``INGEST_MAX_WIDTH=1920`` on the real
server took the loop from 1.59 to 7.89 fps — a five-fold "win" on the watched
number — while the faces arriving at the gate shrank from ~129 px to ~48 px
mean width (IED 46 px -> 23 px), the count of faces passing the 56 px quality
floor went 9 -> ZERO, embed and match never ran, and the run counted NOBODY.
Nothing in the repo caught it.

So this harness is built around one rule: **fps is reported, never scored.**
The headline is the signed unique-count error against a human count of record,
and beside it the whole funnel — frames, person boxes, faces detected, faces
passing the gate, embeds, matches, unique — so a collapse anywhere in the
chain is a named, loud check rather than a number a human has to spot.

Modules:

* :mod:`eval.manifest` — the checked-in ground-truth clip list.
* :mod:`eval.metrics` — PURE scoring: the funnel, the signed error, the
  collapse detectors, the "is this run even scorable" filter.
* :mod:`eval.compare` — PURE A/B verdict; refuses to call a change an
  improvement when accuracy went backwards, however much fps improved.
* :mod:`eval.report` — PURE text rendering of both.
* :mod:`eval.clients` — thin HTTP readers for the runner and the planner.
* :mod:`eval.run` — the CLI that drives real clips through the real pipeline.
* :mod:`eval.compare` also exposes ``python -m eval.compare``.
"""

__version__ = "1.0.0"
