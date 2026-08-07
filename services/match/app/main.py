"""FastAPI app for the match service (port 7106).

Stage 6 of the pipeline plus the staff whitelist and the operator-correction
surface the runner drives from the feedback loop.

Endpoints:
    GET  /health -> {ok, model, version, threshold, canonPx, template policy}
    POST /reset  {runId} -> {ok, runId}
    POST /match  {runId, embedding, quality?, siteId?, appearance?}
        -> {personKey, isNew, cosine, galleryN, subCanon, isStaff, staffId,
            templateN, templateAdded, appearanceSim, appearanceVetoed, templateId,
            nearMiss: {key, cosine, appearanceSim, basis} | null}
    POST /staff/enrol {siteId, staffId, samples:[{embedding, quality?, subCanon?}]}
        -> {staffId, sampleCount}
    POST /staff/purge {siteId, staffIds[]}    -> {siteId, removed}    (erasure)
    POST /merge  {runId, keep, drop, onlyIfSingleton?}
        -> {merged, galleryN}                                         (duplicate)
    POST /split  {runId, a, b}   -> {ok, galleryN}   (false-match / co-presence)
    POST /mark-staff {runId, personKey, siteId, staffId?}
        -> {moved, galleryN, staffKey}                                (mark-staff)
    POST /count/manual {runId, note?}
        -> {personKey, galleryN, manual:true}                         (missed)
    POST /gallery/sweep {maxAgeS?}            -> {swept:[runId]}      (retention)

Staff are checked FIRST (CONTRACTS.md v1): a staff hit is tagged
``isStaff=true`` and excluded from the guest unique count, but the track that
carries it stays visible and tracked upstream.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from . import config, gallery, staff
from .store import close_all_stores


def _env_s(name: str, default: float) -> float:
    """Read a seconds knob; empty means unset (this service stays free of
    heco_common on purpose, so this mirrors its env_float semantics locally —
    compose renders ``${VAR-}`` as an empty string, and an empty string once
    took the whole ingest service down by parsing as an error)."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)

#: 0.10.0 (2026-08-06): a pair under cannot_link no longer earns a near-miss
#: rider — the constraint is now the gallery's single record of "known
#: different" and governs merge refusal, the overlap banner and this one.  No
#: field changed shape; some riders simply stop being emitted.
VERSION = "0.11.0"

#: Default age after which an unreferenced gallery file is sweepable (24 h).
#: Long enough that a same-day re-run of a crashed event still has its data,
#: short enough that guests' embeddings do not outlive the event by a season.
DEFAULT_SWEEP_MAX_AGE_S = 24 * 3600.0

#: How often the background sweep runs.  An hour is frequent enough that a
#: gallery never outlives its retention window by more than ~4%, and rare
#: enough to be free.
DEFAULT_SWEEP_INTERVAL_S = 3600.0


def retention_s() -> float:
    """How long a run's gallery outlives its run (env HECO_GALLERY_RETENTION_S).

    Galleries used to be deleted the moment a run settled.  That destroyed the
    evidence behind an invoice figure at the exact moment it became one — a
    venue disputing a count the next morning could be shown nothing — so the
    runner no longer deletes at settle, and THIS window is the retention
    policy for every gallery: settled, stalled or crashed, all on one clock.
    24 h covers the morning-after dispute; it is deliberately not a season,
    because each file holds real guests' face embeddings.  Staff stores are
    never swept — they persist by design, under consent, until erased.
    """
    return _env_s("HECO_GALLERY_RETENTION_S", DEFAULT_SWEEP_MAX_AGE_S)


async def _sweep_forever() -> None:
    """Age out old galleries on a timer; the retention control, automated.

    /gallery/sweep existed but nothing called it — retention that requires a
    human to remember a curl is not a policy.  One pass at startup (catching
    files that aged while the service was down), then hourly.  Errors are
    logged and the loop continues: a failed sweep must not take the matcher
    down, and the next tick retries.
    """
    log = logging.getLogger("match")
    while True:
        try:
            swept = gallery.sweep(config.data_dir(), retention_s())
            if swept:
                log.info("retention sweep removed %d gallery file(s): %s", len(swept), swept)
        except Exception:  # noqa: BLE001 — the sweep must outlive one bad pass
            log.exception("retention sweep failed; will retry next interval")
        await asyncio.sleep(_env_s("HECO_GALLERY_SWEEP_INTERVAL_S", DEFAULT_SWEEP_INTERVAL_S))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Run the retention sweep for the process lifetime; release stores on exit."""
    sweeper = asyncio.create_task(_sweep_forever())
    yield
    sweeper.cancel()
    close_all_stores()


app = FastAPI(title="heco-match", version=VERSION, lifespan=_lifespan)


class ResetRequest(BaseModel):
    """Body of POST /reset — which run's gallery to wipe."""

    runId: str


class MatchRequest(BaseModel):
    """Body of POST /match — one embedding to resolve against the run gallery.

    ``siteId`` opts a run into the staff whitelist: when present the staff
    store for that site is checked before the guest gallery.

    ``appearance`` is the sighting's optional torso-appearance descriptor:
    exactly 48 floats (the wire contract's L1-normalised 12×4 Hue×Saturation
    histogram).  Omitted/null means the runner could not measure one (no
    person box, tiny crop, old footage) — which disables every appearance
    behaviour for this call rather than acting as a zero histogram.
    """

    runId: str
    embedding: list[float] = Field(min_length=8)
    quality: float | None = None  # face box width in px; <canon is tagged sub-canon
    siteId: str | None = None
    appearance: list[float] | None = None
    # Identities the CALLER has proven this face is not.  The runner sends it
    # when two different bodies in one frame both resolved to the same key —
    # one body cannot be in two places, so the weaker sighting is re-asked
    # with that key off the table.  Empty/absent changes nothing.
    excludeKeys: list[str] | None = None

    @field_validator("appearance")
    @classmethod
    def _appearance_is_48_floats(cls, v: list[float] | None) -> list[float] | None:
        """Reject any present-but-wrong-length descriptor with a readable 422.

        A truncated or padded histogram is a wire bug in the caller, and
        silently comparing it would raise deep inside the store (or worse,
        veto on garbage); the 422 names the contract instead.
        """
        if v is not None and len(v) != 48:
            raise ValueError(
                "appearance must be exactly 48 floats"
                f" (12 hue x 4 saturation bins, L1-normalised); got {len(v)}"
            )
        return v


class EnrolSample(BaseModel):
    """One enrolment sample: a face embedding and its capture quality."""

    embedding: list[float] = Field(min_length=8)
    quality: float | None = None
    subCanon: bool = False


class EnrolRequest(BaseModel):
    """Body of POST /staff/enrol — a staff member's best face samples."""

    siteId: str
    staffId: str
    samples: list[EnrolSample] = Field(min_length=1)


class MergeRequest(BaseModel):
    """Body of POST /merge — fold ``drop`` into ``keep`` (duplicate correction).

    ``onlyIfSingleton`` marks a merge whose caller is a MACHINE (the runner's
    track heal), not an operator.  A machine's evidence is weaker: it saw one
    track match two keys and inferred a junk mint, whereas an operator looked
    at two faces.  When true the merge is refused unless ``drop`` still holds
    exactly one template — a key that has accumulated more views since the
    mint has been independently re-sighted and is no longer safely foldable by
    heuristic.  See :func:`app.gallery.merge`.
    """

    runId: str
    keep: str
    drop: str
    onlyIfSingleton: bool = False


class SplitRequest(BaseModel):
    """Body of POST /split — record a do-not-merge pair (false-match correction)."""

    runId: str
    a: str
    b: str


class ForgetTemplateRequest(BaseModel):
    """Body of POST /template/forget — retract one wrongly-enrolled template."""

    runId: str
    templateId: int


class MarkStaffRequest(BaseModel):
    """Body of POST /mark-staff — move a guest person to the staff store.

    ``siteId`` is optional on the wire so a caller that omits it gets a
    readable 400 rather than a validation dump, but it is REQUIRED in
    practice: without a site there is no staff store to move the templates
    into, and the endpoint refuses (see :func:`mark_staff`).
    """

    runId: str
    personKey: str
    siteId: str | None = None
    staffId: str | None = None


class ManualCountRequest(BaseModel):
    """Body of POST /count/manual — one person the operator saw uncounted."""

    runId: str
    note: str | None = None


class SweepRequest(BaseModel):
    """Body of POST /gallery/sweep — delete gallery files older than maxAgeS."""

    maxAgeS: float | None = Field(default=None, ge=0)


@app.get("/health")
def health() -> dict:
    """Liveness + identity: no ML model here, the 'model' is the gallery policy."""
    return {
        "ok": True,
        "model": "cosine-gallery-sqlite",
        "version": VERSION,
        "threshold": config.threshold(),
        "canonPx": config.canon_px(),
        # The bench must be able to read back which template policy produced a
        # count: the same run with cap 1 and cap 5 is two different numbers.
        "templatesPerPerson": config.templates_per_person(),
        "templateConfidence": config.template_confidence(),
        "templateMargin": config.template_margin(),
        "templateMaxCosine": config.template_max_cosine(),
        # ...and the appearance veto knob, for the same reason: an enrolment
        # refused at clash 0.50 and one refused at 0.30 are different policies,
        # and 0 here means the veto was off for the whole run.
        "appearanceClash": config.appearance_clash(),
        # ...and the near-miss band's floor: a bench reading nearMissMints
        # must know which band produced them (0 = the flag was off).
        "nearMissFloor": config.nearmiss_floor(),
        # ...and BOTH weak-band knobs, for the same audit reason and one more:
        # the weak (clothing) band's default bar sits only 0.033 above the
        # worst measured impostor clothing reading (0.747), so a run whose
        # suggestions are being disputed must be able to say exactly which
        # bar produced them.  weakFloor 0 = the clothing band was off.
        "nearMissWeakFloor": config.nearmiss_weak_floor(),
        "nearMissClothes": config.nearmiss_clothes(),
    }


@app.post("/reset")
def reset(body: ResetRequest) -> dict:
    """Drop the run's gallery so a run always starts with unique count 0."""
    try:
        gallery.reset(config.data_dir(), body.runId)
    except gallery.BadRunIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"ok": True, "runId": body.runId}


@app.post("/match")
def match(body: MatchRequest) -> dict:
    """Resolve one embedding: staff first, then guest gallery (insert if new)."""
    data_dir = config.data_dir()
    threshold = config.threshold()
    canon_px = config.canon_px()
    sub_canon = body.quality is not None and body.quality < canon_px

    # Staff FIRST — a hit is tagged and kept out of the guest count.
    if body.siteId:
        try:
            hit = staff.check(data_dir, body.siteId, body.embedding, threshold)
        except staff.BadSiteIdError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if hit is not None:
            return {
                "personKey": hit.key,
                "isNew": False,
                "cosine": hit.cosine,
                "galleryN": gallery.count(data_dir, body.runId),
                "subCanon": sub_canon,
                "isStaff": True,
                "staffId": hit.key,
                # Multi-template enrolment is a GUEST-gallery policy only: the
                # staff store's templates come from the operator-supervised
                # walk-through and must not grow from unsupervised crossings,
                # so a staff hit never enrols. templateN is not applicable.
                "templateN": None,
                "templateAdded": False,
                # Staff flows carry no appearance handling at all: staff
                # identity is operator-attested, never inferred from clothing,
                # so there is nothing to compare and nothing to veto.
                "appearanceSim": None,
                "appearanceVetoed": False,
                "templateId": None,
                # ...and no near-miss flag either: a staff hit is not a mint.
                "nearMiss": None,
                "overlap": None,
            }

    try:
        r = gallery.match(
            data_dir,
            body.runId,
            body.embedding,
            body.quality,
            threshold=threshold,
            canon_px=canon_px,
            templates_per_person=config.templates_per_person(),
            template_confidence=config.template_confidence(),
            template_margin=config.template_margin(),
            template_max_cosine=config.template_max_cosine(),
            appearance=body.appearance,
            appearance_clash=config.appearance_clash(),
            nearmiss_floor=config.nearmiss_floor(),
            nearmiss_weak_floor=config.nearmiss_weak_floor(),
            nearmiss_clothes=config.nearmiss_clothes(),
            exclude_keys=set(body.excludeKeys or ()),
        )
    except gallery.BadRunIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "personKey": r.person_key,
        "isNew": r.is_new,
        "cosine": r.cosine,
        "galleryN": r.gallery_n,
        "subCanon": r.sub_canon,
        "isStaff": False,
        "staffId": None,
        "templateN": r.template_n,
        "templateAdded": r.template_added,
        "appearanceSim": r.appearance_sim,
        "appearanceVetoed": r.appearance_vetoed,
        # Rowid of the template this call enrolled, or null when it wrote
        # nothing / minted.  Only use: hand it back to POST /template/forget
        # if the caller later PROVES the enrolment was wrong.
        "templateId": r.template_id,
        # Only ever non-null on a MINT that near-missed an existing guest —
        # a one-click-merge suggestion for the operator, never behaviour
        # (see gallery._near_miss: impostors measured face 0.377 / clothes
        # 0.503, and a second impostor pair reached clothes 0.747).  Carries
        # "basis": "face" for the [floor .. threshold) band, "clothing" for
        # the weak band under it that required the torso descriptors to
        # agree.  Consumers must read a MISSING/null basis as "face" — that
        # is the old rider shape.
        "nearMiss": r.near_miss,
        # Non-null when the template this call wrote pulled its identity
        # within the match threshold of a DIFFERENT identity — two guests the
        # gallery itself can no longer tell apart (measured emerging at
        # 0.544/0.452/0.376 across three benches, each pair one real person).
        # Same rule as nearMiss: a suggestion, never a merge.
        "overlap": r.overlap,
    }


@app.post("/staff/enrol")
def staff_enrol(body: EnrolRequest) -> dict:
    """Store a staff member's best face samples into the site staff store."""
    try:
        n = staff.enrol(
            config.data_dir(),
            body.siteId,
            body.staffId,
            [s.model_dump() for s in body.samples],
        )
    except staff.BadSiteIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"staffId": body.staffId, "sampleCount": n}


class PurgeRequest(BaseModel):
    """Erasure request relayed from the planner's staff tombstones."""

    siteId: str
    staffIds: list[str] = Field(min_length=1, max_length=200)


@app.post("/staff/purge")
def staff_purge(body: PurgeRequest) -> dict:
    """Erase the given staff members' templates from the site staff store.

    Idempotent: purging an id with no templates reports 0 removed and is
    still success — the goal is the guarantee that nothing remains.
    """
    try:
        removed = staff.purge(config.data_dir(), body.siteId, body.staffIds)
    except staff.BadSiteIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"siteId": body.siteId, "removed": removed}


@app.post("/merge")
def merge(body: MergeRequest) -> dict:
    """Duplicate correction: fold ``drop`` into ``keep`` (count −1 if applied).

    The cap is passed through because the survivor inherits both identities'
    templates and would otherwise sit above it — see :func:`gallery.merge`.
    """
    try:
        merged, n = gallery.merge(
            config.data_dir(),
            body.runId,
            body.keep,
            body.drop,
            config.templates_per_person(),
            only_if_singleton=body.onlyIfSingleton,
        )
    except gallery.BadRunIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"merged": merged, "galleryN": n}


@app.post("/split")
def split(body: SplitRequest) -> dict:
    """Record a do-not-merge constraint: these two keys are different people.

    Two callers, one meaning.  The OPERATOR posts it as a *false-match*
    correction.  The RUNNER posts it as a CO-PRESENCE assertion — every
    distinct non-staff pair of personKeys appearing in ONE frame, once per pair
    (its ``HECO_COPRESENCE_SPLIT=0`` is that source's off switch) — because two
    faces at different positions in a single frame are two people, which is
    about as certain as machine evidence gets.

    The constraint governs three things and the count is not one of them:
    ``/merge`` refuses the pair (so no heal or track-lock fold can silently
    erase a paying guest), and neither duplicate-suggestion banner —
    gallery-overlap or near-miss — is raised for it again.  ``galleryN`` comes
    back unchanged, always.
    """
    try:
        n = gallery.split(config.data_dir(), body.runId, body.a, body.b)
    except gallery.BadRunIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True, "galleryN": n}


@app.post("/template/forget")
def forget_template(body: ForgetTemplateRequest) -> dict:
    """Retract ONE enrolled template the caller has proven wrong.

    The runner's same-frame guard is the caller: when two different bodies in
    one frame both resolve to the same key, the loser's sighting was already
    enrolled into that key by the /match call that discovered it, and the
    template must come back out or it keeps capturing that person in every
    later frame where they stand alone.

    ``forgotten: false`` is a NORMAL outcome, not a failure: the row may
    already have been evicted by the post-enrolment redundancy prune, or it
    may be its key's last template, which is refused so no identity is left
    existing-but-unmatchable.  ``galleryN`` is unchanged either way — this
    removes a VIEW of somebody, never somebody.
    """
    try:
        forgotten = gallery.forget_template(
            config.data_dir(), body.runId, body.templateId
        )
    except gallery.BadRunIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {
        "ok": True,
        "forgotten": forgotten,
        "galleryN": gallery.count(config.data_dir(), body.runId),
    }


@app.post("/mark-staff")
def mark_staff(body: MarkStaffRequest) -> dict:
    """Move a guest person out of the gallery and into the staff store.

    Refuses (400) without a ``siteId``.  The templates are the ONLY record of
    who this person is: lifting them out of the gallery with nowhere to put
    them destroys them, and the person is then re-counted as a brand-new guest
    on their next crossing — the correction silently undoing itself while the
    audit trail claims it was applied.  A run with no site has no staff store,
    so the honest answer is to refuse and let the operator see it.
    """
    data_dir = config.data_dir()
    if not body.siteId:
        raise HTTPException(
            status_code=400,
            detail=(
                "mark-staff needs a siteId: without a site staff store the "
                "person's templates would be destroyed, not moved"
            ),
        )
    try:
        # Remove only after the destination is known to exist and be valid.
        staff.db_path(data_dir, body.siteId)
        templates, n = gallery.remove(data_dir, body.runId, body.personKey)
    except gallery.BadRunIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except staff.BadSiteIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    staff_key: str | None = None
    if templates:
        staff_key, _ = staff.absorb(data_dir, body.siteId, body.staffId, templates)
    return {"moved": len(templates), "galleryN": n, "staffKey": staff_key}


@app.post("/count/manual")
def count_manual(body: ManualCountRequest) -> dict:
    """Add one operator-attested person to the run's unique count (*missed*).

    The only lever that moves the count UP.  Under-counting is this pipeline's
    dominant failure mode (open-set 1:N at a 1:1 verification threshold), and
    before this the operator could watch an uncounted guest walk through and
    do nothing about it.  The person is stored with an ``m`` key and no
    embedding, so the report can always say how many of the unique total a
    human added by hand.
    """
    try:
        key, n = gallery.add_manual(config.data_dir(), body.runId, body.note)
    except gallery.BadRunIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"personKey": key, "galleryN": n, "manual": True}


@app.post("/gallery/sweep")
def gallery_sweep(body: SweepRequest) -> dict:
    """Delete gallery files older than ``maxAgeS`` (default 24 h); list them.

    The documented cleanup entry point for galleries whose run died without
    releasing them.  Each file holds real guests' face embeddings, so this is
    a retention control, not housekeeping.  Staff stores are never touched —
    they are meant to persist.
    """
    max_age = DEFAULT_SWEEP_MAX_AGE_S if body.maxAgeS is None else body.maxAgeS
    return {"swept": gallery.sweep(config.data_dir(), max_age), "maxAgeS": max_age}
