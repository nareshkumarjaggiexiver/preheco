"""FastAPI app for the match service (port 7106).

Stage 6 of the pipeline plus the staff whitelist and the operator-correction
surface the runner drives from the feedback loop.

Endpoints:
    GET  /health -> {ok, model, version, threshold, canonPx, template policy}
    POST /reset  {runId} -> {ok, runId}
    POST /match  {runId, embedding, quality?, siteId?, appearance?,
                  attributes?, featNorm?, body?}
        -> {personKey, isNew, cosine, galleryN, subCanon, isStaff, staffId,
            templateN, templateAdded, appearanceSim, appearanceVetoed, templateId,
            nearMiss: {key, cosine, appearanceSim, basis} | null}
    POST /review/duplicates {runId, limit?}
        -> {runId, threshold, pairs:[{a, b, cosine, clothes, why}],
            considered, returned, dropped, excluded: {gender, age, stature}}
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
from typing import Literal

from fastapi import FastAPI, HTTPException
from heco_common.gate_auth import install_bearer_gate
from pydantic import BaseModel, Field, field_validator, model_validator

from . import config, gallery, staff
from .appearance import APPEARANCE_DIMS
from .store import EmbedderMismatchError, close_all_stores


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
#: 0.12.0 (2026-09-24): /match accepts a 64-float (v3) torso descriptor beside
#: the 48-float one, plus optional attributes / featNorm / body; the review
#: queue gains a per-pair `why` and an `excluded` count.  Additive only.
VERSION = "0.12.0"

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


# A wrong-embedder store is a PERMANENT conflict between the file's stamped
# identity and this process's configuration — a 409, not a 500: the runner's
# refusal convention (4xx = understood-you-no, settle it; 5xx = try again)
# must see this as settled, or every /match against a mismatched staff store
# becomes an indefinite retry storm at frame rate with the guard's
# carefully-written way-out message never reaching the wire.
@app.exception_handler(EmbedderMismatchError)
async def _embedder_mismatch(request, exc):  # noqa: ANN001, D401
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=409, content={"detail": str(exc)})
# Inbound auth (runbook step 8): armed by HECO_REQUIRE_AUTH=1, this refuses
# LAN callers without a bearer credential — an heco-auth token verified
# locally, or the legacy shared secret while it survives. /health stays
# open for the compose healthchecks. Unarmed, nothing changes.
install_bearer_gate(app)


class ResetRequest(BaseModel):
    """Body of POST /reset — which run's gallery to wipe."""

    runId: str


class FaceAttributes(BaseModel):
    """What the embed service's attribute head read off ONE face.

    ``gender`` is the reported sex, ``genderP`` the probability OF THAT
    REPORTED SEX (so a value near 0.5 is "unsure", never "female"), ``age``
    in years.  Stored per template and aggregated per identity by the review
    queue; never consulted by a verdict.
    """

    gender: Literal["M", "F"]
    genderP: float = Field(ge=0.0, le=1.0)
    age: float = Field(ge=0.0)


class PersonBox(BaseModel):
    """The sighting's containing PERSON box, in raw detector pixels.

    ``h``/``w`` the box size, ``yBottom`` the y of its bottom edge, ``frameH``
    the frame height it was measured in — enough to tell a standing body
    from a seated one and to place it on the camera's perspective line.  The
    stature estimate needs every standing box in the run, so this is logged
    on every guest call, not only when a template is written.
    """

    h: float = Field(gt=0.0)
    w: float = Field(gt=0.0)
    yBottom: float
    frameH: int = Field(gt=0)


class MatchRequest(BaseModel):
    """Body of POST /match — one embedding to resolve against the run gallery.

    ``siteId`` opts a run into the staff whitelist: when present the staff
    store for that site is checked before the guest gallery.

    ``appearance`` is the sighting's optional torso-appearance descriptor:
    exactly 48 floats (v2: 12×3 Hue×Saturation chromatic bins plus 3
    brightness bins, L1-normalised) or exactly 64 (v3: colour below the neck
    on non-skin pixels, texture, edge density — see :mod:`app.appearance`).  Omitted/null means the
    runner could not measure one (no person box, tiny crop, old footage) —
    which disables every appearance behaviour for this call rather than
    acting as a zero histogram.

    ``attributes``, ``featNorm`` and ``body`` (2026-09-24) are recorded and
    never judged here: the embed service's sex/age reading of this face,
    the raw feature's L2 norm, and the sighting's containing PERSON box.
    They feed only the duplicate review queue's ``why`` and its exclusions.
    Each is optional and ``null`` means not measured.
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
    attributes: FaceAttributes | None = None
    featNorm: float | None = None
    body: PersonBox | None = None

    @field_validator("appearance")
    @classmethod
    def _appearance_is_48_or_64_floats(cls, v: list[float] | None) -> list[float] | None:
        """Reject any present-but-wrong-length descriptor with a readable 422.

        A truncated or padded histogram is a wire bug in the caller; the 422
        names the contract instead of letting garbage into the gallery.  Both
        generations are legal on the wire — a v3 runner and a retained v2
        gallery coexist for a day — and a v2-against-v3 comparison is simply
        "not measured" downstream (:func:`app.appearance.intersection`).
        """
        if v is not None and len(v) not in APPEARANCE_DIMS:
            raise ValueError(
                "appearance must be exactly 48 floats (v2: 12 hue x 3 saturation"
                " + 3 brightness bins) or 64 floats (v3, colour + texture + edge"
                f" + reserved), L1-normalised; got {len(v)}"
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


class ReviewDuplicatesRequest(BaseModel):
    """Body of POST /review/duplicates — ask which identities a human should check."""

    runId: str
    # Bounds the queue, never the analysis: everything is examined and the
    # count beyond the cap comes back as `dropped` rather than vanishing.
    limit: int = Field(default=50, ge=1, le=500)


class ForgetTemplateRequest(BaseModel):
    """Body of POST /template/forget — retract one wrongly-enrolled template.

    ``templateId`` is the ``/match`` reply's template rowid, ``bodyId`` its
    body-sighting rowid (2026-09-24 evening); either alone is a valid ask
    and both together is the usual one.  A hit that enrolled nothing still
    logged its box, so a caller that has only a ``bodyId`` must be able to
    retract it.
    """

    runId: str
    templateId: int | None = None
    bodyId: int | None = None

    @model_validator(mode="after")
    def _names_something_to_forget(self) -> "ForgetTemplateRequest":
        """An ask that names neither row is a caller bug, said at the boundary."""
        if self.templateId is None and self.bodyId is None:
            raise ValueError("templateId or bodyId is required")
        return self


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
        # ...and the review queue's floor and its three exclusion signals: a
        # queue of 12 pairs under one policy and 500 under another is the
        # difference between a usable console and a flood, so the policy
        # must be readable next to the count it produced.  0 = that signal
        # is off.
        "reviewFloor": config.review_floor(),
        "reviewGenderMinP": config.review_gender_min_p(),
        "reviewAgeChildMax": config.review_age_child_max(),
        "reviewAgeAdultMin": config.review_age_adult_min(),
        "reviewStatureGap": config.review_stature_gap(),
        "reviewStatureMinN": config.review_stature_min_n(),
        "adultM": config.adult_height_m(),
        "reviewClothesClash": config.review_clothes_clash(),
        "reviewClothesMinN": config.review_clothes_min_n(),
        "reviewClothesSelfMin": config.review_clothes_self_min(),
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

    # Staff is checked first but must now COMPETE — it no longer shadows.
    # The old shape returned on any staff hit >= threshold without ever
    # consulting the guest gallery, and run 27ca33 measured what that costs: a
    # guest matching p00020 at 0.6476 one frame earlier and 0.5998 one frame
    # later was tagged STAFF at 0.4234 in between, and 281 of 339 danger-zone
    # verdicts were staff hits hugging the threshold. A weak staff score is
    # not evidence AGAINST a strong guest identity.
    #
    # The rule: a staff hit wins only when it is at least as strong as the
    # best guest-gallery score for the same probe. Ties go to STAFF, because
    # at equal evidence keeping a possible staff member out of the guest
    # count is the conservative direction for the invoice. The guest lookup
    # is read-only (gallery.best_cosine) — deciding needs both numbers, and
    # fetching one must not mint anybody.
    if body.siteId:
        try:
            hit = staff.check(data_dir, body.siteId, body.embedding, threshold)
        except staff.BadSiteIdError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if hit is not None:
            guest_best = gallery.best_cosine(data_dir, body.runId, body.embedding)
            if guest_best is not None and guest_best > hit.cosine:
                # The gallery knows this face better than the staff store
                # does: fall through to the ordinary guest path below, which
                # will bind to that stronger identity.
                hit = None
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
            attributes=None if body.attributes is None else body.attributes.model_dump(),
            feat_norm=body.featNorm,
            body=None if body.body is None else body.body.model_dump(),
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
        # Rowid of the body_sightings row this call logged, or null when the
        # call carried no usable body.  Same use as templateId: hand it back
        # to POST /template/forget when the sighting is proven to be a
        # different body, so its box leaves this identity's stature evidence.
        "bodyId": r.body_id,
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


@app.post("/review/duplicates")
def review_duplicates(body: ReviewDuplicatesRequest) -> dict:
    """Identity pairs worth a human glance, ranked. Decides nothing.

    Run 0f5c6d counted four people where there were three: one man at the back
    of the room, phone to his ear, was minted twice at a mutual face score of
    0.2117.  No automatic rule could have caught it — his duplicate pair
    agreed on clothing at 0.587 while two genuinely different men in the same
    run agreed at 0.538, and no threshold lives in that gap.

    So the machine stops guessing and hands over a short, ordered list.  Pairs
    the gallery already knows are different — a recorded `cannot_link`, from
    co-presence or from the operator's own false-match click — never appear.
    Clothing orders the queue, and since the v3 torso a CLEAR clash — both
    people's own reads plentiful and self-consistent, and still disagreeing —
    sets a pair aside (reply ``setAside``), where the operator can still see
    and merge it.

    Since 2026-09-24 each pair also carries ``why`` — both identities' sex
    (with confidence), median age and stature ratio, null wherever nothing
    was measured — and pairs those signals say cannot be one person are set
    aside before the cap, counted in ``excluded: {gender, age, stature}``.
    Run f0bfc5 put a man against an elderly woman at #3 of 500; the evidence
    to not ask was already in the gallery.  Setting aside writes nothing (no
    cannot_link row: that stays a human's or co-presence's word) and merges
    nothing.  Stature ``a``/``b`` are ratios against the run's own
    perspective fit; ``adultM`` (1.75 m, the North Indian adult average this
    deployment is anchored on) converts them, and ``aM``/``bM`` are that
    product ready to print.

    Read-only: no template is written, no key is merged, `galleryN` is
    untouched.  Acting on a row is the operator's existing one-click /merge.
    """
    try:
        report = gallery.review_duplicates(
            config.data_dir(),
            body.runId,
            threshold=config.threshold(),
            # Its own knob since 2026-08-13: borrowing nearmiss_weak_floor
            # meant the documented weak-band off switch (=0) silently turned
            # the review floor to 0 and flooded the queue with every pair in
            # the gallery.
            floor=config.review_floor(),
            limit=body.limit,
            gender_min_p=config.review_gender_min_p(),
            age_child_max=config.review_age_child_max(),
            age_adult_min=config.review_age_adult_min(),
            stature_gap=config.review_stature_gap(),
            stature_min_n=config.review_stature_min_n(),
            adult_m=config.adult_height_m(),
            clothes_clash=config.review_clothes_clash(),
            clothes_min_n=config.review_clothes_min_n(),
            clothes_self_min=config.review_clothes_self_min(),
        )
    except gallery.BadRunIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"runId": body.runId, "threshold": config.threshold(), **report}


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

    ``bodyId`` (optional) names the body-sighting row the same /match call
    logged; it is deleted whatever happens to the template (a body log has
    no last-row rule) and ``bodyForgotten`` says whether a row went.  The
    box was logged under a key the caller has since proven wrong, and the
    caller's re-ask logs it again under the right one.
    """
    try:
        forgotten = False
        if body.templateId is not None:
            forgotten = gallery.forget_template(
                config.data_dir(), body.runId, body.templateId
            )
        body_forgotten = False
        if body.bodyId is not None:
            body_forgotten = gallery.forget_body_sighting(
                config.data_dir(), body.runId, body.bodyId
            )
    except gallery.BadRunIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {
        "ok": True,
        "forgotten": forgotten,
        "bodyForgotten": body_forgotten,
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
