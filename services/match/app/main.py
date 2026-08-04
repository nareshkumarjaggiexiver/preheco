"""FastAPI app for the match service (port 7106).

Stage 6 of the pipeline plus the staff whitelist and the operator-correction
surface the runner drives from the feedback loop.

Endpoints:
    GET  /health -> {ok, model, version, threshold, canonPx}
    POST /reset  {runId} -> {ok, runId}
    POST /match  {runId, embedding, quality?, siteId?}
        -> {personKey, isNew, cosine, galleryN, subCanon, isStaff, staffId}
    POST /staff/enrol {siteId, staffId, samples:[{embedding, quality?, subCanon?}]}
        -> {staffId, sampleCount}
    POST /merge  {runId, keep, drop}          -> {merged, galleryN}   (duplicate)
    POST /split  {runId, a, b}                -> {ok, galleryN}       (false-match)
    POST /mark-staff {runId, personKey, siteId?, staffId?}
        -> {moved, galleryN, staffKey}                                (mark-staff)

Staff are checked FIRST (CONTRACTS.md v1): a staff hit is tagged
``isStaff=true`` and excluded from the guest unique count, but the track that
carries it stays visible and tracked upstream.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, gallery, staff

VERSION = "0.2.0"

app = FastAPI(title="heco-match", version=VERSION)


class ResetRequest(BaseModel):
    """Body of POST /reset — which run's gallery to wipe."""

    runId: str


class MatchRequest(BaseModel):
    """Body of POST /match — one embedding to resolve against the run gallery.

    ``siteId`` opts a run into the staff whitelist: when present the staff
    store for that site is checked before the guest gallery.
    """

    runId: str
    embedding: list[float] = Field(min_length=8)
    quality: float | None = None  # face box width in px; <canon is tagged sub-canon
    siteId: str | None = None


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
    """Body of POST /merge — fold ``drop`` into ``keep`` (duplicate correction)."""

    runId: str
    keep: str
    drop: str


class SplitRequest(BaseModel):
    """Body of POST /split — record a do-not-merge pair (false-match correction)."""

    runId: str
    a: str
    b: str


class MarkStaffRequest(BaseModel):
    """Body of POST /mark-staff — move a guest person to the staff store."""

    runId: str
    personKey: str
    siteId: str | None = None
    staffId: str | None = None


@app.get("/health")
def health() -> dict:
    """Liveness + identity: no ML model here, the 'model' is the gallery policy."""
    return {
        "ok": True,
        "model": "cosine-gallery-sqlite",
        "version": VERSION,
        "threshold": config.threshold(),
        "canonPx": config.canon_px(),
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
            }

    try:
        r = gallery.match(
            data_dir,
            body.runId,
            body.embedding,
            body.quality,
            threshold=threshold,
            canon_px=canon_px,
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
    """Duplicate correction: fold ``drop`` into ``keep`` (count −1 if applied)."""
    try:
        merged, n = gallery.merge(config.data_dir(), body.runId, body.keep, body.drop)
    except gallery.BadRunIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"merged": merged, "galleryN": n}


@app.post("/split")
def split(body: SplitRequest) -> dict:
    """False-match correction: record a do-not-merge constraint for the pair."""
    try:
        n = gallery.split(config.data_dir(), body.runId, body.a, body.b)
    except gallery.BadRunIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True, "galleryN": n}


@app.post("/mark-staff")
def mark_staff(body: MarkStaffRequest) -> dict:
    """Move a guest person out of the gallery and into the staff store."""
    data_dir = config.data_dir()
    try:
        templates, n = gallery.remove(data_dir, body.runId, body.personKey)
    except gallery.BadRunIdError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    staff_key: str | None = None
    if templates and body.siteId:
        try:
            staff_key, _ = staff.absorb(data_dir, body.siteId, body.staffId, templates)
        except staff.BadSiteIdError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
    return {"moved": len(templates), "galleryN": n, "staffKey": staff_key}
