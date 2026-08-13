"""FastAPI app for the runner service (port 7100).

Endpoints (CONTRACTS.md):
    GET  /health          -> {ok, model, version, pipeline}
    POST /runs            {eventId, placementId?, source:{url|path}, plannerUrl?,
                           label?, mode?:'count'|'enrol', siteId?, staffId?,
                           exclusionZones?:[{label, points:[[x,y],...]}]}
    GET  /runs/{runId}    -> live local status
    POST /runs/{runId}/stop

``mode`` defaults to ``count`` (the counting loop).  ``mode:'enrol'`` runs the
staff-enrolment walk-through (CONTRACTS.md v1) and requires ``siteId`` +
``staffId``; ``siteId`` on a count run opts it into the staff whitelist.
"""

import importlib
import json
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from heco_common.gate_auth import install_bearer_gate
from pydantic import BaseModel, Field, model_validator

from . import config
from .loop import refuse_model_profile
from .runs import RunManager


def _build_id() -> str:
    """Hash every .py this process could be running, by path and content.

    Deterministic across machines: paths are made relative to each root and
    the roots are walked in sorted order, so two boxes with identical source
    produce an identical digest regardless of where the tree happens to live
    or what order the filesystem returns entries in.

    Truncated to 12 hex characters — the same length a short git sha uses,
    for the same reason: long enough that a collision is not a practical
    concern, short enough to read out over the phone.

    Never raises. A build id is diagnostic, and a health endpoint that fails
    because it could not compute its own diagnostic would be the joke that
    writes itself.
    """
    import hashlib

    roots = [Path(__file__).resolve().parent]
    for mod in ("heco_common", "heco_counting"):
        try:
            roots.append(Path(importlib.import_module(mod).__file__).resolve().parent)
        except Exception:  # noqa: BLE001 — an absent library is not a crash here
            continue
    digest = hashlib.sha256()
    try:
        for root in roots:
            for path in sorted(root.rglob("*.py")):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    except Exception:  # noqa: BLE001 — see the docstring
        return "unknown"
    return digest.hexdigest()[:12]


VERSION = "0.2.0"  # 0.2.0: models stamped into every run config (doc 15 M1)

#: A content hash of the source this process is actually running.
#:
#: WHY THIS EXISTS. ``VERSION`` is a declared string. Every box in the fleet
#: has reported "0.1.0" through every deploy the pipeline has ever had, so it
#: cannot answer the question a deploy actually raises — is this box running
#: what I think it is? Two of the four boxes are not even git checkouts (they
#: are tar copies), so a git sha would not be askable there, and a sha would
#: not notice a file edited in place on the box either.
#:
#: So this hashes the SOURCE, not the provenance: every .py under the app and
#: the two installed libraries, by path and content. Two boxes reporting the
#: same build are running the same code, full stop. It is computed once at
#: import, costs a few milliseconds on a tree this size, and degrades to
#: "unknown" rather than failing a health check it exists to inform.
#:
#: This is the missing half of the deploy-integrity guards: models.lock says
#: the WEIGHTS are what they should be, and this says the CODE is.
BUILD_ID = _build_id()
log = logging.getLogger("runner")

app = FastAPI(title="heco-runner", version=VERSION)
# Inbound auth (runbook step 8): armed by HECO_REQUIRE_AUTH=1, this refuses
# LAN callers without a bearer credential — an heco-auth token verified
# locally, or the legacy shared secret while it survives. /health stays
# open for the compose healthchecks. Unarmed, nothing changes.
install_bearer_gate(app)
manager = RunManager(config.from_env())


class Source(BaseModel):
    """Where frames come from: an RTSP/HTTP url or a local file path.

    Mirrors ingest's OpenSource (exactly one of url/path; `loop` restarts a
    file at EOF so a short clip behaves like an endless camera; `isFile`
    marks a url as a finite recording — paced to its native FPS, run ends at
    EOF — which is how the planner's uploaded-video runs arrive).  The loop
    forwards this dict to ingest verbatim, so the flag must be declared here
    or validation would silently drop it.
    """

    url: str | None = None
    path: str | None = None
    loop: bool = False
    isFile: bool = False
    #: Process EVERY frame of a recording rather than the ones a slower
    #: pipeline happened to be free for — ingest holds each frame until this
    #: loop takes it. File sources only (ingest forces it off for a camera);
    #: the run then takes longer than the footage, which is the honest cost
    #: of examining all of it. See heco_common.schemas.OpenSource.lockstep.
    lockstep: bool = False

    @model_validator(mode="after")
    def _one_of(self) -> "Source":
        if bool(self.url) == bool(self.path):
            raise ValueError("source needs exactly one of url or path")
        return self


class ExclusionZone(BaseModel):
    """One operator-drawn polygon where faces must NOT be counted.

    The planner control proxy copies these from the device config; the shape
    is the wire contract (CONTRACTS.md): ``points`` are ordered polygon
    vertices NORMALIZED 0..1 relative to the full frame, minimum 3.  They
    exist because the live bench minted p00004 from an 87 px face seen THROUGH
    A FROSTED GLASS PARTITION — a real face, inside an office, not at the gate.
    No quality floor can reject a face for being in the wrong place; only the
    operator knows where the mirrors, partitions and TV screens are.

    Validation is strict at this edge so the loop never has to be: a zone that
    reaches the frame loop is guaranteed drawable and testable.
    """

    label: str
    points: list[list[float]] = Field(min_length=3)
    #: 'faces' (default): the zone excludes FACES from counting while bodies
    #: stay detected and tracked — the safe mode for partitions and mirrors a
    #: real guest can walk in front of. 'detections': the zone also drops
    #: PERSON boxes before the tracker, so no track ever forms — for surfaces
    #: that GENERATE garbage tracks (a wall TV playing faces, a poster) and
    #: that no real person can stand inside. A real guest crossing a
    #: detections zone has their track cut and may be double-counted; the
    #: planner's zone editor says so where the mode is chosen.
    mode: Literal["faces", "detections"] = "faces"

    @model_validator(mode="after")
    def _points_are_normalized_pairs(self) -> "ExclusionZone":
        for i, p in enumerate(self.points):
            if len(p) != 2:
                raise ValueError(
                    f"zone '{self.label}' point {i} must be [x, y], got {p!r}"
                )
            if not all(0.0 <= v <= 1.0 for v in p):
                raise ValueError(
                    f"zone '{self.label}' point {i} must be normalized 0..1 "
                    f"relative to the frame, got {p!r}"
                )
        return self


class QualityProfile(BaseModel):
    """Per-run overrides for the composite quality gate.

    The floors are runner-wide environment configuration by default, which is
    right for "this camera, this mount, always" and wrong for the two things
    an operator actually does: trying a stricter gate on tonight's footage
    before trusting it, and running one event under a profile that differs
    from the box's default because the camera was moved.

    So the launcher may send them per run.  Absent fields keep the runner's
    configured value — this is an override, never a reset, so a profile that
    only names ``requireLandmarks`` cannot silently disarm a frontality floor
    an engineer set on the box.

    Whatever arrives is recorded in the run row's gate config (RunLoop's
    ``_gate_config``), because the count is an invoice figure and "which
    floors were in force when this number was produced" has to be recoverable
    from the record months later.
    """

    minPx: float | None = Field(default=None, ge=0)
    minIedPx: float | None = Field(default=None, ge=0)
    minFrontality: float | None = Field(default=None, ge=0, le=1)
    minSharpness: float | None = Field(default=None, ge=0)
    minEyeSpan: float | None = Field(default=None, ge=0, le=1)
    requireLandmarks: bool | None = None
    #: Seconds between face re-verifications of a track that already holds an
    #: identity.  0 disables the saving (search everyone every frame).
    faceReverifyIntervalS: float | None = Field(default=None, ge=0)


class ModelProfile(BaseModel):
    """A named selection from this pipeline's declared model catalog.

    ``stages`` must select at least one model: an empty selection would pass
    the gate vacuously and stamp an empty promise into the permanent run
    record (review finding, 2026-08-14) — a profile that chooses nothing is
    a request with no meaning, refused as such.
    """

    id: str | None = None
    name: str | None = None
    stages: dict[str, str] = Field(min_length=1)


class RunRequest(BaseModel):
    """Body of POST /runs — what to run and where to report it.

    ``mode`` selects the counting loop (default) or the staff-enrolment
    walk-through.  ``siteId`` names the site whose staff store to check (count)
    or enrol into; ``staffId`` names the roster member being enrolled.
    ``exclusionZones`` are the operator-drawn no-count polygons the loop
    filters faces against (see :class:`ExclusionZone`).
    """

    eventId: str
    placementId: str | None = None
    #: The planner's named model selection (doc 15 §5). Honoured only when
    #: this install's LIVE models are the ones it names — checked at POST
    #: time against each service's /health, refused with a sentence naming
    #: the mismatch (contract §2.2: a 400, never a silently different model
    #: under a stamped profile name).
    modelProfile: ModelProfile | None = None
    source: Source
    plannerUrl: str | None = None
    label: str | None = None
    mode: Literal["count", "enrol"] = "count"
    siteId: str | None = None
    staffId: str | None = None
    exclusionZones: list[ExclusionZone] | None = None
    #: Per-run quality-gate overrides (see :class:`QualityProfile`).
    quality: QualityProfile | None = None
    #: Engineering-station bench mode: post the RAW frame for EVERY processed
    #: frame (not just the sampled tap rounds), so the console can scrub the
    #: run frame by frame. Costs one LAN upload per frame — a deliberate,
    #: opt-in trade documented at the toggle.
    forensic: bool = False

    @model_validator(mode="after")
    def _enrol_needs_site_and_staff(self) -> "RunRequest":
        if self.mode == "enrol" and not (self.siteId and self.staffId):
            raise ValueError("enrol mode requires siteId and staffId")
        return self


#: The manifest, read once from the repo root. A missing or unreadable file
#: is not fatal: /health is what the compose healthcheck polls, and refusing
#: to answer it because a descriptive JSON file is absent would take a
#: counting pipeline off the air over documentation.
_MANIFEST_CACHE: dict | None = None


def _manifest() -> dict | None:
    """This pipeline's capability manifest, or None when it cannot be read."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        path = Path(__file__).resolve().parents[3] / "pipeline.json"
        try:
            _MANIFEST_CACHE = json.loads(path.read_text())
        except (OSError, ValueError):
            log.warning("no readable pipeline.json at %s — /health will omit the manifest", path)
            _MANIFEST_CACHE = {}
    return _MANIFEST_CACHE or None


@app.get("/health")
def health() -> dict:
    """Liveness + identity + the MANIFEST the planner registers.

    ``model`` and ``version`` are unchanged: the compose healthcheck and the
    deploy runbook read them, and moving a field somebody polls is a wire
    change dressed as a refactor.

    ``build`` is a content hash of the source actually running — see
    :data:`BUILD_ID`. ``version`` is a declared string that every box has
    reported as "0.1.0" through every deploy this pipeline has ever had, so
    it cannot answer the one question a deploy raises: is this box running
    what I think it is? Two boxes reporting the same ``build`` are running the
    same code; two reporting different ones are not, whatever either says
    about its version, and whether or not the box is even a git checkout.

    ``pipeline`` is the addition — the capability manifest from
    ``pipeline.json`` (apps/heco-pipelines/CONTRACTS.md §1). The planner's
    registry reads it to decide what the console offers, and it branches on
    DECLARED CAPABILITY rather than on which pipeline it is talking to. That
    is the whole mechanism by which a second pipeline becomes offerable by
    declaring itself rather than by editing the planner.
    """
    return {
        "ok": True,
        "model": "orchestrator",
        "version": VERSION,
        "build": BUILD_ID,
        "pipeline": _manifest(),
    }


@app.post("/runs")
def create_run(body: RunRequest) -> dict:
    """Start a run in a background thread; poll GET /runs/{id} for progress."""
    if body.modelProfile is not None:
        refusal = refuse_model_profile(
            body.modelProfile.model_dump(), _manifest(), manager.probe_live_models()
        )
        if refusal:
            raise HTTPException(status_code=400, detail=refusal)
    request = body.model_dump(exclude_none=True)
    if body.modelProfile is not None:
        # The run thread re-checks the profile against the models it STAMPS
        # (TOCTOU guard) and needs the same catalog this gate used.
        request["_manifest"] = _manifest()
    run_id = manager.start(request)
    return {"runId": run_id, "state": "starting"}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """Live local status of a run (planner holds the durable record)."""
    st = manager.get(run_id)
    if st is None:
        raise HTTPException(status_code=404, detail="unknown run")
    return st


@app.post("/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    """Ask a run to end after its current frame (needed for RTSP sources)."""
    if not manager.stop(run_id):
        raise HTTPException(status_code=404, detail="unknown run")
    return {"ok": True, "runId": run_id}
