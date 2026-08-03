"""Pydantic v2 models for every inter-service message in CONTRACTS.md.

Field names are deliberately camelCase so a model's ``model_dump()`` IS the
wire JSON — no alias layer, nothing to keep in sync. One deviation from the
abbreviated table in CONTRACTS.md is intentional: the tracker returns
``ageFrames`` (and ``hits``) where the table shorthands ``age``; the detailed
tracker brief owns that shape and the field name says its unit.

Ports and stage names are also declared here so services and the runner share
one source of truth instead of re-typing the contract.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------- constants

#: Contract port per service (CONTRACTS.md topology table).
PORTS: dict[str, int] = {
    "runner": 7100,
    "ingest": 7101,
    "persons": 7102,
    "tracker": 7103,
    "faces": 7104,
    "embed": 7105,
    "match": 7106,
}

#: Planner stage names — the only values the console accepts.
Stage = Literal[
    "ingest",
    "person-detect",
    "track",
    "face-detect",
    "quality",
    "embed",
    "match",
    "count",
]

#: SFace embedding dimensionality (OpenCV zoo model, fixed by CONTRACTS.md).
EMBEDDING_DIM = 128

#: POC quality gate (CONTRACTS.md geometry): floor to enter embedding, and the
#: band below production canon that must be flagged "sub-canon" in reports.
FACE_MIN_PX = 56
FACE_SUB_CANON_MAX_PX = 79


# ------------------------------------------------------------ basic shapes


class Box(BaseModel):
    """An axis-aligned bounding box in pixels: top-left corner + size.

    ``conf`` is the detector confidence where the box came from a detector
    (persons /detect) and may be omitted where it did not (tracker output
    echoes the last matched detection's confidence, which can be None for a
    coasting prediction).
    """

    x: float
    y: float
    w: float = Field(ge=0)
    h: float = Field(ge=0)
    conf: float | None = None


class Health(BaseModel):
    """GET /health response — every service returns this exact shape."""

    ok: bool
    model: str
    version: str


# ------------------------------------------------------------------ ingest


class OpenSource(BaseModel):
    """POST /open body for ingest: exactly one of ``url`` (RTSP) or ``path``.

    ``loop`` applies to file sources only: when true the file restarts at EOF
    so a short clip behaves like an endless camera.
    """

    url: str | None = None
    path: str | None = None
    loop: bool = False

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "OpenSource":
        """Reject bodies with both or neither of url/path."""
        if bool(self.url) == bool(self.path):
            raise ValueError("provide exactly one of 'url' or 'path'")
        return self


class Frame(BaseModel):
    """GET /frame response: the latest frame, base64-JPEG in JSON.

    ``seq`` increments per captured frame; a stalled ``seq`` means the source
    ended or stalled. ``tMs`` is milliseconds since the source was opened.
    """

    tMs: int
    imageB64: str
    w: int = Field(gt=0)
    h: int = Field(gt=0)
    seq: int = Field(ge=0)


# ----------------------------------------------------------------- persons


class DetectRequest(BaseModel):
    """POST /detect body for persons (and faces without a restriction box)."""

    imageB64: str


class PersonDetections(BaseModel):
    """persons /detect response: body boxes with confidences."""

    boxes: list[Box]


# ----------------------------------------------------------------- tracker


class TrackRequest(BaseModel):
    """POST /track body: per-frame detections for one run's tracker state."""

    runId: str
    tMs: int
    boxes: list[Box]


class ResetRequest(BaseModel):
    """POST /reset body: (re)create the tracker state for a run."""

    runId: str


class Track(BaseModel):
    """One live track: stable id, current box, age and hit counts.

    ``ageFrames`` counts frames since the track was created (including missed
    ones); ``hits`` counts frames a detection actually matched.
    """

    id: int
    box: Box
    ageFrames: int = Field(ge=1)
    hits: int = Field(ge=1)


class TrackResponse(BaseModel):
    """POST /track response: tracks confirmed and updated this frame."""

    tracks: list[Track]


# ------------------------------------------------------------------- faces


class FaceDetectRequest(BaseModel):
    """POST /detect body for faces: full frame plus optional search region.

    ``within`` restricts detection to a body box (the stage-4 design: faces
    are searched inside tracked person boxes, not the whole frame).
    """

    imageB64: str
    within: Box | None = None


class Face(BaseModel):
    """One detected face: box, 5 landmarks (YuNet order), confidence.

    Landmark order: right eye, left eye, nose tip, right mouth corner, left
    mouth corner — as emitted by OpenCV's FaceDetectorYN.
    """

    box: Box
    landmarks: list[tuple[float, float]] = Field(min_length=5, max_length=5)
    conf: float


class FaceDetections(BaseModel):
    """faces /detect response."""

    faces: list[Face]


# ------------------------------------------------------------------- embed


class EmbedRequest(BaseModel):
    """POST /embed body: the frame and the faces (with landmarks) to embed."""

    imageB64: str
    faces: list[Face]


class EmbedResponse(BaseModel):
    """embed response: one 128-d SFace vector per input face, same order."""

    embeddings: list[list[float]]

    @model_validator(mode="after")
    def _dims(self) -> "EmbedResponse":
        """Every embedding must be exactly EMBEDDING_DIM long."""
        for e in self.embeddings:
            if len(e) != EMBEDDING_DIM:
                raise ValueError(f"embedding must have {EMBEDDING_DIM} dims, got {len(e)}")
        return self


# ------------------------------------------------------------------- match


class MatchRequest(BaseModel):
    """POST /match body: one embedding to resolve against the gallery."""

    embedding: list[float] = Field(min_length=EMBEDDING_DIM, max_length=EMBEDDING_DIM)


class MatchResponse(BaseModel):
    """match response: gallery decision at the cosine operating point."""

    personKey: str
    isNew: bool
    cosine: float


# ------------------------------------------------------------------ runner


class RunConfig(BaseModel):
    """POST /runs body for the runner: what to count and where to report.

    ``source`` is either an RTSP url or a file path — the runner passes it to
    ingest's /open. ``plannerUrl`` defaults to the compose-network planner.
    """

    eventId: int | str
    source: str
    plannerUrl: str = "http://host.docker.internal:8787"
    placementId: int | str | None = None
    label: str | None = None
    loop: bool = False


# ------------------------------------------- planner ingest (write side)


class MetricSummary(BaseModel):
    """Aggregate of one measured metric over a window: count/min/mean/max."""

    count: int = Field(ge=0)
    min: float
    mean: float
    max: float


class StageStats(BaseModel):
    """POST /api/pipeline/runs/:id/stats body — upserted per stage.

    ``metrics`` keys are metric names the console charts (personBoxHPx,
    faceBoxWPx, embedMs, matchCosine, ...).
    """

    stage: Stage
    frames: int = Field(ge=0)
    fps: float = Field(ge=0)
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)


class Sample(BaseModel):
    """One time-stamped point measurement from a stage."""

    stage: Stage
    tMs: int
    metrics: dict[str, float]


class SampleBatch(BaseModel):
    """POST /api/pipeline/runs/:id/samples body — batch of ≤ 200 samples."""

    samples: list[Sample] = Field(max_length=200)


class PlannerRunCreate(BaseModel):
    """POST /api/pipeline/runs body — opens a run under an event."""

    eventId: int | str
    placementId: int | str | None = None
    label: str | None = None
    config: dict | None = None


class PlannerRunEnd(BaseModel):
    """PUT /api/pipeline/runs/:id body — closes a run."""

    status: Literal["ended", "failed"]
    notes: str | None = None
