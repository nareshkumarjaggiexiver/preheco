"""Synthetic planner run documents — the fixtures the pure tests are built on.

Every number the harness reads comes out of ``GET /api/pipeline/runs/:id``, so
faking that one document is enough to test the whole scoring path with no
pipeline, no services and no network. The shapes here mirror the planner's
mappers (``pipelineRunFromRow`` / ``stageStatFromRow``) exactly.
"""

from eval.manifest import Clip


def summary(count: int, mn: float | None, mean: float | None, mx: float | None) -> dict:
    """One planner metric aggregate: {count, min, mean, max}."""
    return {"count": count, "min": mn, "mean": mean, "max": mx}


def clip(label: str = "clip-a", actual_unique: int = 10, tags: tuple = ()) -> Clip:
    """A ground-truth clip with an absolute path, as the manifest requires."""
    return Clip(
        label=label,
        path=f"/srv/heco/clips/{label}.mp4",
        actual_unique=actual_unique,
        tags=tags,
    )


def planner_run(
    unique: int = 10,
    frames: int = 400,
    person_boxes: int = 1200,
    faces_detected: int = 300,
    gate_pass: int = 120,
    embeds: int | None = 120,
    matches: int = 120,
    face_w: tuple = (58.0, 129.0, 240.0),
    ied: tuple = (24.0, 46.0, 96.0),
    gate_face_w: tuple = (57.0, 132.0, 240.0),
    fps_ingest: float = 1.59,
    fps_count: float = 1.59,
    end_reason: str | None = "source-ended",
    status: str = "ended",
    quality_min_px: float = 56.0,
    results: dict | None = None,
    include_embed_stage: bool = True,
    run_id: str = "planner-run-1",
) -> dict:
    """Build a planner run document with a fully specified funnel.

    ``embeds=None`` with ``include_embed_stage=True`` models a build that runs
    the embedder but does not emit the per-face metric — 'unknown', which the
    harness must not confuse with a collapse to zero.
    """
    stats = [
        {"stage": "ingest", "frames": frames, "fps": fps_ingest, "metrics": {}},
        {
            "stage": "person-detect",
            "frames": frames,
            "fps": fps_ingest,
            "metrics": {"personBoxHPx": summary(person_boxes, 100.0, 180.0, 300.0)},
        },
        {
            "stage": "face-detect",
            "frames": frames,
            "fps": fps_ingest,
            "metrics": {
                "faceBoxWPx": summary(faces_detected, *face_w),
                "faceIedPx": summary(faces_detected, *ied),
            },
        },
        {
            "stage": "quality",
            "frames": frames,
            "fps": fps_ingest,
            "metrics": {"faceBoxWPx": summary(gate_pass, *gate_face_w)},
        },
    ]
    if include_embed_stage:
        embed_metrics = (
            {} if embeds is None else {"faceBoxWPx": summary(embeds, *gate_face_w)}
        )
        stats.append(
            {"stage": "embed", "frames": frames, "fps": fps_ingest, "metrics": embed_metrics}
        )
    if matches:
        stats.append({"stage": "match", "frames": matches, "fps": fps_ingest, "metrics": {}})
    stats.append({"stage": "count", "frames": frames, "fps": fps_count, "metrics": {}})

    return {
        "id": run_id,
        "eventId": "evt-1",
        "label": "eval/baseline/clip-a",
        "status": status,
        "endReason": end_reason,
        "results": (
            results
            if results is not None
            else {
                "unique": unique,
                "staffCrossings": 0,
                "manualAdditions": 0,
                "frames": frames,
                "matches": matches,
            }
        ),
        "config": {
            "source": {"path": "/srv/heco/clips/clip-a.mp4"},
            "qualityMinPx": quality_min_px,
            "qualityCanonPx": 80.0,
            "qualityMinIedPx": 0.0,
            "qualityMinFrontality": 0.0,
            "qualityMinSharpness": 0.0,
            "gateArmed": [],
        },
        "stats": stats,
        "samples": [],
    }


#: The two runs that made today's INGEST_MAX_WIDTH failure, from the measured
#: numbers: 1.59 -> 7.89 fps, mean face width 129 -> 48 px, IED 46 -> 23 px,
#: faces passing the 56 px gate 9 -> 0, embed and match never ran, count 0.
def ingest_max_width_before() -> dict:
    """The run BEFORE INGEST_MAX_WIDTH=1920: slow, and counting correctly."""
    return planner_run(
        unique=9,
        frames=254,
        person_boxes=812,
        faces_detected=61,
        gate_pass=9,
        embeds=9,
        matches=9,
        face_w=(52.0, 129.0, 268.0),
        ied=(21.0, 46.0, 108.0),
        gate_face_w=(58.0, 141.0, 268.0),
        fps_ingest=1.59,
        fps_count=1.59,
        run_id="planner-run-before",
    )


def ingest_max_width_after() -> dict:
    """The run AFTER: five times the frame rate, and it counted nobody."""
    return planner_run(
        unique=0,
        frames=1261,
        person_boxes=3903,
        faces_detected=274,
        gate_pass=0,
        embeds=0,
        matches=0,
        face_w=(19.0, 48.0, 55.0),
        ied=(8.0, 23.0, 27.0),
        gate_face_w=(None, None, None),
        fps_ingest=7.89,
        fps_count=7.89,
        include_embed_stage=False,
        run_id="planner-run-after",
    )
