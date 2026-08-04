"""FastAPI app for the runner service (port 7100).

Endpoints (CONTRACTS.md):
    GET  /health          -> {ok, model, version}
    POST /runs            {eventId, placementId?, source:{url|path}, plannerUrl?,
                           label?, mode?:'count'|'enrol', siteId?, staffId?}
    GET  /runs/{runId}    -> live local status
    POST /runs/{runId}/stop

``mode`` defaults to ``count`` (the counting loop).  ``mode:'enrol'`` runs the
staff-enrolment walk-through (CONTRACTS.md v1) and requires ``siteId`` +
``staffId``; ``siteId`` on a count run opts it into the staff whitelist.
"""

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator

from . import config
from .runs import RunManager

VERSION = "0.1.0"

app = FastAPI(title="heco-runner", version=VERSION)
manager = RunManager(config.from_env())


class Source(BaseModel):
    """Where frames come from: an RTSP/HTTP url or a local file path.

    Mirrors ingest's OpenSource (exactly one of url/path; `loop` restarts a
    file at EOF so a short clip behaves like an endless camera).
    """

    url: str | None = None
    path: str | None = None
    loop: bool = False

    @model_validator(mode="after")
    def _one_of(self) -> "Source":
        if bool(self.url) == bool(self.path):
            raise ValueError("source needs exactly one of url or path")
        return self


class RunRequest(BaseModel):
    """Body of POST /runs — what to run and where to report it.

    ``mode`` selects the counting loop (default) or the staff-enrolment
    walk-through.  ``siteId`` names the site whose staff store to check (count)
    or enrol into; ``staffId`` names the roster member being enrolled.
    """

    eventId: str
    placementId: str | None = None
    source: Source
    plannerUrl: str | None = None
    label: str | None = None
    mode: Literal["count", "enrol"] = "count"
    siteId: str | None = None
    staffId: str | None = None

    @model_validator(mode="after")
    def _enrol_needs_site_and_staff(self) -> "RunRequest":
        if self.mode == "enrol" and not (self.siteId and self.staffId):
            raise ValueError("enrol mode requires siteId and staffId")
        return self


@app.get("/health")
def health() -> dict:
    """Liveness + identity; the runner has no ML model, it conducts."""
    return {"ok": True, "model": "orchestrator", "version": VERSION}


@app.post("/runs")
def create_run(body: RunRequest) -> dict:
    """Start a run in a background thread; poll GET /runs/{id} for progress."""
    run_id = manager.start(body.model_dump(exclude_none=True))
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
