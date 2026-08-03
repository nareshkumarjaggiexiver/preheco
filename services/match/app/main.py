"""FastAPI app for the match service (port 7106).

Endpoints (CONTRACTS.md):
    GET  /health -> {ok, model, version}
    POST /reset  {runId} -> {ok, runId}
    POST /match  {runId, embedding, quality} -> {personKey, isNew, cosine, galleryN, subCanon}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, gallery

VERSION = "0.1.0"

app = FastAPI(title="heco-match", version=VERSION)


class ResetRequest(BaseModel):
    """Body of POST /reset — which run's gallery to wipe."""

    runId: str


class MatchRequest(BaseModel):
    """Body of POST /match — one embedding to resolve against the run gallery."""

    runId: str
    embedding: list[float] = Field(min_length=8)
    quality: float | None = None  # face box width in px; <80 is tagged sub-canon


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
    """Resolve one embedding: existing personKey (isNew=false) or a fresh one."""
    try:
        r = gallery.match(
            config.data_dir(),
            body.runId,
            body.embedding,
            body.quality,
            threshold=config.threshold(),
            canon_px=config.canon_px(),
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
    }
