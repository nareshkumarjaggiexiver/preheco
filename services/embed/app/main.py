"""FastAPI app for the embed service (heco-pipeline port 7105).

Contract (CONTRACTS.md):
    POST /embed {imageB64, faces: [{box, landmarks, conf?}]}
        -> {embeddings: [[128 floats]], alignMs,
            norms: [float],                       # L2 norm of each RAW feature
            attributes: [{gender, genderP, age} | null] | null,
            attrMs: float | null}
    GET  /health -> {ok, model, version, device, attrModel, attrError,
                     headwear?: {model, stamp, error, device}}
    POST /headwear {imageB64, faces: [{box}], crop?: {x, y, frameW, frameH}}
        -> {readings: [[8 floats] | null], model, ms}

`norms` and `attributes` are index-parallel to `embeddings`. `attributes`
is null as a WHOLE when no attribute model is configured (EMBED_ATTR_MODEL
— see attributes.py); null means "not measured", never "nobody".

The head-covering reader (headwear.py) exists only when EMBED_HEADWEAR_MODEL
names its graph: without it /health carries no `headwear` key, /embed is
untouched either way, and POST /headwear answers 404 exactly like an unknown
route.

Callers (the runner) apply the POC quality gate first — only faces at or above
the 56 px floor should reach embedding (sub-canon ones flagged upstream).
"""

import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from heco_common.gate_auth import install_bearer_gate
from heco_common.ort import is_trt
from pydantic import BaseModel, Field

from . import __version__
from .align import align_face, half_balance
from .attributes import ATTR_MODEL_EXPLICIT, ATTR_MODEL_PATH, AttributeModel, build_attributes
from .codec import b64_to_bgr, frame_from
from .headwear import HEADWEAR_MODEL_PATH, PROMPTS_FILE, HeadwearReader, build_headwear
from .recognizer import BATCH, BATCH_MAX, DEVICE, MODEL_PATH, FaceEmbedder, build_embedder

log = logging.getLogger("embed")

app = FastAPI(title="heco embed", version=__version__)
# Inbound auth (runbook step 8): armed by HECO_REQUIRE_AUTH=1, this refuses
# LAN callers without a bearer credential — an heco-auth token verified
# locally, or the legacy shared secret while it survives. /health stays
# open for the compose healthchecks. Unarmed, nothing changes.
install_bearer_gate(app)

_embedder: FaceEmbedder | None = None
_load_error: str | None = None
_load_lock = threading.Lock()
_failed_stat: tuple | None = None  # (mtime_ns, size) of the file that failed


def _get_embedder() -> FaceEmbedder | None:
    """Load the configured family lazily so the app can boot unhealthy without weights.

    The persons review rules apply here too:

      * ANY load failure is caught, not just a missing file — a truncated
        weight raises InvalidProtobuf, and an exception escaping /health
        turns the documented "boot unhealthy" contract into a 500 per probe;
      * the failure is sticky per FILE STATE, not per process: /health keeps
        answering ok:false cheaply, but when MODEL_PATH's bytes CHANGE on
        disk (the bind-mount delivering weights after boot, a re-fetch
        fixing a truncation) the next probe retries — "ok is false until
        the weights are loadable" now means what it says. This matters MORE
        here than in persons/faces: embed deliberately has no POST /model
        (an embedder change is a deployment — see recognizer.py), so a
        stat-gated retry is the ONLY recovery short of a manual restart;
      * one lock: two concurrent first-probes must not both build a session.
    """
    global _embedder, _load_error, _failed_stat
    if _embedder is not None:
        return _embedder
    with _load_lock:
        if _embedder is not None:
            return _embedder
        if _load_error is not None and _current_stat() == _failed_stat:
            return None  # same broken file — stay cheap, stay unhealthy
        # Stat BEFORE the attempt: a writer replacing the file DURING a
        # failed load must invalidate the memo, not be masked by a
        # post-failure stat of the new file (persons review finding).
        attempt_stat = _current_stat()
        try:
            _embedder = build_embedder()
            _load_error = None
            _failed_stat = None
        except Exception as exc:  # noqa: BLE001 — see the docstring
            _load_error = f"{type(exc).__name__}: {exc}"
            _failed_stat = attempt_stat
            log.error("model load failed: %s", _load_error)
    return _embedder


def _current_stat() -> tuple | None:
    """(mtime_ns, size) of the model file, or None while it is absent."""
    return _stat_of(MODEL_PATH)


def _stat_of(path: Path | None) -> tuple | None:
    """(mtime_ns, size) of `path`, or None while it is absent (or unset)."""
    if path is None:
        return None
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


_attributes: AttributeModel | None = None
_attr_error: str | None = None
_attr_lock = threading.Lock()
_attr_failed_stat: tuple | None = None


def _get_attributes() -> AttributeModel | None:
    """Load the optional genderage model lazily, with the embedder's retry rules.

    Same sticky-per-file-state loop as `_get_embedder` — one failed load is
    memoized until the file's (mtime, size) changes, then retried once —
    with two differences that follow from the pass being OPTIONAL:

      * the DEFAULT file merely being absent is the pass being off, not an
        error: nothing is logged, /health says attrModel null and the
        next probe stats again (so a bind mount delivering the weights
        after boot switches the pass on without a restart);
      * a file the operator NAMED (EMBED_ATTR_MODEL) that is missing or
        refused IS an error — but it is served as /health attrError, not
        as ok:false. Embedding still works and the count must not stop for
        an advisory signal; what must not happen is the loss being silent,
        which is why the reason is served and logged rather than swallowed.
    """
    global _attributes, _attr_error, _attr_failed_stat
    if _attributes is not None:
        return _attributes
    if ATTR_MODEL_PATH is None:
        return None  # EMBED_ATTR_MODEL=off
    with _attr_lock:
        if _attributes is not None:
            return _attributes
        attempt_stat = _stat_of(ATTR_MODEL_PATH)
        if attempt_stat is None and not ATTR_MODEL_EXPLICIT:
            _attr_error = None
            return None  # default file absent: off, quietly
        if _attr_error is not None and attempt_stat == _attr_failed_stat:
            return None  # same broken file — stay cheap
        try:
            # Batching reaches the attribute pass only when switched on; off,
            # this is exactly the call it always was.
            _attributes = (
                build_attributes(ATTR_MODEL_PATH, DEVICE, BATCH, BATCH_MAX)
                if BATCH else build_attributes(ATTR_MODEL_PATH, DEVICE)
            )
            _attr_error = None
            _attr_failed_stat = None
        except Exception as exc:  # noqa: BLE001 — surfaced in /health, see above
            _attr_error = f"{type(exc).__name__}: {exc}"
            _attr_failed_stat = attempt_stat
            log.error("attribute model load failed: %s", _attr_error)
    return _attributes


_headwear: HeadwearReader | None = None
_headwear_error: str | None = None
_headwear_lock = threading.Lock()
_headwear_failed_stat: tuple | None = None


def _headwear_stat(path: Path | None) -> tuple | None:
    """Stat the graph, its prompt set and the default text bank.

    A fix to ANY of the three is worth one more load attempt.
    """
    if path is None:
        return None
    return tuple(
        _stat_of(p) for p in (path, path.parent / PROMPTS_FILE,
                              path.parent / "headwear_text_embeds.npy")
    )


def _get_headwear() -> HeadwearReader | None:
    """Load the optional head-covering reader lazily, with the embedder's retry rules.

    Off (EMBED_HEADWEAR_MODEL unset, empty or `off`): None, nothing touched.
    On, the variable NAMES a file, so a missing or refused set is an error —
    served as /health headwear.error, never as ok:false (embedding and the
    count do not stop for an advisory read) — memoized until one of the
    three files changes on disk, then retried once. Under TensorRT the load
    builds the engine (a cold build is minutes, cached after; the TRT
    overlay's start period covers it).
    """
    global _headwear, _headwear_error, _headwear_failed_stat
    if _headwear is not None:
        return _headwear
    if HEADWEAR_MODEL_PATH is None:
        return None
    with _headwear_lock:
        if _headwear is not None:
            return _headwear
        attempt_stat = _headwear_stat(HEADWEAR_MODEL_PATH)
        if _headwear_error is not None and attempt_stat == _headwear_failed_stat:
            return None  # same broken set — stay cheap
        try:
            _headwear = build_headwear(HEADWEAR_MODEL_PATH, DEVICE)
            _headwear_error = None
            _headwear_failed_stat = None
        except Exception as exc:  # noqa: BLE001 — surfaced in /health, see above
            _headwear_error = f"{type(exc).__name__}: {exc}"
            _headwear_failed_stat = attempt_stat
            log.error("head-covering reader load failed: %s", _headwear_error)
    return _headwear


class FaceIn(BaseModel):
    """One face from the faces service: box + YuNet-order landmarks."""

    box: dict
    landmarks: list[list[float]]
    conf: float | None = None


class EmbedRequest(BaseModel):
    """POST /embed body: the frame plus the faces to align and embed."""

    #: Empty is legal ONLY with a resolvable frameRef — the pixels come from
    #: the shared transport instead. The handler refuses (400) when neither
    #: yields a frame, which says WHICH half failed; a min_length here would
    #: refuse the ref-only request before anything could look at the ref.
    imageB64: str = ""
    #: See persons.DetectRequest.frameRef — same contract, same fallback.
    frameRef: str | None = None
    faces: list[FaceIn]


class HeadwearFaceIn(BaseModel):
    """One face to read the head of: its detector box, in the image's pixels."""

    box: dict


class CropIn(BaseModel):
    """Where the image sits in its frame, when it is a cut and not the frame.

    With it the reader can tell a frame edge (clamp, as the reference does)
    from a cut edge (refuse: the head region would be clipped by the cut).
    """

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    frameW: int = Field(gt=0)
    frameH: int = Field(gt=0)


class HeadwearRequest(BaseModel):
    """POST /headwear body: a picture holding the heads, and the faces to read."""

    imageB64: str = Field(min_length=1)
    faces: list[HeadwearFaceIn] = Field(max_length=64)
    crop: CropIn | None = None


@app.get("/health")
def health() -> dict:
    """Liveness + model identity; ok is false until the weights are loadable."""
    emb = _get_embedder()
    attrs = _get_attributes()
    body = _health_body(emb, attrs)
    if HEADWEAR_MODEL_PATH is not None:
        # Only when the reader is switched on: off, /health keeps exactly the
        # keys it always had.
        body["headwear"] = _headwear_block(_get_headwear())
    return body


def _headwear_block(hw: HeadwearReader | None) -> dict:
    """Build the reader's /health block: graph, stamp, load error, device truth.

    ``model``/``stamp`` null while not loaded, with ``error`` saying why;
    ``device`` is the reader's OWN truth on any device — it is a second
    session with its own fallbacks — with ``trt`` riding along under TRT.
    """
    if hw is None:
        return {"model": None, "stamp": None, "error": _headwear_error, "device": None}
    device = {"requested": hw.device_requested, "active": list(hw.providers_active)}
    if is_trt(hw.device_requested):
        device["trt"] = getattr(hw, "trt", None)
    return {"model": hw.model_name, "stamp": hw.stamp, "error": None, "device": device}


def _health_body(emb, attrs) -> dict:
    """Build the /health reply as it always was (the embedder and the attribute pass)."""
    return {
        "ok": emb is not None,
        "model": emb.model_name if emb else MODEL_PATH.name,
        "version": __version__,
        # The optional gender/age pass: its file name while loaded, null
        # while off — and, when a NAMED file will not load, the reason
        # (ok stays true: embedding works, the loss must just not be silent).
        "attrModel": attrs.model_name if attrs else None,
        "attrError": None if attrs is not None else _attr_error,
        # While unhealthy, say WHY: an operator staring at a red healthcheck
        # needs the blocking error (missing file? truncated? bad dtype?)
        # without docker exec — the deploy-integrity lesson.
        "error": None if emb is not None else _load_error,
        # Family, device truth AND dimension: the runner cross-checks dim
        # against the deployment's HECO_EMBEDDING_DIM story, and the family
        # says which alignment produced these vectors.
        "device": _device_block(emb, attrs) if emb else None,
        # EMBED_BATCH as asked and as it runs (a static-batch graph cannot).
        "knobs": _knobs(emb) if emb else None,
    }


def _knobs(emb) -> dict:
    """Build the /health knobs block: each switch as asked AND as it actually runs."""
    return {
        "batch": {
            "requested": bool(getattr(emb, "batch_requested", False)),
            "active": bool(getattr(emb, "batch_active", False)),
            "max": BATCH_MAX,
        },
    }


def _device_block(emb, attrs=None) -> dict:
    """Build the /health device truth; `trt` rides along only when TRT was asked.

    Keyed on the REQUEST, not on success: a TensorRT that failed to load
    answers "trt": null beside an `active` list without it.

    ``attributes`` rides with it, for the genderage pass when it is loaded:
    under TRT that pass builds an engine of its own (20-35 s cold, measured)
    and can fall back to CUDA or CPU on its own, which until now left only a
    stderr line. Off TRT the block keeps its four keys exactly.
    """
    block = {
        "requested": emb.device_requested,
        "active": emb.providers_active,
        "family": emb.family,
        "dim": emb.dim,
    }
    if is_trt(emb.device_requested):
        block["trt"] = getattr(emb, "trt", None)
        if attrs is not None:
            block["attributes"] = {
                "requested": attrs.device_requested,
                "active": list(attrs.providers_active),
                "trt": getattr(attrs, "trt", None),
            }
    return block


@app.post("/embed")
def embed(req: EmbedRequest) -> dict:
    """Align (by landmarks) and embed each face; order matches the request."""
    emb = _get_embedder()
    if emb is None:
        raise HTTPException(status_code=503, detail=_load_error or "model not loaded")
    try:
        img = frame_from(req.imageB64, req.frameRef)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        embeddings, norms, attributes, align_ms, attr_ms = emb.embed_faces(
            img, [f.model_dump() for f in req.faces], _get_attributes()
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "embeddings": embeddings,
        "alignMs": align_ms,
        "norms": norms,
        "attributes": attributes,
        "attrMs": attr_ms,
        # Index-parallel, like norms: how evenly each face's two halves were
        # seen (align.half_balance). Measured on the same ArcFace-template
        # crop whichever family embeds, so the number means one thing.
        "balance": [_balance(img, f.model_dump()) for f in req.faces],
    }


def _balance(img, face: dict) -> float | None:
    """One face's half-balance, or None when it cannot be measured."""
    try:
        return half_balance(align_face(img, face.get("landmarks")))
    except Exception:  # noqa: BLE001 — a reading, never a reason to fail /embed
        return None


@app.post("/headwear", include_in_schema=HEADWEAR_MODEL_PATH is not None)
def headwear(req: HeadwearRequest) -> dict:
    """Read each face's head covering: 8 logits per face (loose view, then tight).

    The runner's background reader is the caller, once per mint or template
    enrolment, with a PNG context crop (lossless: the frame it holds is
    already JPEG q85) and ``crop`` saying where it was cut. Readings are
    index-parallel to ``faces``; null where the face's head region lies
    outside the image. ``model`` is the reader's stamp (graph sha256[:12] +
    prompt set sha256[:12]) — the match service keeps one per gallery.

    Off (EMBED_HEADWEAR_MODEL unset): 404, exactly an unknown route's answer.
    On but not loadable: 503 with the reason (also on /health). A malformed
    box, an undecodable image, or a crop that does not cover a head region
    it should: 400.
    """
    if HEADWEAR_MODEL_PATH is None:
        raise HTTPException(status_code=404, detail="Not Found")
    reader = _get_headwear()
    if reader is None:
        raise HTTPException(
            status_code=503, detail=_headwear_error or "head-covering reader not loaded"
        )
    try:
        img = b64_to_bgr(req.imageB64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    crop = None if req.crop is None else req.crop.model_dump()
    if crop is not None:
        height, width = img.shape[:2]
        if crop["x"] + width > crop["frameW"] or crop["y"] + height > crop["frameH"]:
            raise HTTPException(
                status_code=400,
                detail="crop: a {w}x{h} image at ({x}, {y}) does not fit a {fw}x{fh} frame".format(
                    w=width, h=height, x=crop["x"], y=crop["y"],
                    fw=crop["frameW"], fh=crop["frameH"],
                ),
            )
    t0 = time.perf_counter()
    try:
        readings = reader.read(img, [f.box for f in req.faces], crop)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "readings": readings,
        "model": reader.stamp,
        "ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
