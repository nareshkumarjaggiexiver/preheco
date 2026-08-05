"""Unit tests for the wire schemas: shapes, validation rules, camelCase."""

import pytest
from heco_common import schemas as S
from pydantic import ValidationError


def test_box_roundtrip_and_optional_conf():
    """Box serialises camelCase-free fields and allows a missing conf."""
    b = S.Box(x=1, y=2, w=3, h=4)
    assert b.model_dump() == {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0, "conf": None}
    assert S.Box.model_validate({"x": 0, "y": 0, "w": 10, "h": 20, "conf": 0.9}).conf == 0.9


def test_box_rejects_negative_size():
    """Width/height below zero are contract violations."""
    with pytest.raises(ValidationError):
        S.Box(x=0, y=0, w=-1, h=5)


def test_open_source_requires_exactly_one():
    """POST /open must carry url XOR path."""
    assert S.OpenSource(path="/tmp/clip.avi").loop is False
    assert S.OpenSource(url="rtsp://cam/1", loop=True).loop is True
    with pytest.raises(ValidationError):
        S.OpenSource()
    with pytest.raises(ValidationError):
        S.OpenSource(url="rtsp://cam/1", path="/tmp/clip.avi")


def test_frame_wire_shape():
    """Frame dumps the exact contract keys."""
    f = S.Frame(tMs=120, imageB64="abc=", w=640, h=480, seq=7)
    # `ended` joined the contract so the runner can tell a finished FILE from a
    # blinking CAMERA — both freeze `seq`, and only one means the count is done.
    assert set(f.model_dump()) == {"tMs", "imageB64", "w", "h", "seq", "ended"}
    assert f.ended is False, "a live source never claims to have ended"


def test_track_shapes():
    """Tracker request/response match the brief (ageFrames + hits)."""
    req = S.TrackRequest(runId="r1", tMs=0, boxes=[{"x": 0, "y": 0, "w": 5, "h": 5, "conf": 0.8}])
    assert req.boxes[0].conf == 0.8
    t = S.Track(id=1, box=S.Box(x=0, y=0, w=5, h=5), ageFrames=3, hits=3)
    assert set(t.model_dump()) == {"id", "box", "ageFrames", "hits"}


def test_face_needs_five_landmarks():
    """YuNet emits exactly 5 landmarks; anything else is malformed."""
    box = {"x": 0, "y": 0, "w": 60, "h": 60}
    pts = [(1.0, 2.0)] * 5
    assert len(S.Face(box=box, landmarks=pts, conf=0.9).landmarks) == 5
    with pytest.raises(ValidationError):
        S.Face(box=box, landmarks=pts[:4], conf=0.9)


def test_embedding_dim_enforced():
    """SFace vectors are exactly 128-d on both embed and match sides."""
    good = [0.0] * S.EMBEDDING_DIM
    assert S.EmbedResponse(embeddings=[good, good])
    with pytest.raises(ValidationError):
        S.EmbedResponse(embeddings=[[0.0] * 127])
    with pytest.raises(ValidationError):
        S.MatchRequest(embedding=[0.0] * 129)


def test_stage_literal_gate():
    """Only the eight contract stage names pass validation."""
    S.StageStats(stage="person-detect", frames=10, fps=15.0)
    with pytest.raises(ValidationError):
        S.StageStats(stage="persons", frames=10, fps=15.0)


def test_sample_batch_cap():
    """A samples POST carries at most 200 samples."""
    one = {"stage": "track", "tMs": 1, "metrics": {"personBoxHPx": 260.0}}
    assert len(S.SampleBatch(samples=[one] * 200).samples) == 200
    with pytest.raises(ValidationError):
        S.SampleBatch(samples=[one] * 201)


def test_poc_quality_constants():
    """The POC geometry numbers from CONTRACTS.md are exposed as constants."""
    assert S.FACE_MIN_PX == 56
    assert S.FACE_SUB_CANON_MAX_PX == 79
    assert S.PORTS["ingest"] == 7101
    assert S.PORTS["tracker"] == 7103
