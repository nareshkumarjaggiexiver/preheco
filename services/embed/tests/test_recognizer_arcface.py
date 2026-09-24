"""The arcface family against real (tiny) ONNX graphs — no zoo download.

tiny_onnx hand-rolls one-node Flatten models, so the "embedding" that comes
back IS the exact blob ArcFaceEmbedder fed the session. That turns every
preprocessing promise into a value-level assertion: the NCHW-vs-NHWC branch,
the RGB conversion, the (x - 127.5) / 127.5 normalization, and the cast to
the graph's declared dtype. The refusal tests pin the init-time guards the
review added: an unsupported dtype or a 1-channel graph must raise (the lazy
loader then surfaces it as /health ok:false) instead of building a session
that 500s on every request behind a green healthcheck.
"""

import cv2
import numpy as np
import pytest
from tiny_onnx import DOUBLE, FLOAT, FLOAT16, write_model

from app.align import ARCFACE_TEMPLATE, align_face
from app.recognizer import ArcFaceEmbedder, build_embedder, spec_for

DIM = 3 * 112 * 112  # a Flatten graph's output width — read from the graph


def _frame(seed=3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


def _face(offset=100.0):
    """A face whose landmarks are the template translated into the frame."""
    lm = ARCFACE_TEMPLATE + offset
    return {
        "box": {"x": offset, "y": offset, "w": 112.0, "h": 112.0},
        "landmarks": [[float(x), float(y)] for x, y in lm],
        "conf": 0.9,
    }


def _expected_blob(img, face):
    """The normalized RGB 112x112 blob embed() documents feeding the graph."""
    aligned = align_face(img, face["landmarks"])
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) - 127.5) / 127.5


def test_nchw_fp32_feeds_the_transposed_normalized_blob(tmp_path):
    emb = ArcFaceEmbedder(
        write_model(tmp_path, "tiny_arcface_nchw.onnx", FLOAT, (1, 3, 112, 112)), "CPU"
    )
    assert emb.family == "arcface"
    assert emb.dim == DIM, "dim comes from the graph, not a spec table"
    assert emb.providers_active == ["CPUExecutionProvider"]
    img, face = _frame(), _face()
    embeddings, align_ms = emb.embed(img, [face])
    assert len(embeddings) == 1 and len(embeddings[0]) == DIM
    expected = _expected_blob(img, face).transpose(2, 0, 1).ravel()
    np.testing.assert_allclose(embeddings[0], expected, atol=1e-6)
    assert align_ms >= 0


def test_nhwc_fp32_takes_the_untransposed_branch(tmp_path):
    emb = ArcFaceEmbedder(
        write_model(tmp_path, "tiny_arcface_nhwc.onnx", FLOAT, (1, 112, 112, 3)), "CPU"
    )
    img, face = _frame(), _face()
    embeddings, _ = emb.embed(img, [face])
    expected = _expected_blob(img, face).ravel()  # HWC order: no transpose
    np.testing.assert_allclose(embeddings[0], expected, atol=1e-6)


def test_fp16_graph_receives_a_fp16_blob(tmp_path):
    """The dtype-cast fix: without the cast to the graph's declared dtype,
    ORT raises InvalidArgument on the float32 blob and every embed 500s —
    the exact green-health/dead-service failure the review reproduced."""
    emb = ArcFaceEmbedder(
        write_model(tmp_path, "tiny_arcface_fp16.onnx", FLOAT16, (1, 3, 112, 112)), "CPU"
    )
    img, face = _frame(), _face()
    embeddings, _ = emb.embed(img, [face])
    expected = (
        _expected_blob(img, face).astype(np.float16).transpose(2, 0, 1).ravel()
    )
    np.testing.assert_array_equal(
        np.asarray(embeddings[0], dtype=np.float32),
        expected.astype(np.float32),
    )


def test_embeddings_come_back_in_request_order(tmp_path):
    emb = ArcFaceEmbedder(
        write_model(tmp_path, "tiny_arcface_nchw.onnx", FLOAT, (1, 3, 112, 112)), "CPU"
    )
    img = _frame()
    first, second = _face(offset=50.0), _face(offset=200.0)
    embeddings, _ = emb.embed(img, [first, second])
    for got, face in zip(embeddings, (first, second), strict=True):
        expected = _expected_blob(img, face).transpose(2, 0, 1).ravel()
        np.testing.assert_allclose(got, expected, atol=1e-6)
    assert embeddings[0] != embeddings[1]


def test_an_unsupported_input_dtype_is_refused_at_init(tmp_path):
    """tensor(double) builds a session fine but could never accept the blob;
    the refusal names the dtype and the file so /health tells the operator
    what to redeploy."""
    path = write_model(tmp_path, "tiny_arcface_double.onnx", DOUBLE, (1, 3, 112, 112))
    with pytest.raises(ValueError) as exc:
        ArcFaceEmbedder(path, "CPU")
    assert "tensor(double)" in str(exc.value)
    assert "tiny_arcface_double.onnx" in str(exc.value)


def test_a_grayscale_nchw_graph_is_refused_at_init(tmp_path):
    path = write_model(tmp_path, "tiny_arcface_gray.onnx", FLOAT, (1, 1, 112, 112))
    with pytest.raises(ValueError) as exc:
        ArcFaceEmbedder(path, "CPU")
    assert "3-channel RGB" in str(exc.value)


def test_a_grayscale_nhwc_graph_is_refused_at_init(tmp_path):
    """The recheck's residual: NHWC-gray passed the NCHW-only channel check,
    built a session, and then 500'd every request behind ok:true. Both
    layouts now get the channel guard."""
    path = write_model(tmp_path, "tiny_arcface_nhwc_gray.onnx", FLOAT, (1, 112, 112, 1))
    with pytest.raises(ValueError) as exc:
        ArcFaceEmbedder(path, "CPU")
    assert "3-channel RGB" in str(exc.value)


def test_wrong_static_spatial_dims_are_refused_at_init(tmp_path):
    """ORT checks dimensions at run() time, not session creation: a 224x224
    export was the last green-health/always-500 shape left open. Alignment
    produces 112x112 crops, so a static mismatch refuses at init."""
    path = write_model(tmp_path, "tiny_arcface_224.onnx", FLOAT, (1, 3, 224, 224))
    with pytest.raises(ValueError) as exc:
        ArcFaceEmbedder(path, "CPU")
    assert "224" in str(exc.value)
    assert "112x112" in str(exc.value)


def test_build_embedder_routes_unknown_names_to_this_family(tmp_path):
    path = write_model(tmp_path, "tiny_w600k_r50.onnx", FLOAT, (1, 3, 112, 112))
    assert spec_for(path) == {"family": "arcface", "dim": None}
    assert isinstance(build_embedder(path, "CPU"), ArcFaceEmbedder)


def test_malformed_landmark_nestings_raise_like_the_sface_path(tmp_path):
    """np.reshape used to re-pair ANY ten floats into wrong (x, y) points and
    embed the garbage crop with 200 OK; the shared guard makes both families
    answer with the same ValueError sentence."""
    emb = ArcFaceEmbedder(
        write_model(tmp_path, "tiny_arcface_nchw.onnx", FLOAT, (1, 3, 112, 112)), "CPU"
    )
    img = _frame()
    two_by_five = _face()
    two_by_five["landmarks"] = [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
    with pytest.raises(ValueError, match=r"five \[x, y\] pairs"):
        emb.embed(img, [two_by_five])
    keyless = _face()
    del keyless["landmarks"]
    with pytest.raises(ValueError, match=r"five \[x, y\] pairs"):
        emb.embed(img, [keyless])  # ValueError, never the old KeyError


def test_degenerate_landmarks_are_refused_in_embed(tmp_path):
    emb = ArcFaceEmbedder(
        write_model(tmp_path, "tiny_arcface_nchw.onnx", FLOAT, (1, 3, 112, 112)), "CPU"
    )
    coincident = _face()
    coincident["landmarks"] = [[50.0, 60.0]] * 5
    with pytest.raises(ValueError, match="degenerate landmarks"):
        emb.embed(_frame(), [coincident])


# ------------------------------------------------- EMBED_BATCH

def _faces(n):
    return [_face(offset=40.0 + 30.0 * i) for i in range(n)]


def test_batching_is_off_by_default_and_the_loop_is_unchanged(tmp_path):
    """OFF is today: no EMBED_BATCH, per-face runs, and the vectors are the
    exact per-face blobs (Flatten echoes its input)."""
    emb = ArcFaceEmbedder(
        write_model(tmp_path, "tiny_dyn.onnx", FLOAT, ("N", 3, 112, 112)), "CPU")
    assert emb.batch_requested is False and emb.batch_active is False
    img, faces = _frame(), _faces(3)
    embeddings, _ = emb.embed(img, faces)
    for got, face in zip(embeddings, faces, strict=True):
        np.testing.assert_array_equal(
            np.asarray(got, np.float32), _expected_blob(img, face).transpose(2, 0, 1).ravel())


def test_a_dynamic_batch_graph_runs_all_faces_in_one_run_same_vectors(tmp_path, monkeypatch):
    """EMBED_BATCH on a dynamic-batch graph: ONE session.run for the request,
    order preserved, and the same vectors the per-face loop gives."""
    import app.recognizer as rec

    path = write_model(tmp_path, "tiny_dyn.onnx", FLOAT, ("N", 3, 112, 112))
    per_face = ArcFaceEmbedder(path, "CPU", batch=False)
    batched = ArcFaceEmbedder(path, "CPU", batch=True)
    assert batched.batch_active is True
    runs = []
    real_run = batched._session.run
    monkeypatch.setattr(batched, "_session", type("S", (), {
        "run": lambda self, names, feeds: runs.append(next(iter(feeds.values())).shape)
        or real_run(names, feeds)})())
    img, faces = _frame(), _faces(5)
    want, want_norms, _, _, _ = per_face.embed_faces(img, faces)
    got, got_norms, _, _, _ = batched.embed_faces(img, faces)
    assert runs == [(5, 3, 112, 112)], "one run for the whole request"
    assert got == want and got_norms == want_norms
    # More faces than BATCH_MAX: chunks, still in order.
    monkeypatch.setattr(rec, "BATCH_MAX", 2)
    runs.clear()
    got, _, _, _, _ = batched.embed_faces(img, faces)
    assert runs == [(2, 3, 112, 112), (2, 3, 112, 112), (1, 3, 112, 112)]
    assert got == want


def test_a_static_batch_graph_keeps_the_per_face_loop(tmp_path):
    """A graph exported with batch 1 cannot take five faces: batching stays
    requested but NOT active, and /health's knobs say so."""
    emb = ArcFaceEmbedder(
        write_model(tmp_path, "tiny_static.onnx", FLOAT, (1, 3, 112, 112)), "CPU", batch=True)
    assert emb.batch_requested is True and emb.batch_active is False
    img, faces = _frame(), _faces(3)
    embeddings, _ = emb.embed(img, faces)
    assert len(embeddings) == 3


def test_a_malformed_face_is_still_a_valueerror_when_batched(tmp_path):
    """Every blob is built before the first run, so the 400 contract holds."""
    emb = ArcFaceEmbedder(
        write_model(tmp_path, "tiny_dyn.onnx", FLOAT, ("N", 3, 112, 112)), "CPU", batch=True)
    bad = {"box": {"x": 1, "y": 1, "w": 5, "h": 5}, "landmarks": [[1, 2]] * 4}
    with pytest.raises(ValueError):
        emb.embed(_frame(), [_face(), bad])
