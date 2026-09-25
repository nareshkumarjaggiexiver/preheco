"""The head-covering reader (app/headwear.py) and POST /headwear.

Three promises, each pinned:

* OFF IS TODAY. EMBED_HEADWEAR_MODEL unset: no session, /health and /embed
  carry exactly the keys they always did, and POST /headwear answers what an
  unknown route answers.
* THE PREPROCESSING IS THE CONTRACT. The evaluation measured that another
  resize moves calls (PIL bicubic for cv2.INTER_AREA: 16 of 461, one of them
  wrong), so the views and the pixels are compared, EXACTLY, with
  tests/headwear_ref.py — the reference reader, verbatim — on synthetic
  frames that exercise every clamp and the half-to-even rounding; the pixels
  are also pinned by a digest the ORIGINAL reference produced; and the whole
  read runs through a real (tiny) ONNX session against the reference
  reader's own read().
* A NAMED SET THAT WILL NOT LOAD IS SERVED, NOT SWALLOWED, and a mismatched
  set (another graph, another text bank, classes out of order) is refused.

The real SigLIP graph (372 MB, never committed) is exercised only when its
directory is given (EMBED_HEADWEAR_TEST_DIR, or services/embed/models/).
"""

import base64
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from headwear_frames import CASES, frame
from tiny_onnx import FLOAT, embedding_model, write_model

import app.headwear as hw
import app.main as app_main
from app.headwear import (
    CLASSES,
    HeadwearReader,
    head_view,
    headwear_model_from_env,
    pixel_values,
)

REF_SHA = "5c96f2f974baffdc897db642343af57baa32c082efe3de231b275b4a97b918e6"
#: sha256 of the ORIGINAL reference reader's pixel_values (float32, little
#: endian) over every face of headwear_frames.CASES — numpy 2.5.3 and 2.5.1,
#: opencv 5.0.0, identical.
PIXELS_SHA = "68c6223b7c75f0169b11d2f56ba5ae6f262c45a79af0e07e564697964f5a6ea6"
#: The real prompt set's views, as its JSON states them.
VIEWS = {
    "loose": {"side_face_widths": 0.3, "up_face_heights": 1.0, "mask": None},
    "tight": {"side_face_widths": 0.15, "up_face_heights": 0.8, "mask": "ellipse"},
}
ONNX = "siglip_b16_224_image_fp32.onnx"
DIM = 8
PROMPT_CLASSES = ["turban", "turban", "bare", "bare", "dupatta_or_scarf", "cap_or_hat"]


def _reference():
    """The verbatim reference module (tests/headwear_ref.py), its bytes checked."""
    path = Path(__file__).resolve().parent / "headwear_ref.py"
    body = path.read_bytes().split(b"\n", 1)[1]  # line 1 is the noqa banner
    assert hashlib.sha256(body).hexdigest() == REF_SHA, "headwear_ref.py is not verbatim"
    import headwear_ref

    return headwear_ref


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_set(folder: Path, *, classes=CLASSES, onnx_sha=None, text_sha=None) -> Path:
    """A complete tiny set: graph, text bank, prompt JSON. Returns the graph path."""
    folder.mkdir(parents=True, exist_ok=True)
    w = np.array([[np.cos(1.3 * c + 0.7 * d) for d in range(DIM)] for c in range(3)], "<f4")
    graph = folder / ONNX
    graph.write_bytes(embedding_model(w.tobytes(), DIM))
    t = np.array([[np.sin(0.9 * p + 1.1 * d + 0.3) for d in range(DIM)]
                  for p in range(len(PROMPT_CLASSES))], np.float32)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    np.save(folder / "headwear_text_embeds.npy", t)
    cfg = {
        "onnx": ONNX,
        "onnx_sha256": onnx_sha or _sha(graph),
        "text_embeds": "headwear_text_embeds.npy",
        "text_embeds_sha256": text_sha or _sha(folder / "headwear_text_embeds.npy"),
        "classes": list(classes),
        "prompts": [{"text": f"p{i}", "class": c} for i, c in enumerate(PROMPT_CLASSES)],
        "logit_scale": 117.33076477050781,
        "logit_bias": -12.9324369430542,
        "views": VIEWS,
        "preprocessing": {"input": "pixel_values", "output": "image_embeds"},
        "thresholds": {"turban": 0.8, "bare": 0.5},
    }
    (folder / "headwear_prompts.json").write_text(json.dumps(cfg))
    return graph


def _b64_png(img) -> str:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ----------------------------------------------------------------- the switch


def test_the_reader_is_off_unless_the_variable_names_a_graph():
    """Unset, empty (compose's ${VAR-}) and `off` are all off; there is no
    default file that switches it on by being present."""
    assert headwear_model_from_env(None) is None
    assert headwear_model_from_env("") is None
    assert headwear_model_from_env("  ") is None
    assert headwear_model_from_env("off") is None and headwear_model_from_env("OFF") is None
    assert headwear_model_from_env("models/x.onnx") == Path("models/x.onnx")


# ----------------------------------------------------- the preprocessing contract


def _faces():
    for h, w, k, boxes in CASES:
        img = frame(h, w, k)
        for box in boxes:
            yield img, box


def test_views_and_pixels_are_the_reference_readers_exactly():
    """Every clamp, the half-to-even edge, the ellipse, the letterbox, the
    INTER_AREA resize and the scaling: array-equal to the reference."""
    ref = _reference()
    for img, box in _faces():
        for spec in VIEWS.values():
            side, up = spec["side_face_widths"], spec["up_face_heights"]
            ell = spec["mask"] == "ellipse"
            mine = head_view(img, box, side, up, ell)
            theirs = ref.head_view(img, box, side, up, ell)
            assert mine.shape == theirs.shape and np.array_equal(mine, theirs), box
        views = [head_view(img, box, s["side_face_widths"], s["up_face_heights"],
                           s["mask"] == "ellipse") for s in VIEWS.values()]
        assert np.array_equal(pixel_values(views), ref.preprocess(views))


def test_the_pixels_are_pinned_by_the_original_references_digest():
    """An OpenCV or numpy change that moves one pixel fails here."""
    digest = hashlib.sha256()
    for img, box in _faces():
        views = [head_view(img, box, s["side_face_widths"], s["up_face_heights"],
                           s["mask"] == "ellipse") for s in VIEWS.values()]
        digest.update(np.ascontiguousarray(pixel_values(views), dtype="<f4").tobytes())
    assert digest.hexdigest() == PIXELS_SHA


def test_the_half_to_even_edge_is_the_references():
    """x - 0.3w = 62.5 rounds to 62 (Python's round), not 63."""
    box = {"x": 100.0, "y": 150.0, "w": 125.0, "h": 150.0}
    assert hw.head_box(box, 320, 320, 0.3, 1.0)[0] == 62


def test_a_read_through_a_real_session_is_the_reference_readers(tmp_path):
    """The tiny tower (mean RGB @ W) through ORT: same logits as the
    reference reader's own read(), loose view then tight, per face."""
    ref = _reference()
    write_set(tmp_path)
    reader = HeadwearReader(tmp_path / ONNX, "CPU")
    theirs_reader = ref.HeadwearReader(tmp_path, threads=1)
    for h, w, k, boxes in CASES:
        img = frame(h, w, k)
        mine = reader.read(img, boxes)
        theirs = theirs_reader.read(img, boxes)
        assert len(mine) == len(boxes)
        for m, t in zip(mine, theirs, strict=True):
            expected = t["views"]["loose"]["logits"] + t["views"]["tight"]["logits"]
            assert np.allclose(m, expected, atol=1e-5), (m, expected)
    prompts_sha = _sha(tmp_path / "headwear_prompts.json")
    assert reader.stamp == f"{_sha(tmp_path / ONNX)[:12]}+{prompts_sha[:12]}"
    assert reader.providers_active == ["CPUExecutionProvider"] and reader.trt is None


def test_a_box_outside_the_image_reads_null_and_a_bad_box_is_refused(tmp_path):
    write_set(tmp_path)
    reader = HeadwearReader(tmp_path / ONNX, "CPU")
    img = frame(200, 200, 0)
    got = reader.read(img, [{"x": 50, "y": 60, "w": 40, "h": 50},
                            {"x": 500, "y": 500, "w": 40, "h": 50}])
    assert len(got[0]) == 8 and got[1] is None, "absent is not zero"
    for bad in ({"x": 1, "y": 2}, {"x": 1, "y": 2, "w": 0, "h": 5},
                {"x": "a", "y": 2, "w": 3, "h": 4}, None):
        with pytest.raises(ValueError, match="face.box"):
            reader.read(img, [bad])


# ------------------------------------------------------------- a mismatched set


@pytest.mark.parametrize("damage, match", [
    ({"onnx_sha": "0" * 64}, "another export"),
    ({"text_sha": "0" * 64}, "another prompt set"),
    ({"classes": ("bare", "turban", "dupatta_or_scarf", "cap_or_hat")}, "in that order"),
])
def test_a_mismatched_set_is_refused_by_name(tmp_path, damage, match):
    write_set(tmp_path, **damage)
    with pytest.raises(ValueError, match=match):
        HeadwearReader(tmp_path / ONNX, "CPU")


def test_a_missing_graph_or_prompt_set_names_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="EMBED_HEADWEAR_MODEL"):
        HeadwearReader(tmp_path / ONNX, "CPU")
    write_set(tmp_path)
    (tmp_path / "headwear_prompts.json").unlink()
    with pytest.raises(FileNotFoundError, match="headwear_prompts.json"):
        HeadwearReader(tmp_path / ONNX, "CPU")


# ------------------------------------------------------------------ the service


def _headwear_at(monkeypatch, path):
    monkeypatch.setattr(app_main, "HEADWEAR_MODEL_PATH", path)
    monkeypatch.setattr(app_main, "_headwear", None)
    monkeypatch.setattr(app_main, "_headwear_error", None)
    monkeypatch.setattr(app_main, "_headwear_failed_stat", None)


@pytest.fixture()
def arcface(tmp_path, monkeypatch):
    """A tiny arcface embedder, the attribute pass off — the rest of embed as always."""
    from app.recognizer import ArcFaceEmbedder

    path = write_model(tmp_path, "tiny_arcface_nchw.onnx", FLOAT, (1, 3, 112, 112))
    monkeypatch.setattr(app_main, "MODEL_PATH", path)
    monkeypatch.setattr(app_main, "_embedder", None)
    monkeypatch.setattr(app_main, "_load_error", None)
    monkeypatch.setattr(app_main, "_failed_stat", None)
    monkeypatch.setattr(app_main, "build_embedder", lambda: ArcFaceEmbedder(path, "CPU"))
    monkeypatch.setattr(app_main, "ATTR_MODEL_PATH", None)
    monkeypatch.setattr(app_main, "ATTR_MODEL_EXPLICIT", True)
    monkeypatch.setattr(app_main, "_attributes", None)
    monkeypatch.setattr(app_main, "_attr_error", None)
    return TestClient(app_main.app)


FACE = {"box": {"x": 60.0, "y": 50.0, "w": 60.0, "h": 78.0},
        "landmarks": [[75, 75], [105, 75], [90, 92], [78, 110], [102, 110]], "conf": 0.9}


def test_off_is_today_no_new_keys_and_the_route_is_not_there(arcface, monkeypatch):
    """/health and /embed keep exactly their keys; POST /headwear answers what
    an unknown route answers, byte for byte."""
    _headwear_at(monkeypatch, None)
    health = arcface.get("/health").json()
    assert set(health) == {"ok", "model", "version", "attrModel", "attrError", "error",
                           "device", "knobs"}
    img = frame(240, 320, 0)
    emb = arcface.post("/embed", json={"imageB64": _b64_png(img), "faces": [FACE]}).json()
    assert set(emb) == {"embeddings", "alignMs", "norms", "attributes", "attrMs", "balance"}
    route = arcface.post("/headwear", json={"imageB64": _b64_png(img), "faces": []})
    unknown = arcface.post("/no-such-route", json={"imageB64": _b64_png(img), "faces": []})
    assert (route.status_code, route.content) == (unknown.status_code, unknown.content) == (
        404, b'{"detail":"Not Found"}')


def test_on_health_serves_the_readers_block_and_headwear_reads(arcface, monkeypatch, tmp_path):
    """The block: graph, stamp, no error, the reader's own device truth; the
    reply: one 8-float reading per face, the stamp, the time."""
    _headwear_at(monkeypatch, write_set(tmp_path / "set"))
    health = arcface.get("/health").json()
    block = health["headwear"]
    assert block["model"] == ONNX and block["error"] is None
    assert block["stamp"] == app_main._headwear.stamp
    assert block["device"] == {"requested": "CPU", "active": ["CPUExecutionProvider"]}
    img = frame(240, 320, 1)
    faces = [{"box": {"x": 120.0, "y": 110.0, "w": 60.0, "h": 78.0}},
             {"box": {"x": 900.0, "y": 900.0, "w": 60.0, "h": 78.0}}]
    r = arcface.post("/headwear", json={"imageB64": _b64_png(img), "faces": faces})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"readings", "model", "ms"} and body["model"] == block["stamp"]
    assert len(body["readings"][0]) == 8 and body["readings"][1] is None
    expected = app_main._headwear.read(img, [f["box"] for f in faces])
    assert body["readings"][0] == pytest.approx(expected[0], abs=1e-6)
    assert arcface.post("/headwear", json={"imageB64": "@@@", "faces": []}).status_code == 400
    bad = arcface.post("/headwear", json={"imageB64": _b64_png(img), "faces": [{"box": {"x": 1}}]})
    assert bad.status_code == 400 and "face.box" in bad.json()["detail"]


def test_a_crop_must_cover_what_it_was_cut_for(arcface, monkeypatch, tmp_path):
    """Cut too tight away from the frame's edge: 400. At the frame's edge the
    region clamps like the reference's does: read. A crop that does not fit
    its frame: 400."""
    _headwear_at(monkeypatch, write_set(tmp_path / "set"))
    img = frame(200, 160, 2)
    box = {"x": 50.0, "y": 100.0, "w": 60.0, "h": 78.0}  # the loose view needs x >= 32, y >= 22
    ok = arcface.post("/headwear", json={
        "imageB64": _b64_png(img), "faces": [{"box": box}],
        "crop": {"x": 1000, "y": 600, "frameW": 3840, "frameH": 2160}})
    assert ok.status_code == 200 and ok.json()["readings"][0] is not None
    tight = {**box, "x": 10.0}  # needs 8 px left of the image, which the cut dropped
    r = arcface.post("/headwear", json={
        "imageB64": _b64_png(img), "faces": [{"box": tight}],
        "crop": {"x": 1000, "y": 600, "frameW": 3840, "frameH": 2160}})
    assert r.status_code == 400 and "does not cover the loose head region" in r.json()["detail"]
    at_edge = arcface.post("/headwear", json={
        "imageB64": _b64_png(img), "faces": [{"box": tight}],
        "crop": {"x": 0, "y": 600, "frameW": 3840, "frameH": 2160}})
    assert at_edge.status_code == 200, "the frame's own edge clamps, as the reference does"
    r = arcface.post("/headwear", json={
        "imageB64": _b64_png(img), "faces": [{"box": box}],
        "crop": {"x": 3800, "y": 0, "frameW": 3840, "frameH": 2160}})
    assert r.status_code == 400 and "does not fit" in r.json()["detail"]


def test_a_named_set_that_will_not_load_is_served_then_retried(arcface, monkeypatch, tmp_path):
    """ok stays true (the count does not stop for an advisory read), the
    reason is on /health and on a 503, and delivering the files loads it."""
    graph = tmp_path / "late" / ONNX
    _headwear_at(monkeypatch, graph)
    calls = []
    real = hw.HeadwearReader

    def build(path, device):
        calls.append(1)
        return real(path, "CPU")

    monkeypatch.setattr(app_main, "build_headwear", build)
    health = arcface.get("/health").json()
    assert health["ok"] is True and health["headwear"]["model"] is None
    assert "FileNotFoundError" in health["headwear"]["error"]
    arcface.get("/health")
    assert len(calls) == 1, "same absent files — memoized, no load storm"
    r = arcface.post("/headwear", json={"imageB64": _b64_png(frame(50, 50, 0)), "faces": []})
    assert r.status_code == 503 and "FileNotFoundError" in r.json()["detail"]
    write_set(graph.parent)
    assert arcface.get("/health").json()["headwear"]["model"] == ONNX
    assert len(calls) == 2, "changed files — exactly one retry"


# ------------------------------------------------------------ TensorRT load path


def test_under_trt_the_truth_is_read_after_the_warm_up_and_the_engine_is_keyed(
    monkeypatch, tmp_path
):
    """A failed engine build drops ORT to CUDA inside the first run: the
    reader must say so; and the engine directory is the weights' hash."""
    from heco_common.ort import model_key

    graph = write_set(tmp_path / "set")
    seen = []

    class Session:
        def __init__(self, model, _opts, providers=None, provider_options=None):
            seen.append((model, provider_options[0]))
            self.ran = False

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.ran else [
                "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

        def get_provider_options(self):
            return {}

        def get_inputs(self):
            return [SimpleNamespace(name="pixel_values", type="tensor(float)",
                                    shape=["N", 3, 224, 224])]

        def get_outputs(self):
            return [SimpleNamespace(shape=["N", DIM])]

        def run(self, _names, feeds):
            self.ran = True
            return [np.zeros((next(iter(feeds.values())).shape[0], DIM), np.float32)]

    options = SimpleNamespace(add_session_config_entry=lambda k, v: None)
    fake = types.ModuleType("onnxruntime")
    fake.SessionOptions = lambda: options
    fake.InferenceSession = Session
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.setenv("HECO_TRT_CACHE", str(tmp_path / "cache"))
    reader = HeadwearReader(graph, "TRT")
    assert reader.providers_active == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert reader.trt is None, "TensorRT did not come up — never claimed"
    (_, opts), = seen
    assert opts["trt_engine_cache_path"] == str(tmp_path / "cache" / model_key(graph))
    assert opts["trt_profile_opt_shapes"] == "pixel_values:2x3x224x224"
    assert opts["trt_profile_max_shapes"] == "pixel_values:16x3x224x224"


# ---------------------------------------------------------------- the real graph

REAL_DIR = Path(os.environ.get("EMBED_HEADWEAR_TEST_DIR")
                or Path(__file__).resolve().parent.parent / "models")
EVAL_DIR = os.environ.get("EMBED_HEADWEAR_EVAL_DIR")


@pytest.mark.skipif(not (REAL_DIR / ONNX).is_file(),
                    reason=f"{ONNX} absent (set EMBED_HEADWEAR_TEST_DIR)")
def test_the_real_graph_reproduces_the_reference_reader():
    """The 372 MB graph, CPU: the same logits as the reference reader on the
    synthetic faces; with EMBED_HEADWEAR_EVAL_DIR (the evaluation's eval/),
    the same CALL as the evaluation recorded on a sample of its labelled
    context crops, read the way the evaluation read them."""
    ref = _reference()
    reader = HeadwearReader(REAL_DIR / ONNX, "CPU")
    theirs = ref.HeadwearReader(REAL_DIR, threads=4)
    for h, w, k, boxes in CASES:
        img = frame(h, w, k)
        for m, t in zip(reader.read(img, boxes), theirs.read(img, boxes), strict=True):
            assert np.allclose(m, t["views"]["loose"]["logits"] + t["views"]["tight"]["logits"],
                               atol=1e-3)
    if not EVAL_DIR:
        return
    meta = json.loads((Path(EVAL_DIR) / "context_meta.json").read_text())
    calls = json.loads((Path(EVAL_DIR) / "reference_calls.json").read_text())
    sample = {c: sorted(k for k, v in calls.items() if v == c)[:8]
              for c in ("turban", "bare", "unsure")}
    for want, ids in sample.items():
        for cid in ids:
            img = cv2.imread(str(Path(EVAL_DIR) / "context" / f"{cid}.png"))
            x, y, w, h = meta[cid]["face"]
            (logits,) = reader.read(img, [{"x": x, "y": y, "w": w, "h": h}])
            p = np.asarray(logits, np.float32).reshape(2, 4)
            p = np.exp(p - p.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            assert hw._call(p) == want, cid
