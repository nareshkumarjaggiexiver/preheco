"""The shared-frame transport through the throughput levers (runner side).

The levers of 2026-09-24 moved WHERE the detector calls are made — the detect
worker (HECO_PIPELINE_OVERLAP), side by side (HECO_PARALLEL_DETECT), the
operator's region (faceRegion), or not at all (HECO_FACE_CADENCE) — and every
one of those paths must carry the same picture the loop would have: the JPEG
and the ref, or under ref-only the ref alone.  A path that dropped the ref
would cost a decode; one that dropped it under ref-only would send a stage
no pixels at all.

Also pinned here: the ref-only probe's frame is the run's FIRST frame — a
live buffer (INGEST_BUFFER_S) dequeues on every GET, so a probe that threw
its frame away lost a frame — and the runner reads the ref for its own
descriptors, because a ref-only frame has no JPEG to decode.
"""

import base64
import json

import httpx
import numpy as np
import pytest
from app import loop as loop_mod
from heco_common import frameref

from tests.test_loop_v1 import make_loop
from tests.test_presence import RUN, Scene, scene_b64, scene_index
from tests.test_reverify_fixtures import SETTLING, settling_frames

N = 6


class Queued(Scene):
    """Ingest with the shared transport and a live buffer: each GET dequeues.

    Frame i is a real 160x120 picture written to the shared dir as ref
    ``f{i+1}_160x120.bgr``; ``?jpeg=0`` drops the JPEG as ingest does.  The
    stage fakes resolve a ref-only body back to its scene, and every stage
    body is recorded as (host, scene, carried ref, JPEG sent).
    """

    def __init__(self, frames_dir, frames, script):
        super().__init__(frames, script)
        self.queue = list(range(len(frames)))
        self.gets: list[str] = []
        self.refs = {}
        for i in range(len(frames)):
            img = np.full((120, 160, 3), 40 + 20 * i, np.uint8)
            img[20:120, 10:70] = (200, 60, 30)          # a blue garment on body A
            self.refs[i] = frameref.write_frame(img, i + 1, frames_dir)
        self.by_ref = {r: i for i, r in self.refs.items()}
        self.stage_bodies: list[tuple] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Serve the queue; resolve ref-only stage bodies; record them."""
        host, path = request.url.host, request.url.path
        if host == "ingest" and path == "/frame":
            self.gets.append(str(request.url.query, "ascii") if request.url.query else "")
            if not self.queue:
                return httpx.Response(200, json={"ended": True})
            i = self.queue.pop(0)
            jpeg = request.url.params.get("jpeg") != "0"
            return httpx.Response(200, json={
                "imageB64": scene_b64(i) if jpeg else "", "frameRef": self.refs[i],
                "tMs": i * 100, "w": 160, "h": 120, "seq": i + 1, "ended": False,
            })
        if host in ("persons", "faces", "embed") and path in ("/detect", "/embed"):
            body = json.loads(request.content)
            ref = body.get("frameRef")
            if not body.get("imageB64"):
                if ref not in self.by_ref:
                    return httpx.Response(400, json={"detail": f"cannot read {ref!r}"})
                body["imageB64"] = scene_b64(self.by_ref[ref])
            self.stage_bodies.append(
                (host, scene_index(body["imageB64"]), ref,
                 bool(json.loads(request.content).get("imageB64")))
            )
            request = httpx.Request(
                request.method, request.url, headers=request.headers,
                content=json.dumps(body).encode(),
            )
        return super().handler(request)


@pytest.fixture()
def frames_dir(tmp_path, monkeypatch):
    """The shared mount, as the runner container sees it."""
    monkeypatch.setenv("HECO_FRAMES_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture()
def runner_reads(monkeypatch):
    """Every ref the RUNNER itself read (its descriptors), in order."""
    seen: list[str] = []
    real = frameref.read_frame

    def spy(ref, directory=None):
        seen.append(ref)
        return real(ref, directory)

    monkeypatch.setattr(loop_mod.frameref, "read_frame", spy)
    return seen


#: Each lever's detection path, as settings (and a request body change).
PATHS = {
    "serial crops": ({}, RUN),
    "serial whole frame": ({"faces_whole_frame": True}, RUN),
    "overlap": ({"pipeline_overlap": True, "faces_whole_frame": True}, RUN),
    "overlap + parallel": (
        {"pipeline_overlap": True, "parallel_detect": True, "faces_whole_frame": True}, RUN,
    ),
    "region": ({"pipeline_overlap": True}, {**RUN, "faceRegion": {
        "x": 0.0, "y": 0.0, "w": 0.6, "h": 0.6}}),
    "cadence": ({"face_cadence": True, "face_reverify_interval_s": 3.0,
                 "face_cadence_max_gap_s": 0.25, "faces_whole_frame": True}, RUN),
}


@pytest.mark.parametrize("name", list(PATHS))
def test_every_detection_path_carries_the_ref_and_no_frame_is_lost(
    name, frames_dir, runner_reads
):
    """Ref-only on, through each lever's path: every frame processed once, in
    order (the probe's included), every stage body carrying that frame's ref,
    and the JPEG dropped after the probe proved everyone can read the ref."""
    settings, request = PATHS[name]
    fake = Queued(frames_dir, settling_frames(N), SETTLING)
    final = make_loop(fake, request, frames_ref_only=True, **settings).run()

    assert final["frames"] == N, "the probe's frame was processed, not thrown away"
    assert fake.gets[0] == "" and set(fake.gets[1:]) == {"jpeg=0"}
    probes, work = fake.stage_bodies[:3], fake.stage_bodies[3:]
    assert [(h, sc, sent) for h, sc, _r, sent in probes] == [
        ("persons", 0, False), ("faces", 0, False), ("embed", 0, False),
    ], "the probe asked each stage for the ref alone"
    persons = [scene for host, scene, *_ in work if host == "persons"]
    assert persons == list(range(N)), "each frame once, in order"
    for host, scene, ref, sent_jpeg in work:
        assert ref == fake.refs[scene], f"{host} got frame {scene} without its ref"
        assert sent_jpeg == (scene == 0), "the JPEG only on the probe's own frame"
    faced = {scene for host, scene, *_ in work if host == "embed"}
    assert faced and all(fake.refs[s] in runner_reads for s in faced if s > 0), (
        "the runner cut its descriptors from the ref: there was no JPEG"
    )
    matches = [b for b in fake.match_bodies if b.get("appearance") is not None]
    assert matches, "the torso was measured from the ref under ref-only"


def test_with_the_jpeg_alongside_every_body_carries_both(frames_dir, runner_reads):
    """ref-only off: the ref still rides beside the JPEG on every path."""
    fake = Queued(frames_dir, settling_frames(N), SETTLING)
    final = make_loop(fake, RUN, pipeline_overlap=True, faces_whole_frame=True).run()
    assert final["frames"] == N
    assert set(fake.gets) == {""}, "nobody asked ingest to drop the JPEG"
    assert fake.stage_bodies and all(
        ref == fake.refs[scene] and sent_jpeg
        for _host, scene, ref, sent_jpeg in fake.stage_bodies
    )
    assert runner_reads == [], "with a JPEG in hand the runner decodes it, as before"


def test_a_runner_that_cannot_read_the_ref_keeps_the_jpegs(
    frames_dir, runner_reads, monkeypatch
):
    """The stages could read the ref but the runner could not (no mount on
    it): ref-only would have silently cost every torso, head and beard
    reading and every face card — so the run keeps its JPEGs instead."""
    monkeypatch.setattr(loop_mod.frameref, "read_frame", lambda ref, directory=None: None)
    fake = Queued(frames_dir, settling_frames(N), SETTLING)
    final = make_loop(fake, RUN, frames_ref_only=True).run()
    assert final["frames"] == N, "and it still loses no frame"
    assert set(fake.gets) == {""}
    assert fake.stage_bodies and all(sent_jpeg for *_, sent_jpeg in fake.stage_bodies), (
        "no stage was probed or sent a ref-only body: the runner's own check came first"
    )


def test_no_ref_on_the_wire_means_no_ref_in_any_body():
    """A stack without the shared transport sends exactly the bodies it
    always sent (the pinned call sequence proves the rest)."""
    frame = {"imageB64": base64.b64encode(b"x").decode(), "frameRef": None}
    assert loop_mod.RunLoop._pixels(frame) == {"imageB64": frame["imageB64"]}
    assert loop_mod.RunLoop._pixels({**frame, "frameRef": "f3_2x2.bgr"}) == {
        "imageB64": frame["imageB64"], "frameRef": "f3_2x2.bgr",
    }
