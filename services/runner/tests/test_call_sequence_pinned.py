"""What the loop does with every lever OFF, pinned before the levers exist.

THE CLAIM EVERY THROUGHPUT LEVER RESTS ON.  The levers of 2026-09-24 (the
detect worker, parallel detection, the face-search cadence, the face search
region) reorganise WHERE and WHEN the stage calls happen.  Each one defaults
off, and off has to mean today — not "the same totals", which two
compensating mistakes also produce, but the same calls, in the same order,
with the same bodies, reasoning the same way on every frame.

So this file does not describe the loop; it REMEMBERS it.  The fixture beside
it (fixtures/call_sequence_pinned.json) was captured from the runner as it
stood at 421210b, before any lever was written, over scripted footage that
reaches every call site the levers touch: the person search, the detections
zones, the tracker, the crop list and its re-verify gate, the whole-frame
switch, the face search, the zero-kept path, the embedder, the matcher and
the co-presence splits.  Each scenario pins four things:

* every stage request, in order, with its body (the frame travels as the
  index of the scene it carries, so a body is readable and still exact);
* the decision ledger, frame by frame, minus its wall times;
* the settled status and the planner's permanent record (config, notes,
  results) — so an OFF lever cannot even add a key to the run row;
* for the prefetcher, which overlaps ingest with the loop by design, the
  per-host call sequences (ingest's position among them is a race, its
  content is not).

A failure here is a lever that was supposed to be off changing the run.  The
fixture must never be regenerated to make it pass: re-capturing it is a
decision that today's behaviour has changed, and that belongs in its own
commit with its own reason.
"""

import hashlib
import json
from pathlib import Path

import httpx
from app.config import Settings

from tests.test_loop_v1 import make_loop, scripted_verdict
from tests.test_presence import A_PRIME, FA, FB, A, B, Scene, scene_index

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "call_sequence_pinned.json"

#: The services whose calls are pinned.  The planner is left out on purpose:
#: its tap rounds are scheduled by a duty guard that reasons from MEASURED
#: wall time, so their count is a property of the machine, not of the code.
#: /health probes are left out too — they run on a thread pool, in any order.
STAGE_HOSTS = ("ingest", "persons", "tracker", "faces", "embed", "match")

#: A face in a third body, for the scenes that need a newcomer.
C = {"x": 60, "y": 20, "w": 40, "h": 100, "conf": 0.9}
FC = {"x": 62, "y": 25, "w": 56, "h": 38}

#: One evening in eight frames: a guest minted then bound, a newcomer beside
#: her back-turned body (track presence), both faces together (co-presence),
#: an empty frame, a contested pair of boxes, a third guest, and a frame
#: whose only face is too small for the gate (the zero-kept path).
NIGHT = [
    {"boxes": [A], "faces": [FA]},
    {"boxes": [A], "faces": [FA]},
    {"boxes": [A, B], "faces": [FB]},
    {"boxes": [A, B], "faces": [FA, FB]},
    {"boxes": [], "faces": []},
    {"boxes": [A, A_PRIME, B], "faces": [FA, FB]},
    {"boxes": [A, C, B], "faces": [FA, FC, FB]},
    {"boxes": [A, B], "faces": [{"x": 12, "y": 25, "w": 30, "h": 20}]},
]
NIGHT_SCRIPT = [
    scripted_verdict("p00001", True, None),
    scripted_verdict("p00001", False, 0.70),
    scripted_verdict("p00002", True, None),
    scripted_verdict("p00001", False, 0.71),
    scripted_verdict("p00002", False, 0.66),
    scripted_verdict("p00001", False, 0.72),
    scripted_verdict("p00002", False, 0.64),
    scripted_verdict("p00001", False, 0.69),
    scripted_verdict("p00003", True, 0.31),
    scripted_verdict("p00002", False, 0.61),
]

#: Operator zones on the 160x120 scene: a DETECTIONS zone over B's body (it
#: never reaches the tracker) and a faces zone over A's face.
ZONES = [
    {"label": "tv", "mode": "detections",
     "points": [[0.7, 0.3], [0.95, 0.3], [0.95, 0.9], [0.7, 0.9]]},
    {"label": "glass", "mode": "faces",
     "points": [[0.2, 0.3], [0.33, 0.3], [0.33, 0.45], [0.2, 0.45]]},
]

#: name -> (settings overrides, extra request fields).  Prefetch is OFF in
#: every exact scenario: with it on, ingest's GET races the loop by design.
SCENARIOS = {
    "crops": ({"frame_prefetch": False}, {}),
    "whole-frame": ({"frame_prefetch": False, "faces_whole_frame": True}, {}),
    "reverify": ({"frame_prefetch": False, "face_reverify_interval_s": 3600.0}, {}),
    "zones": ({"frame_prefetch": False}, {"exclusionZones": ZONES}),
}

STATUS_KEYS = (
    "state", "endReason", "error", "frames", "unique", "matches",
    "coPresenceSplits", "trackPresenceSplits", "sameFrameSplits",
    "lockedTrackFolds", "healedSplits", "distinctTracks",
    "faceSearchesSkipped", "excludedByZone", "personsZoned",
    "gatedByWidth", "gatedUnmeasured", "templatesEnrolled",
)


def _compact(value):
    """A body as the fixture stores it: exact, but not 128 floats per match.

    A long list (an embedding) becomes its length and a digest of its JSON —
    still an exact pin, at a fraction of the fixture's size.
    """
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 16:
            raw = json.dumps(value, sort_keys=True).encode()
            return {"len": len(value), "sha1": hashlib.sha1(raw).hexdigest()[:16]}
        return [_compact(v) for v in value]
    return value


class Recorder(Scene):
    """The scene fake, writing down every stage request it serves."""

    def __init__(self, frames, script):
        super().__init__(frames, script)
        self.stage_calls: list[list] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Record the request (frame as its scene index), then serve it as the scene does."""
        host, path = request.url.host, request.url.path
        if host in STAGE_HOSTS and path != "/health":
            body = json.loads(request.content) if request.content else None
            if isinstance(body, dict) and "imageB64" in body:
                body = {**body, "imageB64": f"#scene-{scene_index(body['imageB64'])}"}
            self.stage_calls.append([host, path] + ([_compact(body)] if body is not None else []))
        return super().handler(request)


def run_scenario(name: str, tmp_path: Path) -> dict:
    """One scripted run under one scenario; everything the pin compares."""
    settings, extra = SCENARIOS[name]
    golden = tmp_path / f"golden-{name}.jsonl"
    fake = Recorder(NIGHT, NIGHT_SCRIPT)
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}, **extra}
    final = make_loop(fake, request, golden_path=str(golden), **settings).run()
    ledger = [
        {k: v for k, v in json.loads(line).items() if k != "ms"}
        for line in golden.read_text().splitlines()
    ]
    record = {
        "calls": fake.stage_calls,
        "ledger": ledger,
        "status": {k: final.get(k) for k in STATUS_KEYS},
        "config": (fake.run_created or {}).get("config"),
        "ended": fake.run_ended,
    }
    # One JSON round trip, so tuples and lists compare as the fixture stores them.
    return json.loads(json.dumps(record))


def run_prefetched(tmp_path: Path) -> dict:
    """The crops scenario with the prefetcher ON (the shipped default)."""
    fake = Recorder(NIGHT, NIGHT_SCRIPT)
    golden = tmp_path / "golden-prefetch.jsonl"
    request = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}
    assert Settings().frame_prefetch is True, "the shipped default this pins"
    make_loop(fake, request, golden_path=str(golden)).run()
    per_host = {
        host: [c for c in fake.stage_calls if c[0] == host]
        for host in STAGE_HOSTS if host != "ingest"
    }
    ledger = [
        {k: v for k, v in json.loads(line).items() if k != "ms"}
        for line in golden.read_text().splitlines()
    ]
    return json.loads(json.dumps({"per_host": per_host, "ledger": ledger}))


def capture_all(tmp_path: Path) -> dict:
    """Every scenario, as the fixture stores it."""
    out = {name: run_scenario(name, tmp_path) for name in SCENARIOS}
    out["prefetch"] = run_prefetched(tmp_path)
    return out


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def test_the_fixture_reaches_every_call_site_the_levers_touch():
    """A pin over footage that never exercises a path pins nothing there."""
    fx = _fixture()
    crops = fx["crops"]
    hosts = {c[0] for c in crops["calls"]}
    assert hosts == set(STAGE_HOSTS)
    assert any(c[:2] == ["match", "/split"] for c in crops["calls"]), "co-presence"
    faces = [c[2] for c in crops["calls"] if c[:2] == ["faces", "/detect"]]
    assert all(isinstance(b["within"], list) for b in faces), "crops search person boxes"
    assert [] in [b["within"] for b in faces], "an empty frame searches nothing"
    whole = [c[2] for c in fx["whole-frame"]["calls"] if c[:2] == ["faces", "/detect"]]
    assert whole and all(b["within"] is None for b in whole), "whole frame sends None"
    assert fx["reverify"]["status"]["faceSearchesSkipped"] > 0, "the re-verify gate engaged"
    assert fx["zones"]["status"]["personsZoned"] > 0, "a detections zone ate a body"
    assert fx["zones"]["status"]["excludedByZone"] > 0, "a faces zone ate a face"
    assert crops["status"]["trackPresenceSplits"] > 0, "track presence asserted"
    assert any(not r["verdicts"] for r in crops["ledger"]), "the zero-kept path ran"


def test_every_scenario_replays_the_captured_run_exactly(tmp_path):
    """Levers off: the same calls, bodies, decisions and permanent record."""
    fx = _fixture()
    got = capture_all(tmp_path)
    for name in (*SCENARIOS, "prefetch"):
        want, have = fx[name], got[name]
        for part in want:
            assert have[part] == want[part], f"{name}: {part} moved"
