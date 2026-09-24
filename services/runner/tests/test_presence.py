"""Track-presence co-presence and the embedder's riders (R1 + R2).

THE RUN THIS ANSWERS (f0bfc5, 2026-09-23, Punjab wedding-hall overview camera,
74 guests, 500 pairs in the review queue).  Pair #5 was p00048 (girl, yellow
top) against p00052 (woman, dark green dress) at face 0.338 — and p00048 was
IN THE SAME FRAME as p00052, with her back to the camera.  Face co-presence
needs two faces; the ledger held the proof and the loop could not use it.  The
reference algorithm (scratchpad replay.py) replays that ledger to 48 pairs
proven distinct against 25 from faces alone, and these tests pin its port.

Everything runs against the scripted V1 fake over httpx.MockTransport, with
per-frame scenes: each frame states its person boxes and its faces, the
tracker echoes boxes as tracks (id = position + 1, so a box that keeps its
slot keeps its id), and the match script replays exact verdicts.  Frames are
told apart by their payload, never by a call counter, so the pins hold with
the prefetcher on or off.
"""

import base64
import json

import httpx
import pytest
from app.config import Settings
from app.loop import RunLoop, parse_embed_reply

from tests.test_loop_v1 import V1Fake, make_loop, scripted_verdict

# Three bodies.  A and B never overlap; A' overlaps A at IoU 0.41 — over the
# 0.4 contest floor, which is the whole point of it.
A = {"x": 10, "y": 20, "w": 60, "h": 100, "conf": 0.9}
B = {"x": 100, "y": 20, "w": 60, "h": 100, "conf": 0.9}
A_PRIME = {"x": 35, "y": 20, "w": 60, "h": 100, "conf": 0.9}
# 60 px faces (over the 56 px width floor) whose centres sit inside A and B
# where a HEAD sits: a quarter of the way down the box, on its middle.  A
# face lower or wider off than 0.35 of the box does not bind (see
# RunLoop._head_sits_in and the mis-binding it exists for).
FA = {"x": 12, "y": 25, "w": 60, "h": 40}
FB = {"x": 102, "y": 25, "w": 60, "h": 40}
# A face between the bodies: no person box contains its centre.
F_NOBODY = {"x": 50, "y": 25, "w": 60, "h": 40}

RUN = {"eventId": "ev-1", "source": {"path": "/x.mp4"}}


def scene_b64(i: int) -> str:
    """A distinct opaque payload per frame — the fake's stand-in for pixels."""
    return base64.b64encode(f"scene-{i}".encode()).decode()


def scene_index(image_b64: str) -> int:
    """Which scene a stage was handed, read off the payload it received."""
    return int(base64.b64decode(image_b64).decode().split("-")[1])


class Scene(V1Fake):
    """Per-frame bodies and faces; the stages read WHICH frame off the payload.

    ``frames[i] = {"boxes": [...], "faces": [...]}``.  ``embed`` is an optional
    callable ``n_faces -> extra reply keys`` so a test can hand the loop norms
    and attributes (wire E1) or, by leaving it None, an older embed service
    that sends neither.
    """

    def __init__(self, frames: list[dict], match_script: list[dict], embed=None):
        super().__init__(n_frames=len(frames), match_script=list(match_script))
        self.frames = frames
        self.embed = embed

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Serve the scene's frame, bodies, faces and embed reply; defer the rest."""
        host, path = request.url.host, request.url.path
        if host == "ingest" and path == "/frame":
            i = min(self.frame_i, self.n_frames - 1)
            exhausted = self.frame_i >= self.n_frames
            if not exhausted:
                self.frame_i += 1
            return httpx.Response(200, json={
                "imageB64": scene_b64(i), "tMs": i * 100, "w": 160, "h": 120,
                "seq": i, "ended": exhausted,
            })
        body = json.loads(request.content) if request.content else {}
        if host == "persons" and path == "/detect":
            scene = self.frames[scene_index(body["imageB64"])]
            return httpx.Response(200, json={"boxes": [dict(b) for b in scene["boxes"]]})
        if host == "faces" and path == "/detect":
            scene = self.frames[scene_index(body["imageB64"])]
            return httpx.Response(200, json={"faces": [
                {"box": dict(f), "landmarks": [[1, 1]] * 5, "conf": 0.9,
                 "widthPx": float(f["w"]), "iedPx": 30.0, "frontality": 0.9,
                 "sharpness": 400.0}
                for f in scene["faces"]
            ]})
        if host == "embed" and path == "/embed":
            n = len(body["faces"])
            # Vector k is all k's, so a match body says which face it carried.
            reply = {"embeddings": [[float(k)] * 128 for k in range(n)]}
            if self.embed is not None:
                reply.update(self.embed(n))
            return httpx.Response(200, json=reply)
        return super().handler(request)


def run_scene(frames, script, embed=None, **settings_kw):
    """Run a scene to the end; returns (fake, final status)."""
    fake = Scene(frames, script, embed=embed)
    final = make_loop(fake, RUN, **settings_kw).run()
    return fake, final


# A guest seen alone, comfortably, then standing back-turned beside a newcomer.
MINT_THEN_BIND = [
    {"boxes": [A], "faces": [FA]},  # frame 0: minted (a mint never binds)
    {"boxes": [A], "faces": [FA]},  # frame 1: matched at 0.70 -> track 1 bound
]
BACK_TURNED_BESIDE_NEWCOMER = {"boxes": [A, B], "faces": [FB]}  # no face on A


# ------------------------------------------------------- track presence


def test_a_back_turned_guest_is_asserted_distinct_by_their_track():
    """THE REPLAY OF PAIR #5: p00048 back-turned in p00052's frame.

    Frame 2 holds two bodies and one face.  Face co-presence sees one key and
    says nothing; the track bound in frame 1 stands in for the other, and the
    pair goes through the same /split door — counted as a TRACK-presence
    split, because the evidence is a tracker's word rather than a second
    face, and never as a face co-presence split.
    """
    frames = MINT_THEN_BIND + [BACK_TURNED_BESIDE_NEWCOMER, BACK_TURNED_BESIDE_NEWCOMER]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("p00002", True, None),  # frame 2: the newcomer
        scripted_verdict("p00002", False, 0.66),  # frame 3: again
    ]
    fake, final = run_scene(frames, script, tap_interval_s=0.0, tap_duty_factor=0.0)

    assert fake.splits == [{"runId": "prun-1", "a": "p00001", "b": "p00002"}], (
        "one sorted pair, asserted once, on the track's word"
    )
    assert final["trackPresenceSplits"] == 1
    assert final["coPresenceSplits"] == 0, "no second face: not a face co-presence split"
    assert final["unique"] == 2, "an assertion never moves the count"
    # On the permanent record, beside the face counter.
    assert "trackPresenceSplits=1" in fake.run_ended["notes"]
    assert fake.run_ended["results"]["trackPresenceSplits"] == 1
    assert fake.run_ended["results"]["coPresenceSplits"] == 0
    # Published for the console like a face pair, so a stale banner retires.
    payload = [t["payload"] for t in fake.taps if t["stage"] == "match"][-1]
    assert payload["coPresent"] == [["p00001", "p00002"]]
    # And on the frame ledger, named for what it was.
    events = [e for r in fake.frame_records for e in r["events"]]
    assert "track-presence p00001 != p00002" in events
    assert not any(e.startswith("co-presence ") for e in events)


def test_the_off_switch_binds_nothing_and_asserts_nothing():
    """HECO_PRESENCE_SPLIT=0: the frame is exactly what it was before."""
    frames = MINT_THEN_BIND + [BACK_TURNED_BESIDE_NEWCOMER]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("p00002", True, None),
    ]
    fake, final = run_scene(frames, script, presence_split=0)

    assert fake.splits == []
    assert final["trackPresenceSplits"] == 0
    assert final["coPresenceSplits"] == 0
    assert final["unique"] == 2


def test_two_faces_stay_a_face_co_presence_split_even_with_a_track_bound():
    """Where a face was matched, the face speaks and the track says nothing.

    Frame 2 has BOTH faces.  The bound track over body A adds nothing (A
    carries a face verdict), so the pair is face evidence and is counted as
    such — the track counter must never claim a pair two faces proved.
    """
    frames = MINT_THEN_BIND + [{"boxes": [A, B], "faces": [FA, FB]}]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("p00002", True, None),
    ]
    fake, final = run_scene(frames, script)

    assert fake.splits == [{"runId": "prun-1", "a": "p00001", "b": "p00002"}]
    assert final["coPresenceSplits"] == 1
    assert final["trackPresenceSplits"] == 0


def test_a_contested_track_never_binds():
    """Two tracks overlapping at IoU >= 0.4 is the shape of an identity swap.

    Frame 1 matches p00001 comfortably on a track that A' contests, so no
    binding is made; frame 2 then has A faceless beside a newcomer, which
    WOULD have produced a split had the track been bound.  None appears.
    """
    frames = [
        {"boxes": [A], "faces": [FA]},            # minted
        {"boxes": [A, A_PRIME], "faces": [FA]},   # matched 0.70, but contested
        BACK_TURNED_BESIDE_NEWCOMER,               # A faceless, B a newcomer
    ]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("p00002", True, None),
    ]
    fake, final = run_scene(frames, script)

    assert fake.splits == [], "a contested track is not evidence of anybody"
    assert final["trackPresenceSplits"] == 0


def test_a_contest_unbinds_a_track_that_was_bound():
    """A binding made in a clean frame drops the frame the box is contested.

    Frame 2 has no faces at all — the zero-kept path — and that is exactly
    where the unbind has to run: a contest seen only on frames with faces
    would let a swap pass unnoticed between two blinks of the detector.
    """
    frames = MINT_THEN_BIND + [
        {"boxes": [A, A_PRIME], "faces": []},    # contested, nobody's face seen
        BACK_TURNED_BESIDE_NEWCOMER,
    ]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("p00002", True, None),
    ]
    fake, final = run_scene(frames, script)

    assert fake.splits == []
    assert final["trackPresenceSplits"] == 0


def test_a_track_the_tracker_dropped_is_not_evidence_when_its_id_comes_back():
    """Absent from the tracker output = unbound, even if the id is reused.

    Frame 2 is an empty scene (no boxes, no tracks); frame 3 puts a body back
    in slot 1 — same id, and the fake tracker will happily reuse it — beside
    a newcomer.  The old binding must be gone.
    """
    frames = MINT_THEN_BIND + [
        {"boxes": [], "faces": []},
        BACK_TURNED_BESIDE_NEWCOMER,
    ]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("p00002", True, None),
    ]
    fake, final = run_scene(frames, script)

    assert fake.splits == []
    assert final["trackPresenceSplits"] == 0


def test_a_comfortable_match_to_somebody_else_rebinds_the_track():
    """Fresh evidence beats a stale claim.

    Track 1 binds to p00001, then matches p00003 at 0.70 and re-binds; the
    faceless frame after that asserts p00003 (not p00001) against the
    newcomer.
    """
    frames = MINT_THEN_BIND + [
        {"boxes": [A], "faces": [FA]},  # frame 2: the same track now says p00003
        BACK_TURNED_BESIDE_NEWCOMER,    # frame 3
    ]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("p00003", False, 0.70),
        scripted_verdict("p00002", True, None),
    ]
    fake, final = run_scene(frames, script)

    assert fake.splits == [{"runId": "prun-1", "a": "p00002", "b": "p00003"}]
    assert final["trackPresenceSplits"] == 1


def test_a_mint_and_an_impostor_range_match_never_bind():
    """A mint's cosine is a distance to OTHER people; 0.30 is inside the
    measured impostor range (ceiling 0.377).  Neither is a claim about whose
    track this is, so a faceless frame beside a newcomer asserts nothing."""
    frames = [
        {"boxes": [A], "faces": [FA]},   # minted
        {"boxes": [A], "faces": [FA]},   # matched, but at 0.30
        BACK_TURNED_BESIDE_NEWCOMER,
    ]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.30),
        scripted_verdict("p00002", True, None),
    ]
    fake, final = run_scene(frames, script)

    assert fake.splits == []
    assert final["trackPresenceSplits"] == 0


def test_a_pair_proven_by_faces_is_not_re_sent_on_the_tracks_word():
    """The memo is per pair, not per mechanism.

    Frame 1 proved the pair by two faces; frame 2 has both guests back-turned
    and both tracks bound, which is the same pair on weaker evidence — and it
    is never re-sent, exactly as a second face frame is not.
    """
    frames = [
        {"boxes": [A, B], "faces": [FA, FB]},  # both minted
        {"boxes": [A, B], "faces": [FA, FB]},  # both matched -> face split, both bound
        {"boxes": [A, B], "faces": []},        # both back-turned
    ]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00002", True, None),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("p00002", False, 0.70),
    ]
    fake, final = run_scene(frames, script)
    assert fake.splits == [{"runId": "prun-1", "a": "p00001", "b": "p00002"}]
    assert final["coPresenceSplits"] == 1
    assert final["trackPresenceSplits"] == 0


def test_a_refused_split_is_retried_on_a_faceless_frame_by_the_tracks_alone():
    """Two bound tracks with NO face in frame still carry the pair.

    A 5xx from the gallery leaves the pair unrecorded so it can be retried
    while the two share a frame (the existing rule).  Here the retry that
    lands is on a frame with no faces at all — both guests back-turned, both
    tracks bound — which is the zero-kept path asserting on the tracks' word
    alone, and it is counted as the track evidence it is.
    """
    class RefusesTwice(Scene):
        def handler(self, request):
            if request.url.host == "match" and request.url.path == "/split":
                self.split_status = 503 if len(self.splits) < 2 else None
            return super().handler(request)

    frames = [
        {"boxes": [A], "faces": [FA]},       # p00001 minted
        {"boxes": [A], "faces": [FA]},       # p00001 bound on track 1
        {"boxes": [A, B], "faces": [FB]},    # p00002 minted: attempt 1 -> 503
        {"boxes": [A, B], "faces": [FB]},    # p00002 bound on track 2: attempt 2 -> 503
        {"boxes": [A, B], "faces": []},      # both back-turned: attempt 3 lands
    ]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("p00002", True, None),
        scripted_verdict("p00002", False, 0.70),
    ]
    fake = RefusesTwice(frames, script)
    final = make_loop(fake, RUN).run()

    pair = {"runId": "prun-1", "a": "p00001", "b": "p00002"}
    assert fake.splits == [pair, pair, pair], "two refusals retried, the third landed"
    assert final["trackPresenceSplits"] == 1
    assert final["coPresenceSplits"] == 0
    assert "track-presence p00001 != p00002" in fake.frame_records[-1]["events"], (
        "the assertion is on the ledger of the faceless frame that carried it"
    )


# ------------------------------------------------------ the embed riders


# The double-boxed body: one person, two raw boxes.  UPPER is the detector's
# second box for A's own head-and-shoulders (IoU 0.33 with A: under the 0.4
# contest floor, as 10 of the 12 measured cases on f0bfc5 were).  FA's centre
# is inside both and nearer UPPER's centre, so both the person-box rule and
# the track rule hand the face to UPPER while A keeps its own track.
UPPER = {"x": 15, "y": 20, "w": 50, "h": 40, "conf": 0.8}
# The neighbour: a shorter, further-away body whose box overlaps A's head
# region (IoU 0.24 with A).  FA's centre lies inside it too, nearer its
# centre than A's — and 64% of the way down it, where no head sits.
BEHIND = {"x": 30, "y": 0, "w": 60, "h": 70, "conf": 0.9}
BEHIND_MOVED = {"x": 90, "y": 0, "w": 60, "h": 70, "conf": 0.9}


def test_one_body_double_boxed_never_asserts_against_itself():
    """THE BLOCKER: a full box and an upper box for ONE person are one body.

    Frame 2 detects A twice.  The face lands on UPPER (nearest centre), A's
    own track stays bound to p00001 on the full box, and the face — a fresh
    mint at cosine under the threshold — says p00002.  Two tokens for one
    body, and without the guard that is a permanent cannot_link between one
    person's two keys, hidden from the review queue and unmergeable.  A
    bound track whose box CONTAINS a keyed face's centre says nothing.
    """
    frames = MINT_THEN_BIND + [{"boxes": [A, UPPER], "faces": [FA]}]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("p00002", True, 0.30),  # the split mint, on UPPER
    ]
    fake, final = run_scene(frames, script)
    assert fake.splits == []
    assert final["trackPresenceSplits"] == 0
    assert final["unique"] == 2, "the mint itself is not this guard's business"


def test_a_face_binds_only_where_a_head_sits_in_the_box():
    """THE seq-1012 CASE: the nearest-centre tie rule binds the neighbour's track.

    A's face is inside both A and BEHIND and nearer BEHIND's centre, so
    track 2 is what the face is handed — 64% of the way down that box.  The
    old rule bound it; then, when BEHIND walks off and A's face is re-minted
    on her own box, track 2 (still alive, still "p00001") stood on another
    body and asserted p00001 != p00002: one woman, two keys, cannot_link.
    A head sits in the top 35% of a standing box; this face does not, so
    nothing binds and nothing is asserted.
    """
    frames = [
        {"boxes": [A, BEHIND], "faces": [FA]},         # minted
        {"boxes": [A, BEHIND], "faces": [FA]},         # 0.70: would bind track 2
        {"boxes": [A, BEHIND_MOVED], "faces": [FA]},   # re-minted on her own box
    ]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("p00002", True, 0.31),
    ]
    fake, final = run_scene(frames, script)
    assert fake.splits == []
    assert final["trackPresenceSplits"] == 0
    # ...and the same face where a head sits DOES bind (the baseline test
    # above proves it end to end); the rule itself, in numbers:
    loop = RunLoop.__new__(RunLoop)
    assert loop._head_sits_in(FA, A) is True
    assert loop._head_sits_in(FA, BEHIND) is False, "0.64 of the way down"
    assert loop._head_sits_in({"x": 12, "y": 55, "w": 60, "h": 40}, A) is False, "0.55 down"
    assert loop._head_sits_in({"x": 40, "y": 25, "w": 60, "h": 40}, A) is False, "0.5 widths off"
    assert loop._head_sits_in(FA, None) is False


def test_a_mint_never_binds_even_when_it_carries_a_comfortable_cosine():
    """The docstring's rule, enforced here and not by the wire's habits.

    Today a mint's cosine is under the 0.363 threshold by construction, so
    the 0.45 floor screens it; a match service that one day mints on an
    appearance clash would hand the runner a mint at 0.70.  isNew is the
    guard, not the number.
    """
    frames = [
        {"boxes": [A], "faces": [FA]},   # minted, at a cosine the floor would pass
        BACK_TURNED_BESIDE_NEWCOMER,
    ]
    script = [
        scripted_verdict("p00001", True, 0.70),
        scripted_verdict("p00002", True, None),
    ]
    fake, final = run_scene(frames, script)
    assert fake.splits == []
    assert final["trackPresenceSplits"] == 0


def test_a_staff_face_on_a_bound_track_unbinds_it():
    """A staff verdict is certain evidence the body is not the bound guest.

    Track 1 binds to p00001, then the face on it resolves as staff (a swap
    the contest missed, or a staff member minted as a guest before
    enrolment).  The binding must not go on asserting p00001 on that body:
    the next faceless frame beside a newcomer asserts nothing.
    """
    frames = MINT_THEN_BIND + [
        {"boxes": [A], "faces": [FA]},   # the same track now reads as staff
        BACK_TURNED_BESIDE_NEWCOMER,
    ]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("st-1", False, 0.7, staff=True),
        scripted_verdict("p00002", True, None),
    ]
    fake, final = run_scene(frames, script)
    assert fake.splits == []
    assert final["trackPresenceSplits"] == 0


def riders(n: int) -> dict:
    """Norms and attributes for n faces: the middle one embeds short."""
    norms = [30.0, 5.0, 28.0][:n]
    attrs = [
        {"gender": "M", "genderP": 0.91, "age": 41.2},
        {"gender": "F", "genderP": 0.6, "age": 30.0},
        {"gender": "M", "genderP": 0.99, "age": 7.0},
    ][:n]
    return {"norms": norms, "attributes": attrs}


def test_the_norm_gate_drops_a_face_with_its_vector_norm_and_attributes():
    """Index-consistency: the survivors reach the matcher with THEIR riders.

    Three faces embed as vectors of 0s, 1s and 2s with norms 30 / 5 / 28
    against a floor of 10.  The middle face must vanish together with its
    vector and its attributes, or face three is matched on face two's
    embedding — which is the failure that turns a gate into a mis-count.
    """
    frames = [{"boxes": [A, B], "faces": [FA, F_NOBODY, FB]}]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00002", True, None),
    ]
    fake, final = run_scene(frames, script, embed=riders, quality_min_feat_norm=10.0)

    assert len(fake.match_bodies) == 2, "the gated face never reached the matcher"
    assert [b["embedding"][0] for b in fake.match_bodies] == [0.0, 2.0]
    assert [b["featNorm"] for b in fake.match_bodies] == [30.0, 28.0]
    assert [b["attributes"]["age"] for b in fake.match_bodies] == [41.2, 7.0]
    assert final["gatedByFeatNorm"] == 1
    assert final["unique"] == 2

    # The ledger names the reason and carries the riders on each verdict.
    rec = fake.frame_records[0]
    assert rec["faces"]["gatedBy"] == {"featnorm": 1}
    assert rec["faces"]["faces"][1]["gate"] == "featnorm"
    assert rec["faces"]["kept"] == 2
    assert [v["featNorm"] for v in rec["verdicts"]] == [30.0, 28.0]
    assert rec["verdicts"][0]["attrs"] == {"gender": "M", "genderP": 0.91, "age": 41.2}
    assert rec["verdicts"][1]["attrs"] == {"gender": "M", "genderP": 0.99, "age": 7.0}


def test_the_norm_floor_is_off_by_default_and_recorded_when_armed():
    """0 gates nothing; armed, the run row says so like every other floor."""
    frames = [{"boxes": [A, B], "faces": [FA, F_NOBODY, FB]}]
    script = [scripted_verdict(f"p0000{i}", True, None) for i in (1, 2, 3)]
    fake, final = run_scene(frames, script, embed=riders)
    assert len(fake.match_bodies) == 3
    assert final["gatedByFeatNorm"] == 0
    assert final["gateArmed"] == ()
    assert fake.run_created["config"]["qualityMinFeatNorm"] == 0.0
    assert "featnorm" not in fake.run_created["config"]["gateArmed"]

    fake, final = run_scene(frames, script, embed=riders, quality_min_feat_norm=10.0)
    assert final["gateArmed"] == ("featnorm",)
    assert fake.run_created["config"]["qualityMinFeatNorm"] == 10.0
    assert fake.run_created["config"]["gateArmed"] == ["featnorm"]


def test_an_embed_reply_without_norms_or_attributes_still_counts():
    """An older embed service: nothing gated, riders absent, not zero.

    With the floor ARMED and no norm to read, the face is kept and counted
    unmeasured — an armed floor never rejects a face it could not measure —
    and the match body carries no featNorm/attributes keys at all.
    """
    frames = [{"boxes": [A], "faces": [FA]}] * 2
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.7),
    ]
    fake, final = run_scene(frames, script, embed=None, quality_min_feat_norm=10.0)

    assert len(fake.match_bodies) == 2
    assert all("featNorm" not in b and "attributes" not in b for b in fake.match_bodies)
    assert final["gatedByFeatNorm"] == 0
    assert final["gatedUnmeasured"] == 2, "one per face per frame: passed by absence"
    assert final["unique"] == 1
    rec = fake.frame_records[0]
    assert rec["verdicts"][0]["featNorm"] is None
    assert rec["verdicts"][0]["attrs"] is None
    assert rec["faces"]["faces"][0]["gateUnmeasured"] == ["featnorm"]

    # Floor off: the old service is simply the old service — nothing counted.
    fake, final = run_scene(frames, script, embed=None)
    assert final["gatedUnmeasured"] == 0
    assert "gateUnmeasured" not in fake.frame_records[0]["faces"]["faces"][0]


def test_the_body_rider_is_the_containing_person_box_and_the_frame_height():
    """R1: {h, w, yBottom, frameH} from the RAW detector box; None without one."""
    frames = [{"boxes": [A], "faces": [FA, F_NOBODY]}]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00002", True, None),
    ]
    fake, _ = run_scene(frames, script)

    assert fake.match_bodies[0]["body"] == {
        "h": 100.0, "w": 60.0, "yBottom": 120.0, "frameH": 120,
    }
    assert "body" not in fake.match_bodies[1], "no containing box: no body, not a zero body"


def test_body_box_needs_a_frame_height_to_mean_anything():
    """yBottom without frameH cannot be placed against the frame edge."""
    assert RunLoop._body_box(A, None) is None
    assert RunLoop._body_box(None, 120) is None
    assert RunLoop._body_box({"x": 1, "y": 2, "w": 3, "h": 4}, 120.0) == {
        "h": 4.0, "w": 3.0, "yBottom": 6.0, "frameH": 120,
    }


def test_parse_embed_reply_pads_and_cuts_to_the_embedding_count():
    """The three lists zip cleanly whatever the service sent."""
    e, n, a = parse_embed_reply({"embeddings": [[0.1], [0.2]]})
    assert (e, n, a) == ([[0.1], [0.2]], [None, None], [None, None])

    e, n, a = parse_embed_reply({
        "embeddings": [[0.1], [0.2], [0.3]],
        "norms": [21.0],                       # short: padded
        "attributes": [None, {"gender": "F", "genderP": 0.8, "age": 33.0}, {}, {}],  # long: cut
    })
    assert n == [21.0, None, None]
    assert a == [None, {"gender": "F", "genderP": 0.8, "age": 33.0}, None], (
        "an empty reading is not a reading"
    )


def test_a_malformed_attribute_reading_is_not_measured_and_age_is_clamped():
    """The match service 422s a bad rider, and a 422 on /match fails the run.

    So the runner forwards only what it can vouch for: an unknown sex, a
    probability outside 0..1 or a non-numeric age is None (not measured), and
    an age under zero — the head's output is an unclamped regression — is 0.
    """
    reply = {
        "embeddings": [[0.1]] * 6,
        "attributes": [
            {"gender": "X", "genderP": 0.9, "age": 30},
            {"gender": "M", "genderP": 1.5, "age": 30},
            {"gender": "M", "genderP": 0.9, "age": "old"},
            {"gender": "F", "genderP": 0.7, "age": -2.5},
            "junk",
            {"gender": "F", "genderP": 0.7, "age": 41.0},
        ],
    }
    _, _, a = parse_embed_reply(reply)
    assert a == [
        None, None, None,
        {"gender": "F", "genderP": 0.7, "age": 0.0},
        None,
        {"gender": "F", "genderP": 0.7, "age": 41.0},
    ]

    e, n, a = parse_embed_reply({"embeddings": [], "norms": "junk"})
    assert (e, n, a) == ([], [], [])
    assert parse_embed_reply({}) == ([], [], [])


# ---------------------------------------------------------------- config


def test_the_two_knobs_read_from_the_environment(monkeypatch):
    """HECO_PRESENCE_SPLIT (default 1) and HECO_QUALITY_MIN_FEAT_NORM (default
    0 = off); an empty value means unset, never zero — compose renders a knob
    left out of .env as ""."""
    from app import config as cfg

    s = cfg.from_env()
    assert s.presence_split == 1
    assert s.quality_min_feat_norm == 0.0
    assert Settings().presence_split == 1

    monkeypatch.setenv("HECO_PRESENCE_SPLIT", "0")
    monkeypatch.setenv("HECO_QUALITY_MIN_FEAT_NORM", "12.5")
    s = cfg.from_env()
    assert s.presence_split == 0
    assert s.quality_min_feat_norm == 12.5

    monkeypatch.setenv("HECO_PRESENCE_SPLIT", "")
    monkeypatch.setenv("HECO_QUALITY_MIN_FEAT_NORM", "")
    s = cfg.from_env()
    assert s.presence_split == 1, "empty is unset, not off"
    assert s.quality_min_feat_norm == 0.0


@pytest.mark.parametrize("floor", [0.0, -1.0])
def test_a_zero_or_negative_lock_floor_disables_presence_binding(floor):
    """The lock's off switch is presence's too: binding a track on ANY cosine
    would assert cannot_links on impostor-range evidence all night."""
    frames = MINT_THEN_BIND + [BACK_TURNED_BESIDE_NEWCOMER]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00001", False, 0.70),
        scripted_verdict("p00002", True, None),
    ]
    fake, final = run_scene(frames, script, track_lock_min_cosine=floor)
    assert fake.splits == []
    assert final["trackPresenceSplits"] == 0


def balance_riders(n: int) -> dict:
    """Norms, attributes and half-balance for n faces: the middle one has a
    dark bar across half of it (0.27, run f0bfc5's p00002) but a strong norm —
    the case the norm floor cannot see."""
    out = riders(n)
    out["norms"] = [30.0, 29.0, 28.0][:n]
    out["balance"] = [0.81, 0.27, 0.74][:n]
    return out


def test_the_balance_gate_drops_a_half_hidden_face_the_norm_floor_passes():
    """p00002's shape: a clean norm, one half dark. Armed at 0.33 with the
    norm floor also armed, the norm floor passes her and the balance floor
    drops her — with her vector, norm and attributes, index-consistently."""
    frames = [{"boxes": [A, B], "faces": [FA, F_NOBODY, FB]}]
    script = [
        scripted_verdict("p00001", True, None),
        scripted_verdict("p00002", True, None),
    ]
    fake, final = run_scene(frames, script, embed=balance_riders,
                            quality_min_feat_norm=18.0, quality_min_balance=0.33)

    assert [b["embedding"][0] for b in fake.match_bodies] == [0.0, 2.0]
    assert [b["featNorm"] for b in fake.match_bodies] == [30.0, 28.0]
    assert (final["gatedByFeatNorm"], final["gatedByBalance"]) == (0, 1)
    rec = fake.frame_records[0]
    assert rec["faces"]["gatedBy"] == {"balance": 1}
    assert rec["faces"]["faces"][1]["gate"] == "balance"
    assert [f.get("balance") for f in rec["faces"]["faces"]] == [0.81, 0.27, 0.74], \
        "every face's reading is in the ledger, kept or not, so the floor can be re-priced"


def test_the_balance_floor_is_off_by_default_and_an_unmeasured_face_is_kept():
    """0 gates nothing and arms nothing; armed, a face the embedder could not
    measure (None: off the frame edge, an older embed) is kept and counted."""
    frames = [{"boxes": [A, B], "faces": [FA, F_NOBODY, FB]}]
    script = [scripted_verdict(f"p0000{i}", True, None) for i in (1, 2, 3)]
    fake, final = run_scene(frames, script, embed=balance_riders)
    assert len(fake.match_bodies) == 3 and final["gatedByBalance"] == 0
    assert "balance" not in fake.run_created["config"]["gateArmed"]
    assert "qualityMinBalance" not in fake.run_created["config"], "off leaves the record as it was"

    def unmeasured(n):
        out = balance_riders(n)
        out["balance"] = [None, None, None][:n]
        return out

    fake, final = run_scene(frames, script, embed=unmeasured, quality_min_balance=0.33)
    assert len(fake.match_bodies) == 3, "absent is not zero: nobody is dropped on a missing reading"
    assert final["gatedByBalance"] == 0 and final["gatedUnmeasured"] >= 1
    assert "balance" in fake.run_created["config"]["gateArmed"]
    assert fake.run_created["config"]["qualityMinBalance"] == 0.33
