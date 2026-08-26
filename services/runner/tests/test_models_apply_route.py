"""Route-level tests of the hot-swap seams: /models and /models/apply
through ``app.main`` with a REAL RunManager.

The unit tests of ``loop.apply_models`` (test_loop.py) exercise the
orchestration maths against a MockTransport, but every shipped defect in
this feature lived in the seams those tests never touch: the live-run veto's
race with POST /runs, the HTTPException mapping, the probe client's
lifecycle, and the failure paths where compensation used to be skipped
entirely. So this module drives the routes with the real manager — the
swap-window lock, the registry lock and the exception mapping are the code
under test, and only the outbound HTTP is faked.
"""

import json
import threading

import app.main as main
import httpx
import pytest
from app import config
from app.loop import probe_models
from app.runs import RunManager
from fastapi.testclient import TestClient


class HarnessManager(RunManager):
    """A real RunManager whose outbound probe clients speak MockTransport.

    Everything that matters here — begin_swap/end_swap, the shared registry
    lock, live_run_ids — is inherited REAL; only the transports are faked,
    so a test failure indicts the manager or the route, never a stand-in.
    """

    def __init__(self, handler):
        super().__init__(config.from_env())
        self._handler = handler
        self.probe_clients: list[httpx.Client] = []

    def probe_client(self):
        """A MockTransport-backed client, remembered so tests can assert closure."""
        client = httpx.Client(transport=httpx.MockTransport(self._handler))
        self.probe_clients.append(client)
        return client

    def probe_live_models(self):
        """Run the REAL probe_models, just through the fake transport."""
        client = httpx.Client(transport=httpx.MockTransport(self._handler), timeout=2.0)
        try:
            return probe_models(client, self.settings)
        finally:
            client.close()


@pytest.fixture
def harness(monkeypatch):
    """Install a HarnessManager for a handler; returns (manager, TestClient)."""

    def install(handler):
        mgr = HarnessManager(handler)
        monkeypatch.setattr(main, "manager", mgr)
        return mgr, TestClient(main.app, raise_server_exceptions=False)

    return install


def _recording_handler(posts):
    """Healthy persons+faces services that record every /model POST."""

    def handler(request):
        host, path = request.url.host, request.url.path
        if path == "/health":
            return httpx.Response(200, json={"ok": True, "model": f"old-{host}.onnx"})
        if path == "/model":
            posts.append((host, json.loads(request.content)["file"]))
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(f"unexpected {host}{path}")

    return handler


# ------------------------------------------------- the bidirectional veto


def test_a_live_run_vetoes_the_apply_naming_its_ids(harness):
    """A swap under a running count would falsify that run's model stamp,
    so the apply must refuse AND tell the operator which runs to stop."""
    posts = []
    mgr, client = harness(_recording_handler(posts))
    gate = threading.Event()
    thread = threading.Thread(target=gate.wait, daemon=True)
    thread.start()
    # Registered exactly as start() would leave it: an alive thread in the
    # registry is the veto's whole evidence.
    with mgr._lock:
        mgr._threads["run-live-1"] = thread
    try:
        r = client.post("/models/apply", json={"stages": {"persons": "yolox_s.onnx"}})
        assert r.status_code == 409
        assert "run-live-1" in r.json()["detail"]
        assert posts == [], "a vetoed apply must not have touched any service"
    finally:
        gate.set()
        thread.join(timeout=5)


def test_the_swap_window_refuses_run_starts_and_sibling_applies(harness):
    """Both directions of the veto, plus concurrent-apply serialization.

    begin_swap and RunManager.start share ONE lock, so a run cannot start
    mid-swap (409 with the swap sentence) and a second apply cannot
    interleave per-service swaps with the first (409 with the collision
    sentence). Non-blocking on purpose: the second operator must SEE the
    collision, not silently last-writer-win.
    """
    posts = []
    mgr, client = harness(_recording_handler(posts))
    assert mgr.begin_swap() == [], "an idle box grants the window"
    try:
        run = client.post(
            "/runs", json={"eventId": "ev-1", "source": {"path": "/tmp/clip.mp4"}}
        )
        assert run.status_code == 409
        assert "model swap is in progress" in run.json()["detail"]

        second = client.post("/models/apply", json={"stages": {"faces": "scrfd.onnx"}})
        assert second.status_code == 409
        assert "already in flight" in second.json()["detail"]
    finally:
        mgr.end_swap()
    # The window is RELEASED: the next apply proceeds normally.
    r = client.post("/models/apply", json={"stages": {"persons": "yolox_s.onnx"}})
    assert r.status_code == 200


def test_a_refused_apply_still_releases_the_swap_window(harness):
    """end_swap rides a finally: one 400 must not leave the box refusing
    every future run and apply forever."""
    mgr, client = harness(_recording_handler([]))
    r = client.post("/models/apply", json={"stages": {"embed": "arcface.onnx"}})
    assert r.status_code == 400
    assert "deployment" in r.json()["detail"]
    assert mgr.begin_swap() == [], "the window must be free again"
    mgr.end_swap()


# ------------------------------------------------- happy path and mapping


def test_happy_path_reports_applied_and_before_and_closes_the_client(harness):
    """A clean two-stage apply answers 200 with applied+before, and the
    probe client the route built is closed on the way out."""
    posts = []
    mgr, client = harness(_recording_handler(posts))
    r = client.post(
        "/models/apply",
        json={"stages": {"persons": "yolox_s.onnx", "faces": "scrfd_2.5g_kps.onnx"}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["applied"] == {"persons": "yolox_s.onnx", "faces": "scrfd_2.5g_kps.onnx"}
    assert body["before"] == {"persons": "old-persons.onnx", "faces": "old-faces.onnx"}
    assert mgr.probe_clients and all(c.is_closed for c in mgr.probe_clients)


def test_apply_models_refusal_maps_to_400_with_the_error_as_detail(harness):
    """{ok: false} from the orchestration becomes the route's 400 detail."""
    _, client = harness(_recording_handler([]))
    r = client.post("/models/apply", json={"stages": {"tracker": "sort.onnx"}})
    assert r.status_code == 400
    assert "not hot-swappable" in r.json()["detail"]


# ------------------------------------------------- the compensation seams


def test_unreachable_second_stage_rolls_the_first_back_and_tells_the_truth(harness):
    """The shipped bug: persons swapped, faces' health probe raised, and the
    route answered 'nothing was changed' with persons still on the new
    weights. Now the first stage is compensated and the sentence says so."""
    posts = []

    def handler(request):
        host, path = request.url.host, request.url.path
        if path == "/health":
            if host == "faces":
                raise httpx.ConnectError("faces mid-restart")
            return httpx.Response(200, json={"ok": True, "model": f"old-{host}.onnx"})
        if path == "/model":
            posts.append((host, json.loads(request.content)["file"]))
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(f"unexpected {host}{path}")

    _, client = harness(handler)
    r = client.post(
        "/models/apply", json={"stages": {"persons": "yolox_s.onnx", "faces": "scrfd.onnx"}}
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "nothing was changed" not in detail, "persons DID change and roll back"
    assert "rolled back to the previous selection" in detail
    assert posts == [
        ("persons", "yolox_s.onnx"),  # applied
        ("persons", "old-persons.onnx"),  # compensated
    ]


def test_a_timed_out_apply_post_compensates_and_maps_to_400_not_500(harness):
    """The shipped bug: a transport exception from POST /model escaped the
    route entirely — 500, no rollback, and (because a client timeout does
    not cancel server work) a box state no report described. Now the prior
    stage is rolled back, the answer is a mapped 400, and the sentence sends
    the operator to /health for the one truth we cannot know."""
    posts = []

    def handler(request):
        host, path = request.url.host, request.url.path
        if path == "/health":
            return httpx.Response(200, json={"ok": True, "model": f"old-{host}.onnx"})
        if path == "/model":
            posts.append((host, json.loads(request.content)["file"]))
            if host == "faces":
                raise httpx.ReadTimeout("cold session build outlived the budget")
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(f"unexpected {host}{path}")

    _, client = harness(handler)
    r = client.post(
        "/models/apply",
        json={"stages": {"persons": "yolox_s.onnx", "faces": "scrfd_10g.onnx"}},
    )
    assert r.status_code == 400, "a transport failure is a mapped refusal, never a 500"
    detail = r.json()["detail"]
    assert "re-probe /health" in detail
    assert "may still have completed" in detail, "the timed-out swap's fate is unknowable"
    assert posts == [
        ("persons", "yolox_s.onnx"),   # applied
        ("faces", "scrfd_10g.onnx"),   # the POST that timed out client-side
        ("persons", "old-persons.onnx"),  # compensated anyway
    ]


# ------------------------------------------------- the live-models read


def test_get_models_serves_the_probe_truth_per_stage(harness):
    """The planner's honesty check: after an answer it never read, it asks
    the box. Unreachable and unloaded stages keep probe_models' truthful
    strings — a filename here would repeat the healthy-while-dead incident
    inside the audit trail itself."""

    def handler(request):
        host = request.url.host
        if host == "embed":
            raise httpx.ConnectError("embed down")
        if host == "faces":
            return httpx.Response(200, json={"ok": False, "model": "scrfd.onnx"})
        return httpx.Response(200, json={"ok": True, "model": f"live-{host}.onnx"})

    _, client = harness(handler)
    r = client.get("/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert models["persons"] == "live-persons.onnx"
    assert models["faces"] == "unloaded (scrfd.onnx)"
    assert models["embed"] == "unreachable"
    assert models["reid"] == "none"
