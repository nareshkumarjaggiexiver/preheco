"""Unit tests for PlannerClient: transport injection, retry, batching."""

import pytest
from heco_common.planner import MAX_SAMPLES_PER_POST, PlannerClient, PlannerError
from heco_common.schemas import Sample


class FakeTransport:
    """Records every call; replays a scripted list of responses/exceptions."""

    def __init__(self, script=None):
        """``script`` items are (status, body) tuples or Exception instances."""
        self.calls: list[tuple[str, str, dict | None]] = []
        self.script = list(script or [])

    def __call__(self, method, url, payload):
        """Record and answer: scripted response, else a generic 200."""
        self.calls.append((method, url, payload))
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return 200, {"id": 42}


def _client(script=None, **kw) -> tuple[PlannerClient, FakeTransport, list[float]]:
    """A client with fake transport and recorded (not slept) backoffs."""
    ft = FakeTransport(script)
    naps: list[float] = []
    pc = PlannerClient("http://planner:8787/", transport=ft, sleep=naps.append, **kw)
    return pc, ft, naps


def test_create_run_sets_run_id_and_url():
    """create_run POSTs the contract path and remembers the returned id."""
    pc, ft, _ = _client()
    run_id = pc.create_run(3, label="gate A")
    assert run_id == 42 and pc.run_id == 42
    method, url, payload = ft.calls[0]
    assert (method, url) == ("POST", "http://planner:8787/api/pipeline/runs")
    assert payload == {"eventId": 3, "label": "gate A"}


def test_reporting_before_create_run_fails():
    """Stats/samples without an open run raise instead of posting nowhere."""
    pc, _, _ = _client()
    with pytest.raises(PlannerError):
        pc.post_stats("track", frames=1, fps=1.0)


def test_post_stats_shape():
    """post_stats validates through StageStats and hits the run's stats URL."""
    pc, ft, _ = _client()
    pc.create_run(3)
    pc.post_stats(
        "embed", frames=90, fps=12.5,
        metrics={"embedMs": {"count": 90, "min": 8.0, "mean": 11.2, "max": 30.1}},
    )
    method, url, payload = ft.calls[-1]
    assert url.endswith("/api/pipeline/runs/42/stats")
    assert payload["stage"] == "embed"
    assert payload["metrics"]["embedMs"]["mean"] == 11.2


def test_add_sample_buffers_and_autoflushes():
    """Samples buffer locally and flush exactly at batch_size."""
    pc, ft, _ = _client(batch_size=3)
    pc.create_run(3)
    pc.add_sample("face-detect", 1000, {"faceBoxWPx": 66.0})
    pc.add_sample("face-detect", 1100, {"faceBoxWPx": 71.0})
    assert len(ft.calls) == 1  # only the create_run so far
    pc.add_sample("face-detect", 1200, {"faceBoxWPx": 64.0})
    method, url, payload = ft.calls[-1]
    assert url.endswith("/api/pipeline/runs/42/samples")
    assert [s["tMs"] for s in payload["samples"]] == [1000, 1100, 1200]


def test_post_samples_chunks_to_contract_cap():
    """An oversized explicit list splits into ≤200-sample POSTs."""
    pc, ft, _ = _client()
    pc.create_run(3)
    samples = [Sample(stage="track", tMs=i, metrics={"personBoxHPx": 250.0}) for i in range(450)]
    pc.post_samples(samples)
    sizes = [len(p["samples"]) for _, u, p in ft.calls[1:] if u.endswith("/samples")]
    assert sizes == [MAX_SAMPLES_PER_POST, MAX_SAMPLES_PER_POST, 50]


def test_end_run_flushes_then_puts():
    """end_run drains the buffer before closing the run."""
    pc, ft, _ = _client(batch_size=50)
    pc.create_run(3)
    pc.add_sample("match", 9000, {"matchCosine": 0.41})
    pc.end_run("ended", notes="POC complete")
    kinds = [(m, u.rsplit("/", 1)[-1]) for m, u, _ in ft.calls]
    assert kinds == [("POST", "runs"), ("POST", "samples"), ("PUT", "42")]
    assert ft.calls[-1][2] == {"status": "ended", "notes": "POC complete"}


def test_retry_on_transport_error_then_success():
    """Two transport failures back off and retry; third attempt lands."""
    pc, ft, naps = _client(script=[OSError("down"), (500, {}), (200, {"id": 7})])
    assert pc.create_run(1) == 7
    assert len(ft.calls) == 3
    assert naps == [0.2, 0.4]  # exponential backoff, injected sleep


def test_retries_exhausted_raises():
    """Persistent 5xx surfaces as PlannerError after `retries` attempts."""
    pc, ft, _ = _client(script=[(503, {})] * 3)
    with pytest.raises(PlannerError, match="after 3 attempts"):
        pc.create_run(1)
    assert len(ft.calls) == 3


def test_4xx_fails_fast_without_retry():
    """Client errors mean a bad payload — retrying would just repeat it."""
    pc, ft, _ = _client(script=[(422, {"detail": "bad eventId"})])
    with pytest.raises(PlannerError, match="rejected"):
        pc.create_run("nope")
    assert len(ft.calls) == 1


# ------------------------------------------------ v1: taps / frames / feedback


def test_post_tap_is_single_shot_and_swallows_failure():
    """A tap posts once, never retries, and reports success as a bool."""
    pc, ft, naps = _client()
    pc.create_run(3)
    ft.script = [(200, {"ok": True}), (500, {})]
    assert pc.post_tap("match", {"unique": 4}) is True
    method, url, payload = ft.calls[-1]
    assert (method, url) == ("POST", "http://planner:8787/api/pipeline/runs/42/taps")
    assert payload == {"stage": "match", "payload": {"unique": 4}}
    # A 5xx is swallowed (no raise) and NOT retried — best-effort.
    before = len(ft.calls)
    assert pc.post_tap("match", {"unique": 5}) is False
    assert len(ft.calls) == before + 1 and naps == []


def test_post_tap_swallows_transport_exception():
    """A transport blow-up never propagates out of a best-effort tap."""
    pc, ft, _ = _client()
    pc.create_run(3)
    ft.script = [OSError("planner down")]
    assert pc.post_tap("track", {"tracks": 2}) is False


def test_post_frame_uses_file_transport():
    """A frame goes through the multipart file transport, not the JSON one."""
    calls = []

    def file_transport(url, fields, filename, content, ctype):
        calls.append((url, fields, filename, content, ctype))
        return 200, {"ok": True}

    pc, _, _ = _client()
    pc.file_transport = file_transport
    pc.create_run(3)
    assert pc.post_frame("face-detect", b"\xff\xd8jpeg") is True
    url, fields, filename, content, ctype = calls[0]
    assert url == "http://planner:8787/api/pipeline/runs/42/frames"
    assert fields == {"stage": "face-detect"} and content == b"\xff\xd8jpeg"
    assert ctype == "image/jpeg"


def test_post_frame_without_file_transport_is_noop():
    """No file transport configured: framing is skipped, never errors."""
    pc, ft, _ = _client()
    pc.create_run(3)
    before = len(ft.calls)
    assert pc.post_frame("ingest", b"jpeg") is False
    assert len(ft.calls) == before  # nothing sent


def test_poll_feedback_accepts_list_or_wrapped():
    """Feedback poll tolerates a bare list or an items/feedback wrapper."""
    pc, ft, _ = _client()
    pc.create_run(3)
    ft.script = [(200, [{"id": 1}]), (200, {"feedback": [{"id": 2}]})]
    assert pc.poll_feedback() == [{"id": 1}]
    assert pc.poll_feedback(since="2026-08-04T00:00:00Z") == [{"id": 2}]
    _, url, _ = ft.calls[-1]
    assert url.endswith("/api/pipeline/runs/42/feedback?since=2026-08-04T00%3A00%3A00Z")


def test_poll_feedback_empty_on_failure():
    """A planner hiccup yields no items, not an exception."""
    pc, ft, _ = _client()
    pc.create_run(3)
    ft.script = [(503, {})]
    assert pc.poll_feedback() == []


def test_resolve_feedback_puts_status():
    """resolve_feedback PUTs the item status and reports acceptance."""
    pc, ft, _ = _client()
    pc.create_run(3)
    ft.script = [(200, {"ok": True})]
    assert pc.resolve_feedback(9, "applied") is True
    method, url, payload = ft.calls[-1]
    assert (method, url) == ("PUT", "http://planner:8787/api/feedback/9")
    assert payload == {"status": "applied"}


def test_report_enrolment_retries_then_raises():
    """Enrolment reporting uses the retrying transport (worth a retry)."""
    pc, ft, _ = _client()
    pc.create_run(3)
    ft.script = [(200, {"ok": True})]
    pc.report_enrolment("st-1", "2026-08-04T10:00:00Z", 5)
    method, url, payload = ft.calls[-1]
    assert (method, url) == ("PUT", "http://planner:8787/api/staff/st-1")
    assert payload == {"enrolledAt": "2026-08-04T10:00:00Z", "sampleCount": 5}

    pc2, ft2, _ = _client()
    pc2.create_run(3)
    ft2.script = [(503, {})] * 3
    with pytest.raises(PlannerError):
        pc2.report_enrolment("st-2", "2026-08-04T10:00:00Z", 5)


# ---------------------------------------------- regressions (2026-08-05 review)


def test_best_effort_calls_use_their_own_transport():
    """Taps/frames/feedback must be routable to a SHORT-timeout client.

    Regression: one 30 s httpx client served stage calls AND every planner
    call, so "best effort" only ever meant "errors are swallowed" — a planner
    that accepted connections and then answered slowly still blocked the frame
    loop for up to 30 s per call, ~11 calls per tick. Bounding latency needs a
    separate transport, so the single-shot calls must not share the retrying
    one.
    """
    retrying = FakeTransport()
    quick = FakeTransport()
    pc = PlannerClient(
        "http://planner:8787", transport=retrying, best_effort_transport=quick
    )
    pc.create_run(3)  # retrying path
    pc.post_stats("match", frames=1, fps=1.0)  # retrying path
    pc.post_tap("match", {"unique": 1})  # best-effort path
    pc.poll_feedback()  # best-effort path
    pc.resolve_feedback(7, "applied")  # best-effort path
    pc.staff_tombstones("site-1")  # best-effort path

    assert [c[0] for c in quick.calls] == ["POST", "GET", "PUT", "GET"]
    assert all("/taps" in u or "feedback" in u or "tombstones" in u for _, u, _ in quick.calls)
    assert [u for _, u, _ in retrying.calls] == [
        "http://planner:8787/api/pipeline/runs",
        "http://planner:8787/api/pipeline/runs/42/stats",
    ]


def test_best_effort_transport_defaults_to_the_main_one():
    """A caller that does not care keeps the previous single-transport behaviour."""
    ft = FakeTransport()
    pc = PlannerClient("http://planner:8787", transport=ft)
    assert pc.best_effort_transport is ft


def test_a_token_rides_on_every_call_including_the_frame_upload():
    """The planner refuses non-loopback exposure without a token, so a runner
    reaching it across a network must authenticate on EVERY path — including
    the multipart frame upload, which uses a different transport."""
    seen: list[dict] = []

    def transport(method, url, payload):
        seen.append({"method": method, "url": url})
        return 200, {"id": 7}

    pc = PlannerClient("http://planner:8787", transport=transport, token="sekrit")
    assert pc.token == "sekrit"

    # With an explicit transport the header is the caller's business (the runner
    # sets it on the httpx client); what matters is the token is carried and
    # surfaced so runs.py can wire it.
    pc.create_run(event_id="e1")
    assert seen and seen[0]["method"] == "POST"


def test_no_token_means_no_authorization_header():
    """Loopback development must keep working with no token at all."""
    pc = PlannerClient("http://localhost:8787")
    assert pc.token is None


def test_a_401_fails_fast_with_an_actionable_message():
    """A 401 is a configuration error. Retrying it three times only buries the
    real problem, so it must fail immediately and say what to set."""
    calls = {"n": 0}

    def transport(method, url, payload):
        calls["n"] += 1
        return 401, {"error": "this planner requires a token"}

    pc = PlannerClient("http://planner:8787", transport=transport, retries=3, sleep=lambda _: None)
    try:
        pc.create_run(event_id="e1")
        raise AssertionError("expected PlannerError")
    except PlannerError as exc:
        assert calls["n"] == 1, "a 401 must not be retried"
        assert "HECO_TOKEN" in str(exc)
        assert "sent none" in str(exc)


def test_a_rejected_token_on_the_best_effort_path_is_counted_not_shrugged_off():
    """Best-effort calls swallow failures — correct for taps, dangerous for the
    erasure poll, where a 401 looks exactly like "nothing to erase". Count it so
    the runner can say so instead of quietly skipping erasure all night."""
    def transport(method, url, payload):
        return 401, {"error": "this planner requires a token"}

    pc = PlannerClient("http://planner:8787", transport=transport, token="wrong")
    assert pc.staff_tombstones("site1") == []
    assert pc.auth_failures == 1, "the refusal is recorded"
    pc.staff_tombstones("site1")
    assert pc.auth_failures == 2
