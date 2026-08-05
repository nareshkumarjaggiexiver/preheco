"""Logging must never be the thing that leaks a camera password."""

import logging

from heco_common.logs import RunLog, safe, setup_logging


def test_credentials_are_scrubbed_from_anything_logged():
    """Source URLs embed rtsp://user:pass@ and logs get pasted into tickets."""
    assert safe("rtsp://admin:Hunter2@192.168.1.64/main") == "rtsp://***@192.168.1.64/main"
    assert "Hunter2" not in safe("opening rtsp://admin:Hunter2@cam/1 failed")
    # A URL without credentials is untouched.
    assert safe("rtsp://192.168.1.64/main") == "rtsp://192.168.1.64/main"
    assert safe(None) == "None"


def test_every_line_carries_its_run_id(caplog):
    """A busy night runs several runs; without run= the logs cannot be split."""
    log = RunLog(setup_logging("test"), "run-abc123")
    with caplog.at_level(logging.WARNING):
        log.warning("planner unreachable")
    assert "run=run-abc123" in caplog.text
    assert "planner unreachable" in caplog.text


def test_a_run_log_scrubs_too(caplog):
    """The scrub applies to run-scoped lines, not just the bare helper."""
    log = RunLog(setup_logging("test"), "run-1")
    with caplog.at_level(logging.INFO):
        log.info("opened rtsp://admin:s3cret@cam/1")
    assert "s3cret" not in caplog.text


def test_setup_is_idempotent():
    """uvicorn imports modules twice under reload; a second handler doubles lines."""
    setup_logging("a")
    before = len(logging.getLogger().handlers)
    setup_logging("b")
    assert len(logging.getLogger().handlers) == before
