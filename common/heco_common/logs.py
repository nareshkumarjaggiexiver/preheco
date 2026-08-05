"""Logging for the pipeline services.

The runner had none. Fourteen places swallowed an exception to keep a live
count alive — the right instinct, since a planner hiccup must never stop the
counting — but the reason went nowhere, so an operator whose numbers looked
wrong at midnight had nothing to read. "It just stopped" was the whole
diagnostic record.

Three rules this module exists to enforce:

1. **Every line carries its run id.** A venue runs several gates; a night runs
   several runs. Without ``run=<id>`` on every line the logs of a busy evening
   cannot be separated back into the runs that produced them, which is exactly
   when you need them.

2. **Nothing logs a credential.** Source URLs embed ``rtsp://user:pass@`` and
   logs get pasted into tickets. :func:`safe` scrubs userinfo from anything
   before it is written; prefer passing an already-safe label.

3. **Swallowed is not silent.** A best-effort call that fails still logs at
   WARNING with what it was doing, so the pattern stays "the count survives AND
   the reason is recorded" rather than "the count survives".

Level comes from ``HECO_LOG_LEVEL`` (default INFO). Format is plain text with
a timestamp, because these are read by a person on site with `tail -f`, not
shipped to an aggregator.
"""

from __future__ import annotations

import logging
import os
import re
import sys

_CONFIGURED = False

# rtsp://user:pass@host -> rtsp://***:***@host, and any bare user:pass@ pair.
_USERINFO = re.compile(r"//[^/@\s]+@")


def safe(text: object) -> str:
    """Render a value for a log line with any URL credentials scrubbed."""
    return _USERINFO.sub("//***@", str(text))


def setup_logging(service: str) -> logging.Logger:
    """Configure root logging once and return this service's logger.

    Idempotent: uvicorn imports application modules more than once under
    reload, and a second handler would double every line.
    """
    global _CONFIGURED
    logger = logging.getLogger(service)
    if _CONFIGURED:
        return logger
    level = getattr(logging, os.environ.get("HECO_LOG_LEVEL", "INFO").upper(), logging.INFO)
    # basicConfig, deliberately NOT force=True: it installs our handler when the
    # process has none, and stands aside when a host (uvicorn, pytest) has
    # already configured logging. Replacing root's handlers would silence the
    # host's own output and break test capture — a logging module that breaks
    # the thing it is meant to help is not an improvement.
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().setLevel(level)
    # httpx logs every request at INFO. The runner makes ~11 HTTP calls per
    # frame, so at 15 fps that is ~165 lines a second burying the handful that
    # describe the run. Their WARNINGs still get through, which is the level
    # that would actually matter.
    for chatty in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(chatty).setLevel(logging.WARNING)
    _CONFIGURED = True
    return logger


class RunLog:
    """A logger bound to one run, so every line it writes is attributable.

    Wraps rather than subclasses ``Logger`` because ``LoggerAdapter``'s extra
    dict does not survive ``%``-style formatting the way a prefix does, and a
    prefix is what makes ``grep run=abc123 heco.log`` work.
    """

    def __init__(self, logger: logging.Logger, run_id: str) -> None:
        self._log = logger
        self._run = run_id

    def _fmt(self, msg: str) -> str:
        return f"run={self._run} {safe(msg)}"

    def info(self, msg: str) -> None:
        """A lifecycle event worth seeing on a normal night."""
        self._log.info(self._fmt(msg))

    def warning(self, msg: str) -> None:
        """Something failed but the run continues — the swallowed-error case."""
        self._log.warning(self._fmt(msg))

    def error(self, msg: str) -> None:
        """Something failed that ends or corrupts the run."""
        self._log.error(self._fmt(msg))

    def debug(self, msg: str) -> None:
        """Per-tick detail; off unless HECO_LOG_LEVEL=DEBUG."""
        self._log.debug(self._fmt(msg))
