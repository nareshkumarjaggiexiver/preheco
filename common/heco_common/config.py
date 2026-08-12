"""Typed config-from-env helpers.

Services configure themselves from environment variables (compose-friendly,
no config files at POC scale). These helpers centralise the parsing rules so
"truthy" means the same thing in every service. Read at call time — tests may
set env per-case.
"""

import os

_TRUTHY = {"1", "true", "yes", "on"}
#: "" is NOT here: an empty value means the operator did not choose,
#: because that is what docker compose's `${VAR-}` idiom renders. See
#: env_bool for the incident that established it.
_FALSY = {"0", "false", "no", "off"}


def env_str(name: str, default: str) -> str:
    """Return env var ``name`` as a string, or ``default`` when unset."""
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    """Return env var ``name`` as int; ValueError names the variable.

    An EMPTY value counts as unset, because that is what an unset variable
    looks like by the time it reaches the process: docker-compose renders
    ``${VAR-}`` as an empty string, so a knob simply left out of .env arrives
    as "" rather than absent. Treating that as a parse error took the whole
    ingest service down — every /open returned 500 and no run could start —
    the moment INGEST_MAX_WIDTH was removed from .env after a trial.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def env_float(name: str, default: float) -> float:
    """Return env var ``name`` as float; ValueError names the variable.

    Empty counts as unset, for the same reason as :func:`env_int`.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def env_bool(name: str, default: bool) -> bool:
    """Return env var ``name`` as bool.

    Truthy: 1/true/yes/on; falsy: 0/false/no/off (case-insensitive).
    EMPTY or absent means "not chosen" and yields ``default``.
    Anything else raises ValueError naming the variable — a misspelled flag
    should fail loudly, not silently pick a side.
    """
    raw = os.environ.get(name)
    # EMPTY MEANS UNSET, and that is a compose fact, not a preference. Every
    # optional setting in docker-compose.yml is passed as `${HECO_X-}`, which
    # renders as the EMPTY STRING when the operator has not set it — so a
    # container always receives the variable, always with "" for "I did not
    # choose". Reading "" as False silently forces every default-True flag off
    # in the only environment that ships, and reports nothing.
    #
    # It cost a real one: HECO_ASYNC_REPORTING is the first env_bool in this
    # repo whose default is True, so it was the first to notice. A plain
    # `docker compose up` ran the synchronous reporter while the code, the
    # tests and the commit message all said async was the default. Nothing
    # failed; the box was just quietly slower than the thing that had been
    # measured.
    #
    # An operator who genuinely means false has four spellings to choose from
    # (0/false/no/off), all of which still work. Nobody means false by typing
    # nothing.
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    raise ValueError(f"{name} must be a boolean-ish value, got {raw!r}")
