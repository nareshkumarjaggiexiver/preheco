"""Typed config-from-env helpers.

Services configure themselves from environment variables (compose-friendly,
no config files at POC scale). These helpers centralise the parsing rules so
"truthy" means the same thing in every service. Read at call time — tests may
set env per-case.
"""

import os

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


def env_str(name: str, default: str) -> str:
    """Return env var ``name`` as a string, or ``default`` when unset."""
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    """Return env var ``name`` as int; ValueError names the variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def env_float(name: str, default: float) -> float:
    """Return env var ``name`` as float; ValueError names the variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def env_bool(name: str, default: bool) -> bool:
    """Return env var ``name`` as bool.

    Truthy: 1/true/yes/on; falsy: 0/false/no/off/empty (case-insensitive).
    Anything else raises ValueError naming the variable — a misspelled flag
    should fail loudly, not silently pick a side.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    raise ValueError(f"{name} must be a boolean-ish value, got {raw!r}")
