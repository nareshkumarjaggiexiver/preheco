"""Unit tests for the config-from-env helpers."""

import pytest
from heco_common import config


def test_env_str(monkeypatch):
    """String getter returns the value or the default."""
    monkeypatch.setenv("HECO_T_STR", "rtsp://cam/1")
    assert config.env_str("HECO_T_STR", "x") == "rtsp://cam/1"
    assert config.env_str("HECO_T_MISSING", "fallback") == "fallback"


def test_env_int(monkeypatch):
    """Int getter parses, defaults, and names the variable on garbage."""
    monkeypatch.setenv("HECO_T_INT", "56")
    assert config.env_int("HECO_T_INT", 0) == 56
    assert config.env_int("HECO_T_MISSING", 80) == 80
    monkeypatch.setenv("HECO_T_INT", "eighty")
    with pytest.raises(ValueError, match="HECO_T_INT"):
        config.env_int("HECO_T_INT", 0)


def test_env_float(monkeypatch):
    """Float getter parses, defaults, and rejects garbage."""
    monkeypatch.setenv("HECO_T_F", "0.363")
    assert config.env_float("HECO_T_F", 0.0) == 0.363
    assert config.env_float("HECO_T_MISSING", 1.5) == 1.5
    monkeypatch.setenv("HECO_T_F", "cosine")
    with pytest.raises(ValueError, match="HECO_T_F"):
        config.env_float("HECO_T_F", 0.0)


def test_env_bool(monkeypatch):
    """Bool getter accepts the documented spellings and rejects the rest."""
    for raw, want in [("1", True), ("true", True), ("YES", True), ("on", True),
                      ("0", False), ("False", False), ("no", False), ("", False)]:
        monkeypatch.setenv("HECO_T_B", raw)
        assert config.env_bool("HECO_T_B", not want) is want
    assert config.env_bool("HECO_T_MISSING", True) is True
    monkeypatch.setenv("HECO_T_B", "maybe")
    with pytest.raises(ValueError, match="HECO_T_B"):
        config.env_bool("HECO_T_B", False)


def test_an_empty_value_counts_as_unset(monkeypatch):
    """docker-compose renders `${VAR-}` as "" for a variable left out of .env.

    Treating that as a parse error took ingest down completely: every /open
    returned 500 and no run could start, the moment INGEST_MAX_WIDTH was
    removed from .env after a trial. An empty env var means unset.
    """
    monkeypatch.setenv("HECO_T_INT", "")
    assert config.env_int("HECO_T_INT", 7) == 7
    monkeypatch.setenv("HECO_T_INT", "   ")
    assert config.env_int("HECO_T_INT", 7) == 7

    monkeypatch.setenv("HECO_T_FLOAT", "")
    assert config.env_float("HECO_T_FLOAT", 1.5) == 1.5

    # Genuine rubbish must still fail loudly — this is not a licence to guess.
    monkeypatch.setenv("HECO_T_INT", "wide")
    with pytest.raises(ValueError, match="HECO_T_INT"):
        config.env_int("HECO_T_INT", 7)
