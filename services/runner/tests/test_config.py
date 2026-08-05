"""The runner's settings-from-env: defaults, overrides, and empty values."""

def test_an_empty_env_var_never_crashes_the_runner(monkeypatch):
    """Every knob compose can render empty must fall back, not raise.

    docker-compose writes `${VAR-}` as "" for a knob left out of .env. Reading
    that with a raw float() took ingest down completely — every /open returned
    500 — so the runner's 17 numeric reads all go through env_float/env_int,
    which treat empty as unset.
    """
    from app import config as cfg

    for name in (
        "HECO_QUALITY_MIN_PX", "HECO_QUALITY_MIN_IED_PX",
        "HECO_QUALITY_MIN_FRONTALITY", "HECO_QUALITY_MIN_SHARPNESS",
        "HECO_SOURCE_STALL_S", "HECO_FLUSH_INTERVAL_S", "HECO_TAP_BUDGET_S",
    ):
        monkeypatch.setenv(name, "")
    s = cfg.from_env()          # must not raise
    assert s.quality_min_px == 56.0, "an empty value means unset, not zero"
    assert s.quality_min_frontality == 0.0
    assert s.source_stall_s == 45.0
