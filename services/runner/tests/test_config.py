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
        "HECO_TRACK_LOCK_MIN_COSINE", "HECO_HEAL_APPEARANCE_UNSURE",
        "HECO_HEAL_APPEARANCE_CLASH",
    ):
        monkeypatch.setenv(name, "")
    s = cfg.from_env()          # must not raise
    assert s.quality_min_px == 56.0, "an empty value means unset, not zero"
    assert s.quality_min_frontality == 0.0
    assert s.source_stall_s == 45.0
    # An empty lock floor must fall back to 0.45, NOT to 0.0 — 0 is the off
    # switch, so reading "" as zero would silently disable a mechanism the
    # operator never touched.
    assert s.track_lock_min_cosine == 0.45


def test_the_clothing_bands_default_to_the_measured_numbers(monkeypatch):
    """0.35 / 0.55, and the env moves both (bench 6e1a5d, finding C).

    The clash floor was 0.50 and vetoed a probably-correct fold at histogram
    intersection 0.4991 — nine ten-thousandths.  It is now 0.35 (only a
    genuine disagreement blocks) with an uncertain band up to 0.55 that
    proceeds and is counted instead of guessing.
    """
    from app import config as cfg

    s = cfg.from_env()
    assert s.heal_appearance_clash == 0.35
    assert s.heal_appearance_unsure == 0.55
    assert s.heal_appearance_clash <= 0.4991, "the measured case must not clash"
    assert s.heal_appearance_unsure > 0.4991, "...and must land in the uncertain band"

    monkeypatch.setenv("HECO_HEAL_APPEARANCE_CLASH", "0.5")
    monkeypatch.setenv("HECO_HEAL_APPEARANCE_UNSURE", "0.6")
    monkeypatch.setenv("HECO_TRACK_LOCK_MIN_COSINE", "0")
    s = cfg.from_env()
    assert s.heal_appearance_clash == 0.5
    assert s.heal_appearance_unsure == 0.6
    assert s.track_lock_min_cosine == 0.0, "0 is the lock's off switch"
