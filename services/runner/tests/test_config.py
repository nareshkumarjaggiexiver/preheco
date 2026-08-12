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
        "HECO_HEAL_APPEARANCE_CLASH", "HECO_COPRESENCE_SPLIT",
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


def test_co_presence_splits_are_on_by_default_and_switchable_off(monkeypatch):
    """1 = on (the default), 0 = off, and empty must NOT read as off.

    Run 05b3b7 raised two false "likely duplicate" banners (0.316 and 0.360
    against a 0.363 threshold, clothing 0.94 and 0.57) while its own tap ledger
    held p00002 and p00007 matched in the SAME FRAME — so the mechanism ships
    ON.  It has a real cost (a hand-held phone showing its owner's face is one
    person asserted as two), so it also ships with an off switch; and a knob
    left out of .env arrives as "" from compose, which must mean "unset", not
    "disabled" — reading it as 0 would silently switch off a guard nobody
    touched.
    """
    from app import config as cfg

    assert cfg.Settings().copresence_split == 1

    monkeypatch.setenv("HECO_COPRESENCE_SPLIT", "")
    assert cfg.from_env().copresence_split == 1, "empty means unset, not off"

    monkeypatch.setenv("HECO_COPRESENCE_SPLIT", "0")
    assert cfg.from_env().copresence_split == 0, "0 is the off switch"


# --------------------------------------------- per-run quality profile
# The floors are box-wide env config by default. A launcher may override them
# for ONE run, because the two things an operator actually does are trying a
# stricter gate on tonight's footage before trusting it, and running one event
# under a profile that differs from the box's default.

from app.config import Settings  # noqa: E402
from app.loop import _apply_quality_profile  # noqa: E402


def test_no_profile_leaves_the_configured_settings_alone():
    """Sending nothing changes nothing — the identity case, and the default."""
    s = Settings(quality_min_frontality=0.55)
    assert _apply_quality_profile(s, None) is s
    assert _apply_quality_profile(s, {}) is s


def test_a_profile_overrides_only_what_it_names():
    """An OVERRIDE, not a reset: sending one field must not disarm a floor an
    engineer set on the box."""
    s = Settings(quality_min_frontality=0.55, quality_min_px=56.0)
    out = _apply_quality_profile(s, {"requireLandmarks": True})
    assert out.quality_require_landmarks is True
    assert out.quality_min_frontality == 0.55  # untouched
    assert out.quality_min_px == 56.0


def test_null_means_absent_not_zero():
    """The wire shape has optional fields; 'absent' and 'explicitly nothing'
    mean the same thing to an operator, and neither may disarm a floor."""
    s = Settings(quality_min_frontality=0.55)
    out = _apply_quality_profile(s, {"minFrontality": None, "minEyeSpan": 0.30})
    assert out.quality_min_frontality == 0.55
    assert out.quality_min_eye_span == 0.30


def test_unknown_keys_are_ignored_not_fatal():
    """The console and the box deploy separately: a launcher one version ahead
    must not be able to fail a run over a field this box has not learned."""
    s = Settings()
    out = _apply_quality_profile(s, {"minGlasses": 0.9, "requireLandmarks": True})
    assert out.quality_require_landmarks is True


def test_the_profile_reaches_the_gate_and_the_run_record():
    """What the operator chose has to be what the gate enforces AND what the
    run row records — the count is an invoice figure, so 'which floors were in
    force' must be recoverable months later."""
    from heco_counting.gate import GateThresholds

    s = _apply_quality_profile(
        Settings(), {"minEyeSpan": 0.30, "requireLandmarks": True, "faceReverifyIntervalS": 5.0}
    )
    t = GateThresholds.from_settings(s)
    assert t.armed == ("landmarks", "eyespan")
    assert s.face_reverify_interval_s == 5.0
