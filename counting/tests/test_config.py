"""The counting knobs: overrides that override, and a record that stays readable."""

from dataclasses import dataclass

import pytest
from heco_counting.config import (
    QUALITY_FIELDS,
    CountingConfig,
    fold_quality_profile,
    gate_config,
)


@dataclass(frozen=True)
class Wider(CountingConfig):
    """A host config with fields the library does not know about."""

    planner_url: str = "http://planner:8787"


def test_an_omitted_key_keeps_the_boxs_own_setting():
    """The profile is an OVERRIDE, not a reset.

    Sending {"requireLandmarks": true} must not silently disarm a frontality
    floor an engineer set on that machine — that would be a discard threshold
    moving because somebody touched a different one.
    """
    armed = CountingConfig(quality_min_frontality=0.55)
    out = fold_quality_profile(armed, {"requireLandmarks": True})
    assert out.quality_min_frontality == 0.55
    assert out.quality_require_landmarks is True


def test_an_explicit_null_means_the_same_as_omitted():
    """The wire shape has optional fields, and an operator means one thing by
    both."""
    armed = CountingConfig(quality_min_ied_px=30.0)
    assert fold_quality_profile(armed, {"minIedPx": None}).quality_min_ied_px == 30.0


def test_an_unknown_key_is_ignored_not_rejected():
    """The console and the box deploy separately.

    A console one version ahead must not be able to fail a run over a field
    the box has not learned yet — that turns a cosmetic version skew into a
    gate that counts nobody.
    """
    out = fold_quality_profile(CountingConfig(), {"someFutureFloor": 1.23})
    assert out == CountingConfig()


def test_an_empty_profile_returns_the_same_object():
    """No profile is not a reason to rebuild the config."""
    c = CountingConfig()
    assert fold_quality_profile(c, None) is c
    assert fold_quality_profile(c, {}) is c


def test_it_folds_onto_a_host_config_with_extra_fields():
    """One rule, two callers: the runner folds onto its full Settings and a
    worker folds onto a CountingConfig. Generic so they cannot drift apart."""
    out = fold_quality_profile(Wider(), {"minPx": 64.0})
    assert out.quality_min_px == 64.0
    assert out.planner_url == "http://planner:8787", "host fields must survive"


def test_a_field_the_host_lacks_is_skipped_rather_than_exploding():
    """A minimal host config must not crash on a profile key it has no room
    for — the profile is the console's, and the console serves many hosts."""

    @dataclass(frozen=True)
    class Minimal:
        quality_min_px: float = 56.0

    assert fold_quality_profile(Minimal(), {"minEyeSpan": 0.3}).quality_min_px == 56.0


def test_every_wire_key_maps_to_a_real_field():
    """A typo in the table would be a knob the console offers and nothing
    reads, with no error anywhere."""
    c = CountingConfig()
    for field in QUALITY_FIELDS.values():
        assert hasattr(c, field), f"{field} is in the table but not on the config"


def test_the_gate_record_carries_every_key_including_the_unarmed_ones():
    """Unarmed floors are recorded as 0.0, never omitted.

    The count is an invoice figure, so a config with no IED key must mean an
    OLD run rather than an unarmed one — otherwise "which gate produced this
    number" becomes unanswerable months later, which is exactly when it is
    asked.
    """
    rec = gate_config(CountingConfig(), armed=())
    assert set(rec) == {
        "qualityMinPx", "qualityMinConf", "qualityCanonPx", "qualityMinIedPx",
        "qualityMinFrontality",
        "qualityMinSharpness", "qualityMinEyeSpan", "qualityRequireLandmarks",
        "faceReverifyIntervalS", "gateArmed",
    }
    assert rec["qualityMinIedPx"] == 0.0
    assert rec["gateArmed"] == []


def test_the_gate_record_is_json_shaped():
    """armed is a LIST, not a tuple: it is written to a permanent run row."""
    rec = gate_config(CountingConfig(), armed=("ied", "frontality"))
    assert isinstance(rec["gateArmed"], list)
    assert rec["gateArmed"] == ["ied", "frontality"]


def test_from_settings_reads_a_duck_typed_host():
    """This package must not import the runner; it reads whatever carries the
    names."""

    @dataclass
    class HostSettings:
        quality_min_px: float = 64.0
        quality_canon_px: float = 96.0
        quality_min_ied_px: float = 30.0
        quality_min_frontality: float = 0.55
        quality_min_sharpness: float = 25.0
        quality_min_eye_span: float = 0.3
        quality_require_landmarks: bool = True
        face_reverify_interval_s: float = 2.0

    c = CountingConfig.from_settings(HostSettings())
    assert (c.quality_min_px, c.quality_canon_px) == (64.0, 96.0)
    assert c.quality_require_landmarks is True
    assert c.face_reverify_interval_s == 2.0


def test_the_config_is_frozen():
    """A threshold that can move mid-run is a count nobody can attribute."""
    with pytest.raises(AttributeError):
        CountingConfig().quality_min_px = 1.0
