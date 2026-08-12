"""The runner's half of the gate contract: our config still means what the
library thinks it means.

counting/tests/test_gate.py pins what the gate IS, using its own declaration
of the shipped defaults — that package must not import the runner, which is
the point of it being a package. This file pins the other half: that the
RUNNER's Settings, the thing an operator actually edits and an env var
actually overrides, still produces exactly those numbers.

Split deliberately. If someone changes a default on either side, this test
fails and names the drift, instead of the two halves quietly disagreeing while
each one's own tests stay green — which is precisely how a discard threshold
moves without anyone deciding to move it.
"""

from app.config import Settings
from heco_counting.gate import GateThresholds


def test_the_runners_defaults_produce_the_libraries_shipped_gate():
    """Width-only at 56 px, every other floor unarmed. The shipped gate.

    Armed floors are the pipeline's only irreversible discard, and a guessed
    floor costs guests off an invoice — so the default has to be the gate that
    has always counted, and it has to be asserted from the config an operator
    edits, not from the library's own idea of itself.
    """
    t = GateThresholds.from_settings(Settings())
    assert t.min_px == 56.0
    assert t.min_ied_px == 0.0
    assert t.min_frontality == 0.0
    assert t.min_sharpness == 0.0
    assert t.min_eye_span == 0.0
    assert t.require_landmarks is False
    assert t.armed == (), "no signal beyond width may gate by default"


def test_every_quality_setting_reaches_the_gate():
    """Each floor is tunable per camera without a code change — all of them.

    Enumerated rather than spot-checked: a field added to Settings and not
    read by from_settings would be a knob the console offers and the gate
    ignores, and the operator would have no way to tell.
    """
    armed = Settings(
        quality_min_px=64.0,
        quality_min_ied_px=30.0,
        quality_min_frontality=0.55,
        quality_min_sharpness=25.0,
        quality_min_eye_span=0.30,
        quality_require_landmarks=True,
    )
    t = GateThresholds.from_settings(armed)
    assert t.min_px == 64.0
    assert (t.min_ied_px, t.min_frontality) == (30.0, 0.55)
    assert (t.min_sharpness, t.min_eye_span) == (25.0, 0.30)
    assert t.require_landmarks is True
    assert set(t.armed) == {"ied", "frontality", "sharpness", "eyespan", "landmarks"}


def test_canon_px_is_not_the_gates_business():
    """quality_canon_px is a REPORTING threshold, not a discard threshold.

    It marks a face as sub-canon so a report can state how much low-pixel
    evidence sits behind a number; it never rejects anything, which is why it
    is absent from GateThresholds entirely. Asserted so nobody 'completes' the
    dataclass by adding it and quietly turns a label into a discard.
    """
    assert not hasattr(GateThresholds(), "canon_px")
    assert Settings().quality_canon_px == 80.0
