"""The knobs the counting decisions read, and nothing else.

No URLs, no timeouts, no tap intervals, no credentials. A process that wants
to count needs the thresholds; it does not need to know there is a planner.
That separation is the point of this module: a per-camera worker and a
per-gate fusion process will each hold one of these, and neither should
inherit the runner's whole configuration to get it.

The defaults are copied verbatim from the runner's own ``Settings``, and the
runner has a test asserting the two still agree — see
``services/runner/tests/test_gate_binding.py`` for why that lives on the
runner's side rather than here.
"""

from dataclasses import dataclass, replace

#: Wire key -> config field, for the per-run quality profile the launcher
#: sends. The mapping lives in ONE place; a new floor is a line here and
#: nowhere else.
QUALITY_FIELDS = {
    "minPx": "quality_min_px",
    "minConf": "quality_min_conf",
    "minIedPx": "quality_min_ied_px",
    "minFrontality": "quality_min_frontality",
    "minSharpness": "quality_min_sharpness",
    "minEyeSpan": "quality_min_eye_span",
    "requireLandmarks": "quality_require_landmarks",
    "faceReverifyIntervalS": "face_reverify_interval_s",
}


@dataclass(frozen=True)
class CountingConfig:
    """Every threshold the counting logic reads.

    Frozen, because a configuration that can move mid-run is a discard
    threshold that can move mid-count — and the resulting number would be
    unattributable to any gate anybody chose.
    """

    #: The quality gate. Width is the only floor armed by default; the rest
    #: are opt-in per camera, because a guessed floor costs guests off an
    #: invoice (see heco_counting.gate).
    quality_min_px: float = 56.0
    #: Detector confidence floor; 0 = unarmed. Belongs to the detector family
    #: (yunet 0.8, scrfd 0.5), so there is no safe default to inherit.
    quality_min_conf: float = 0.0
    quality_canon_px: float = 80.0
    quality_min_ied_px: float = 0.0
    quality_min_frontality: float = 0.0
    quality_min_sharpness: float = 0.0
    quality_min_eye_span: float = 0.0
    quality_require_landmarks: bool = False

    #: How often a track that ALREADY holds an identity is re-searched for a
    #: face. 0.0 searches every track every frame.
    face_reverify_interval_s: float = 0.0

    @classmethod
    def from_settings(cls, s) -> "CountingConfig":
        """Read the counting knobs off any settings object that carries them.

        Duck-typed on purpose: this package must not import the runner, and a
        worker or a fusion process will have its own settings type.
        """
        return cls(
            quality_min_px=float(s.quality_min_px),
            quality_canon_px=float(s.quality_canon_px),
            quality_min_ied_px=float(s.quality_min_ied_px),
            quality_min_frontality=float(s.quality_min_frontality),
            quality_min_sharpness=float(s.quality_min_sharpness),
            quality_min_eye_span=float(s.quality_min_eye_span),
            quality_require_landmarks=bool(s.quality_require_landmarks),
            face_reverify_interval_s=float(s.face_reverify_interval_s),
        )


def fold_quality_profile(config, profile: dict | None):
    """Fold a per-run quality profile onto a configuration.

    An OVERRIDE, not a reset: a key the launcher omitted keeps whatever the
    box was configured with, so sending ``{"requireLandmarks": true}`` cannot
    silently disarm a frontality floor an engineer set on that machine. A key
    sent as null is treated as omitted for the same reason — the wire shape
    has optional fields, and "absent" and "explicitly nothing" mean the same
    thing to an operator.

    Unknown keys are ignored rather than rejected: the launcher and the runner
    deploy separately, and a console one version ahead must not be able to
    fail a run over a field the box has not learned yet.

    Generic over any frozen dataclass, so the runner can keep folding onto its
    full ``Settings`` while a worker folds onto a ``CountingConfig`` — one rule,
    two callers, no chance of them drifting apart.
    """
    if not profile:
        return config
    changes = {}
    for wire, field in QUALITY_FIELDS.items():
        value = profile.get(wire)
        if value is None:
            continue
        if hasattr(config, field):
            changes[field] = value
    return replace(config, **changes) if changes else config


def gate_config(config, armed) -> dict:
    """The gate a run enforced, for its permanent record.

    The count is an invoice figure, so "which floors were in force when this
    number was produced" has to be recoverable from the record months later —
    not inferred from whatever the environment happens to hold at the time
    somebody asks. Unarmed floors are recorded as 0.0 rather than omitted, so
    a config with no IED key means an OLD run, not an unarmed one.
    """
    return {
        "qualityMinPx": config.quality_min_px,
        "qualityMinConf": getattr(config, "quality_min_conf", 0.0),
        "qualityCanonPx": config.quality_canon_px,
        "qualityMinIedPx": config.quality_min_ied_px,
        "qualityMinFrontality": config.quality_min_frontality,
        "qualityMinSharpness": config.quality_min_sharpness,
        "qualityMinEyeSpan": config.quality_min_eye_span,
        "qualityRequireLandmarks": config.quality_require_landmarks,
        "faceReverifyIntervalS": config.face_reverify_interval_s,
        "gateArmed": list(armed),
    }
