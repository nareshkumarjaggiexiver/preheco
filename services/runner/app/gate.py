"""The quality gate: which detected faces are worth embedding, and WHY not.

This is the pipeline's only irreversible discard.  A face the gate rejects is
never embedded, never matched, and therefore never counted — and the count is
an invoice figure.  So the gate is a pure, unit-tested function that lives
apart from the loop, and it reports a REASON for every rejection instead of
silently shortening a list.

**Why it is composite.**  The gate was box width alone.  Width is the wrong
axis in two directions at once:

* Recognition size is honestly read in INTER-EYE DISTANCE, which is what FR
  standards specify and what SFace actually consumes after alignment.  Our
  56/80 px width floor maps to only ~24/34 px IED, so the number in the config
  is not the number that decides whether an embedding is usable.
* Size is not the only thing that decides it.  A wide face turned side-on has
  little of the identity-bearing geometry left, and a wide face smeared by a
  guest walking past has none of the detail — both embed badly at any width.
  ``frontality`` and ``sharpness`` see exactly those two failures and width
  sees neither.

**Why every threshold defaults to off.**  We have no measured floor for any of
the three yet (the eval harness that would price them does not exist), and
guessing one costs guests off an invoice.  So each new signal is armed only by
an explicit env override, and with none set the gate is byte-for-byte the
width-only gate it has always been.

**Why a missing signal never rejects.**  A face without landmarks carries no
``iedPx`` and no ``frontality``; a degenerate crop carries no ``sharpness``.
Absence means UNKNOWN, not BAD.  Under-counting is this pipeline's dominant
failure mode (open-set 1:N matching at a 1:1 verification threshold), so an
unmeasured signal must not become a third way to lose a guest.  The faces that
were admitted with an armed signal unmeasured are reported instead
(:attr:`Verdict.unmeasured`), because an operator who has armed an IED floor
needs to know if a third of the faces are slipping past it unmeasured.
"""

from dataclasses import dataclass

#: Rejection reasons, in the order the signals are tested.  A face is reported
#: against the FIRST signal it fails, not all of them: the console shows one
#: label per face, and the cheapest, oldest, best-understood test goes first.
REASONS = ("width", "ied", "frontality", "sharpness")


@dataclass(frozen=True)
class GateThresholds:
    """One resolved floor per signal; 0.0 (or below) means NOT ARMED.

    Kept separate from ``Settings`` so the gate can be exercised without the
    runner's whole configuration, and so the "armed" question has exactly one
    answer in one place.
    """

    min_px: float = 56.0
    min_ied_px: float = 0.0
    min_frontality: float = 0.0
    min_sharpness: float = 0.0

    @classmethod
    def from_settings(cls, s) -> "GateThresholds":
        """Read the four floors off a runner Settings snapshot."""
        return cls(
            min_px=float(s.quality_min_px),
            min_ied_px=float(s.quality_min_ied_px),
            min_frontality=float(s.quality_min_frontality),
            min_sharpness=float(s.quality_min_sharpness),
        )

    @property
    def armed(self) -> tuple[str, ...]:
        """The signals beyond box width that this configuration gates on.

        Empty for the default configuration, which is the whole point: the
        composite gate changes nothing until an operator opts in.
        """
        return tuple(
            name
            for name, floor in (
                ("ied", self.min_ied_px),
                ("frontality", self.min_frontality),
                ("sharpness", self.min_sharpness),
            )
            if floor > 0.0
        )


@dataclass(frozen=True)
class Verdict:
    """What the gate decided about one face, and on what evidence.

    ``reason`` is None when the face passes.  ``unmeasured`` names the ARMED
    signals this face did not carry — it passed those tests by absence, not by
    merit, and a gate that cannot say so is a gate nobody should trust.
    """

    reason: str | None = None
    unmeasured: tuple[str, ...] = ()

    @property
    def kept(self) -> bool:
        """True when this face goes on to be embedded and matched."""
        return self.reason is None


def _signal(face: dict, key: str) -> float | None:
    """Read one measured signal off a face, or None when it is not there.

    A non-numeric value from a stage that has drifted is treated as absent
    rather than raising: the gate runs inside the frame loop, and crashing a
    live count over a malformed debug field would be a far worse trade.
    """
    if key not in face:
        return None
    try:
        return float(face[key])
    except (TypeError, ValueError):
        return None


def gate_face(face: dict, t: GateThresholds) -> Verdict:
    """Decide one face against every armed floor; report the first failure.

    Width is tested against ``face["box"]["w"]``, which the detector always
    provides; the other three are read from the optional signals the faces
    service emits (``iedPx``, ``frontality``, ``sharpness``).
    """
    if float(face.get("box", {}).get("w", 0.0)) < t.min_px:
        return Verdict("width")

    unmeasured: list[str] = []
    for name, key, floor in (
        ("ied", "iedPx", t.min_ied_px),
        ("frontality", "frontality", t.min_frontality),
        ("sharpness", "sharpness", t.min_sharpness),
    ):
        if floor <= 0.0:
            continue  # not armed
        value = _signal(face, key)
        if value is None:
            unmeasured.append(name)  # unknown is not bad — keep and report
            continue
        if value < floor:
            return Verdict(name, tuple(unmeasured))
    return Verdict(None, tuple(unmeasured))


def reported_reason(face: dict, min_px: float) -> str | None:
    """The verdict to SHOW for a face: the stamped one, else width-only.

    The single place the reporting surfaces (debug taps, annotated frames) are
    allowed to answer "was this face kept?".  They must READ the verdict the
    gate stamped, never re-derive it: a second implementation of the gate in
    the reporting path is a second gate, and the two would disagree about the
    same frame the moment either grew a signal — while the console's number is
    the one an operator argues an invoice from.  The width comparison survives
    only as the fallback for a face list that was never gated at all.
    """
    if "gateReason" in face:
        return face["gateReason"]
    w = float(face.get("box", {}).get("w", 0.0))
    return None if w >= min_px else "width"


@dataclass(frozen=True)
class GateOutcome:
    """One frame's gate result: what survived, what did not, and what it cost.

    ``unmeasured`` counts faces that were KEPT while an armed floor could not
    be evaluated on them.  It is not an error and not a rejection — it is the
    number that tells an operator how much of their gate is actually running.
    """

    kept: list[dict]
    gated_by: dict[str, int]
    unmeasured: int = 0


def gate_faces(faces: list[dict], t: GateThresholds) -> GateOutcome:
    """Gate a frame's faces, stamping each with its verdict.

    Every face is stamped in place with ``gateReason`` (None when kept) and,
    when relevant, ``gateUnmeasured`` — those keys ride the frame snapshot into
    the debug taps and the annotated frame, so the console can show an operator
    WHY a face beside a counted guest was not counted.  Stamping rather than
    returning a parallel list keeps the reason attached to the face across
    three consumers that each re-order or truncate the list.
    """
    kept: list[dict] = []
    gated_by: dict[str, int] = {}
    unmeasured = 0
    for face in faces:
        v = gate_face(face, t)
        face["gateReason"] = v.reason
        if v.unmeasured:
            face["gateUnmeasured"] = list(v.unmeasured)
            unmeasured += 1
        if v.kept:
            kept.append(face)
        else:
            gated_by[v.reason] = gated_by.get(v.reason, 0) + 1
    return GateOutcome(kept, gated_by, unmeasured)
