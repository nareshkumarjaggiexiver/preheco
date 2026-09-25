"""Head-covering reads for the review queue: stored logits in, per-identity calls out.

WHY.  Run 8b8b87's review pair p00005/p00009 put a Sikh man in a sky-blue
turban beside a bare-headed man.  The colour-histogram head descriptor cannot
tell a dark turban from black hair, so ``gallery.head_apart`` deliberately
compares only headwear against headwear and never covered against bare.
SigLIP B/16 reads the head semantically: the embed service's reader
(``services/embed/app/headwear.py``) scores two views of the head against a
prompt set and returns, per read, 8 class LOGITS — turban, bare,
dupatta_or_scarf, cap_or_hat for the LOOSE view, then the same four for the
TIGHT (ellipse-masked) view.  The runner writes them onto the sighting's body
row for mints and template enrolments only.

LOGITS ARE STORED, NOT VERDICTS.  The call is made here, at review time, from
config, so its thresholds can move without re-reading a single head.  A read
is CONFIDENTLY turban (bare) when both views argmax that class and the smaller
of the two views' softmax probabilities is at least ``turban_p`` (``bare_p``)
— the reference reader's rule (siglip ``onnx/headwear_ref.py``), in its
float32 arithmetic.  Anything else is unsure; dupatta and cap have no
confident call at all (dupatta had zero examples in the evaluation; caps come
off).

MEASURED (offline, 461 crops labelled by eye from run D02's wedding, CPU):
at turban >= 0.80 / bare >= 0.50 no confident call was wrong — 31 turban
calls, 312 bare — with every uncalled turban unsure, never bare.  One-sided
95 % lower bound on turban precision ~0.91 (31 calls): unproven at 0.97,
which is why the rule that acts on these reads ships LOG-ONLY
(``HECO_REVIEW_HEADWEAR=0``) and is re-verified on the next event first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The wire order of the 4 classes, per view.  The embed service refuses to
#: load a prompt set whose classes are in any other order, so position is
#: meaning on both sides.
HEADWEAR_CLASSES = ("turban", "bare", "dupatta_or_scarf", "cap_or_hat")
#: ...and of the two views: the loose one (sees a whole turban) first.
HEADWEAR_VIEWS = ("loose", "tight")
#: One read on the wire and in the ``body_sightings.headwear`` BLOB.
HEADWEAR_DIM = len(HEADWEAR_CLASSES) * len(HEADWEAR_VIEWS)
_TURBAN = HEADWEAR_CLASSES.index("turban")
_BARE = HEADWEAR_CLASSES.index("bare")


def read_call(logits, turban_p: float, bare_p: float) -> str:
    """One read's call — ``"turban"``, ``"bare"`` or ``"unsure"``.

    Softmax per view (max-shifted, float32 like the reference), argmax per
    view; a call only when both views name the same class, that class has a
    threshold (turban or bare), and min over the views of its probability is
    at least that threshold.  A read that is not 8 finite numbers is unsure.
    """
    z = np.asarray(logits, dtype=np.float32)
    if z.size != HEADWEAR_DIM or not np.all(np.isfinite(z)):
        return "unsure"
    z = z.reshape(len(HEADWEAR_VIEWS), len(HEADWEAR_CLASSES))
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    arg = p.argmax(axis=1)
    if not (arg == arg[0]).all():
        return "unsure"
    cls = int(arg[0])
    tau = {_TURBAN: turban_p, _BARE: bare_p}.get(cls)
    if tau is None or float(p[:, cls].min()) < tau:
        return "unsure"
    return HEADWEAR_CLASSES[cls]


@dataclass(frozen=True)
class HeadwearTally:
    """One identity's head-covering reads: how many, and how many were confident.

    ``turban`` and ``bare`` count CONFIDENT reads; ``n - turban - bare`` were
    unsure (a dupatta, a cap, a turned or small head, the two views
    disagreeing).
    """

    n: int
    turban: int
    bare: int


def headwear_tallies(evidence: list, turban_p: float, bare_p: float) -> dict[str, HeadwearTally]:
    """Each identity's :class:`HeadwearTally` from the body log; absent when unread."""
    counts: dict[str, list[int]] = {}
    for row in evidence:
        vec = getattr(row, "headwear", None)
        if vec is None or vec.size != HEADWEAR_DIM:
            continue
        c = counts.setdefault(row.key, [0, 0, 0])
        c[0] += 1
        call = read_call(vec, turban_p, bare_p)
        if call == "turban":
            c[1] += 1
        elif call == "bare":
            c[2] += 1
    return {k: HeadwearTally(n, t, b) for k, (n, t, b) in counts.items()}


def headwear_label(t: HeadwearTally | None, min_n: int) -> str | None:
    """An identity's head covering, as the review row prints it.

    ``turban`` (``bare``): at least ``min_n`` confident reads of it and ZERO
    of the other; ``mixed``: confident reads of both (the one observed false
    turban is a neighbour's turban inside the crop, and a single clean bare
    read cancels it); ``unsure``: reads, but not enough confident ones; None:
    never read — absent is not zero.
    """
    if t is None or t.n == 0:
        return None
    need = max(1, int(min_n))
    if t.turban >= need and t.bare == 0:
        return "turban"
    if t.bare >= need and t.turban == 0:
        return "bare"
    if t.turban and t.bare:
        return "mixed"
    return "unsure"


def headwear_why(ta: HeadwearTally | None, tb: HeadwearTally | None, min_n: int) -> dict:
    """``why.headwear`` for one pair: each side's label, reads and confident reads."""
    return {
        "a": headwear_label(ta, min_n),
        "b": headwear_label(tb, min_n),
        "nA": 0 if ta is None else ta.n,
        "nB": 0 if tb is None else tb.n,
        "turbanA": 0 if ta is None else ta.turban,
        "bareA": 0 if ta is None else ta.bare,
        "turbanB": 0 if tb is None else tb.turban,
        "bareB": 0 if tb is None else tb.bare,
    }


def headwear_apart(
    ta: HeadwearTally | None,
    tb: HeadwearTally | None,
    gender_a: tuple[str | None, float | None],
    gender_b: tuple[str | None, float | None],
    min_n: int,
    gender_min_p: float,
) -> bool:
    """Does the head covering say these two identities cannot be one man?

    Only a TURBAN against a BARE head, only between two men:

    * both identities confidently male — ``identity_gender`` says M at
      ``gender_min_p`` or more (the review's gender bar; at 0 that bar is
      off, and so is this rule: a woman's dupatta goes on and off within one
      wedding, dupatta had zero examples in the evaluation, and genderage
      reads this camera's children as adults, so age cannot stand in);
    * one identity has at least ``min_n`` confident turban reads and none
      bare, the other at least ``min_n`` confident bare reads and none
      turban (``min_n <= 0`` is off).

    Never dupatta, cap or unsure against anything, and turban against turban
    stays with the colour rule (``gallery.head_apart``).  A safa tied for the
    baraat comes off later in the night — the known limit, why the rule is
    switched separately (``HECO_REVIEW_HEADWEAR``) and ships off.
    """
    if min_n <= 0 or gender_min_p <= 0 or ta is None or tb is None:
        return False
    for sex, p in (gender_a, gender_b):
        if sex != "M" or p is None or p < gender_min_p:
            return False
    labels = {headwear_label(ta, min_n), headwear_label(tb, min_n)}
    return labels == {"turban", "bare"}
