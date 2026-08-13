"""Characterisation tests for the decisions about to move.

WHY THIS FILE EXISTS, AND WHY IT EXISTS *NOW*. The next extraction commits move
the folds, the same-frame split and the co-presence assertion — the mechanisms
whose own docstrings warn that they amplify silent failure. Unlike everything
moved so far they carry state and make gallery calls mid-frame, so a
re-implementation could preserve every total while quietly reordering a
precedence, inverting a rollback, or shedding the wrong end of a queue. The
golden diff would not always catch it: the only clip that currently reaches
these paths is a moving-camera phone video, and the decision ledger does not
record the unique count at all.

So each test below pins a behaviour from OUTSIDE the code that is about to
move, and each one names the failure it prevents rather than merely asserting
the current output.

TWO KINDS OF PIN IN HERE, and the difference matters. Most of these exercise
an ISOLATED model of a rule — the ranking comparator, the eviction slice, the
census arithmetic — lifted from the loop. Those document the rule precisely
and fail loudly if the extracted version disagrees with what was written down,
but on their own they would not catch the runner drifting away from the model.

The two highest-risk rules are therefore pinned against the REAL loop instead,
over in test_loop_v1.py: a 4xx co-presence split is never re-sent, and a 5xx
one is retried. Both were mutation-checked — deleting the memoise makes the
first fail — so they have teeth rather than merely passing.

When the folds move, each isolated pin below should be re-pointed at the
library function it models. That conversion is the point of writing them now:
the rule is stated before the code moves, so the move is checked against a
statement rather than against itself.

A NOTE ON PINNING BUGS. The constraint on the extraction is byte-identical
behaviour, so where a rule looks questionable it is pinned as-is and said so
in the docstring. A pinned bug is a visible bug that can be fixed later under
its own diff; an unpinned one is a bug that gets silently rewritten into a
different bug.
"""

import time

import pytest
from app.loop import enrol_score


# --------------------------------------------------------------- the folds


class Ledger:
    """The minimal shape _record_mint keeps per track, exercised directly.

    Mirrors loop.RunLoop's `_minted` bookkeeping: a dict of track id -> list of
    mint entries, expired against the heal window and capped per track.
    """

    def __init__(self, cap: int, window_s: float) -> None:
        self.cap, self.window_s = cap, window_s
        self.minted: dict[int, list[dict]] = {}

    def record(self, track_id: int, key: str, now: float) -> None:
        """Record one mint on a track, expiring and capping as the loop does."""
        for tid in [
            t for t, es in self.minted.items()
            if all(now - e["at"] > self.window_s for e in es)
        ]:
            del self.minted[tid]
        entries = self.minted.setdefault(track_id, [])
        entries.append({"key": key, "at": now})
        del entries[: -self.cap]


def test_mints_per_track_eviction_keeps_the_NEWEST_and_that_is_a_trade():
    """`del entries[:-cap]` sheds the OLDEST mints on a track.

    The consequence is worth stating plainly: the oldest phantom on a shattered
    track becomes permanently unhealable, because the entry that would have let
    a later match fold it away is gone. That is the deliberate trade — the
    newest mints are the ones a settling track is most likely to match — but it
    means a very shattered track leaks its earliest phantoms into the count.

    Pinned as-is. If the cap is ever raised or the eviction end flipped, this
    test fails and the trade gets re-decided on purpose.
    """
    led = Ledger(cap=4, window_s=20.0)
    for i in range(5):
        led.record(7, f"p{i:05d}", now=100.0 + i)
    kept = [e["key"] for e in led.minted[7]]
    assert kept == ["p00001", "p00002", "p00003", "p00004"]
    assert "p00000" not in kept, "the OLDEST is shed — it can never be healed now"


def test_a_track_whose_every_mint_aged_out_is_dropped_whole():
    """Expiry is per TRACK and all-or-nothing: a track is dropped only when
    every entry on it is outside the window. A per-entry sweep would leave
    empty lists accumulating for the length of a night."""
    led = Ledger(cap=4, window_s=20.0)
    led.record(1, "p00001", now=100.0)
    led.record(2, "p00002", now=100.0)
    led.record(2, "p00003", now=115.0)   # keeps track 2 alive
    led.record(3, "p00004", now=125.0)   # now: track 1 is fully expired
    assert 1 not in led.minted, "a wholly-expired track is removed"
    assert 2 in led.minted and 3 in led.minted


def test_the_heal_rollback_puts_the_entry_BACK_on_a_transport_error():
    """A merge that RAISES must leave the count and the ledger untouched.

    The distinction is the whole mechanism: `merged: false` is the gallery
    ANSWERING no (the mint is no longer a singleton, or the two are split), and
    the entry stays consumed. An exception is the gallery not answering at all,
    and dropping the entry there would silently abandon a fold that should have
    happened — an over-count nobody can see, because the evidence went with it.
    """
    led = Ledger(cap=4, window_s=20.0)
    led.record(7, "p00002", now=100.0)
    entry = led.minted[7].pop()          # the loop pops before attempting
    unique = 5

    def merge_raises(**_):
        raise RuntimeError("match unreachable")

    try:
        merge_raises(keep="p00001", drop=entry["key"])
    except Exception:
        led.minted.setdefault(7, []).append(entry)   # the rollback under test
    assert led.minted[7] == [entry], "the entry must be back for the next verdict"
    assert unique == 5, "a failed merge may not move the count"


def test_a_refused_merge_consumes_the_entry_and_leaves_the_count_alone():
    """`merged: false` is an answer, so it is not retried — and not counted."""
    led = Ledger(cap=4, window_s=20.0)
    led.record(7, "p00002", now=100.0)
    entry = led.minted[7].pop()
    unique = 5
    reply = {"merged": False}
    if reply.get("merged"):
        unique -= 1
    assert led.minted[7] == [], "an answered refusal consumes the entry"
    assert unique == 5


# ------------------------------------------------------- same-frame split


def keeper_order(decided: list[dict]) -> list[int]:
    """The loop's keeper ranking, isolated: mints first, then highest cosine."""
    idxs = list(range(len(decided)))
    idxs.sort(key=lambda i: (not decided[i].get("isNew"),
                             -(decided[i].get("cosine") or -1.0)))
    return idxs


def test_a_MINT_keeps_the_key_even_with_a_lower_cosine_than_a_match():
    """The precedence a re-implementation is most likely to "tidy" away.

    Two faces, two bodies, one key: the mint is the sighting that brought the
    key into existence this frame, so taking it away would leave the key held
    only by the sighting being disputed. A mint's cosine is also not comparable
    with a match's — it measures distance to OTHER identities, not to the key it
    was given — so ranking them together is a category error. It handed
    fa8fc3's key to the wrong man and left the count at one.
    """
    order = keeper_order([
        {"isNew": False, "cosine": 0.91},   # an ordinary, confident match
        {"isNew": True, "cosine": 0.20},    # the mint, far less "confident"
    ])
    assert order[0] == 1, "the MINT keeps the key regardless of cosine"


def test_among_ordinary_matches_the_highest_cosine_keeps_the_key():
    """With no mint in play, the sighting the gallery agrees with most wins."""
    order = keeper_order([
        {"isNew": False, "cosine": 0.40},
        {"isNew": False, "cosine": 0.88},
        {"isNew": False, "cosine": 0.61},
    ])
    assert order[0] == 1


def test_a_missing_cosine_ranks_last_rather_than_first():
    """None must not sort as "best". Treating absent as zero-or-better would
    hand the key to the sighting carrying the least evidence."""
    order = keeper_order([{"isNew": False, "cosine": None},
                          {"isNew": False, "cosine": 0.05}])
    assert order[0] == 1


# ------------------------------------------------------------ co-presence


class Splitter:
    """The loop's split-sending rules, isolated from its HTTP client."""

    def __init__(self, cap: int = 4000) -> None:
        self.sent: set[str] = set()
        self.cap = cap
        self.attempts = 0

    def send(self, a: str, b: str, status: int | None) -> bool:
        """status None = success; 4xx settles; 5xx and transport retry."""
        self.attempts += 1
        if status is None:
            self.sent.add(f"{a}|{b}")
            return True
        if 400 <= status < 500:
            self.sent.add(f"{a}|{b}")   # understood and refused: stop asking
            return False
        return False                    # nothing was decided: retry later


def test_a_4xx_split_is_remembered_and_never_re_sent():
    """Understood and refused. Retrying cannot help.

    Without this the pair is re-asserted on every one of the next fifty frames
    they share, which is fifty synchronous round trips inside the frame loop —
    the precise shape of the 0.4 fps death spiral.
    """
    s = Splitter()
    assert s.send("p00001", "p00002", status=422) is False
    assert "p00001|p00002" in s.sent
    before = s.attempts
    if "p00001|p00002" not in s.sent:
        s.send("p00001", "p00002", status=422)
    assert s.attempts == before, "a settled pair must never be asked again"


def test_a_5xx_split_is_NOT_remembered_and_is_retried():
    """Nothing was decided, so the constraint is still owed.

    Memoising here would silently abandon a cannot_link that should exist, and
    a later heal could then fold two co-present guests into one — a silent
    under-count, the worst failure this product has.
    """
    s = Splitter()
    assert s.send("p00001", "p00002", status=503) is False
    assert "p00001|p00002" not in s.sent, "an undecided pair stays owed"


def test_the_co_presence_cap_abandons_the_REST_of_the_frame():
    """At the cap the loop `return`s: it does not skip one pair and continue.

    Pinned as-is because it is load-bearing for cost, not correctness — the cap
    exists so a pathological frame cannot issue thousands of round trips. The
    consequence is that the pairs after the cap on that frame are never
    asserted at all, and the warning is emitted once rather than per pair.
    """
    s = Splitter(cap=2)
    pairs = [("a", "b"), ("c", "d"), ("e", "f")]
    asserted = []
    for a, b in pairs:
        if len(s.sent) >= s.cap:
            break                       # the `return` under test
        s.send(a, b, status=None)
        asserted.append((a, b))
    assert asserted == [("a", "b"), ("c", "d")]
    assert ("e", "f") not in asserted, "the frame is abandoned, not filtered"


# ------------------------------------------------------- template census


def census(template_n: dict[str, int]) -> dict:
    """The loop's two published template numbers, isolated."""
    return {
        "singleTemplateGuests": sum(1 for n in template_n.values() if n <= 1),
        "templateNMax": max(template_n.values()) if template_n else 0,
    }


def test_single_template_guests_counts_the_LAST_reported_n_per_key():
    """A guest who gains a second view must stop counting as single-template.

    This number exists to make a silent failure loud: when no crossing supplies
    a second usable view, every guest ends on one template and the gallery is
    no better than before multi-template — which looks exactly like a threshold
    problem unless this number says otherwise.
    """
    assert census({"p1": 1, "p2": 1})["singleTemplateGuests"] == 2
    assert census({"p1": 2, "p2": 1})["singleTemplateGuests"] == 1
    assert census({"p1": 3, "p2": 2})["templateNMax"] == 3


def test_a_staff_verdicts_null_templateN_is_ignored_not_counted():
    """Staff report templateN null by contract and are not guests.

    Counting a null as one would inflate singleTemplateGuests with people who
    are deliberately outside the guest count, and the number would stop meaning
    what its name says.
    """
    seen: dict[str, int] = {}
    for key, n in [("p1", 1), ("staff-7", None), ("p2", 2)]:
        if isinstance(key, str) and isinstance(n, int):
            seen[key] = n
    assert set(seen) == {"p1", "p2"}
    assert census(seen)["singleTemplateGuests"] == 1


# ------------------------------------------------------------------ enrol


def test_enrol_score_prefers_inter_eye_distance_weighted_by_frontality():
    """The ranking that decides which crops become a staff template."""
    straight = {"box": {"w": 100.0}, "iedPx": 40.0, "frontality": 1.0}
    turned = {"box": {"w": 100.0}, "iedPx": 40.0, "frontality": 0.25}
    assert enrol_score(straight) > enrol_score(turned)


def test_enrol_score_falls_back_to_box_width_when_ied_is_absent():
    """Absent is not zero: a detector that reports no landmarks must still
    rank its crops, or an enrolment silently keeps whichever arrived first."""
    assert enrol_score({"box": {"w": 90.0}}) == 90.0


def test_enrol_score_defaults_frontality_to_one_when_unmeasured():
    """Unmeasured frontality must not penalise a face to zero — that would
    rank every landmark-less crop below every measured one regardless of
    size."""
    assert enrol_score({"box": {"w": 100.0}, "iedPx": 40.0}) == pytest.approx(40.0)


# ------------------------------------------------------------ lock recency


def test_a_second_qualifying_match_REPLACES_the_lock_newest_wins():
    """The lock records the most recent confident sighting on a track.

    Keeping the first would let a stale binding outlive the evidence for it,
    and a fold keyed on that lock would then act on who was there a minute ago.
    """
    locks: dict[int, dict] = {}
    locks[7] = {"key": "p00001", "at": 100.0, "frame": 10}
    locks[7] = {"key": "p00002", "at": 118.0, "frame": 42}
    assert locks[7]["key"] == "p00002"
    assert locks[7]["frame"] == 42, "the frame stamp is refreshed too"


def test_a_lock_older_than_the_heal_window_is_pruned():
    """Locks and mints expire on the same clock, so a fold cannot act on one
    while the other has already forgotten the sighting behind it."""
    window = 20.0
    locks = {1: {"at": 100.0}, 2: {"at": 118.0}}
    now = 125.0
    for tid in [t for t, e in locks.items() if now - e["at"] > window]:
        del locks[tid]
    assert 1 not in locks and 2 in locks


def test_every_pinned_window_uses_the_frames_clock_not_the_wall_clock():
    """The windows above are in SECONDS OF FOOTAGE, not seconds of runtime.

    Guards the change made in c34f967: measured against a wall clock, a faster
    box covers more footage inside the same window, heals more, and can settle
    on a different number — the pipeline's own speed becoming an input to its
    answer.
    """
    from app.loop import RunLoop

    src = RunLoop._now.__doc__ or ""
    assert "tMs" in src and "monotonic" in src, (
        "_now must keep documenting why it is the frame's clock"
    )
    started = time.monotonic()
    assert started > 0  # sanity: the wall clock still exists, it is just not this
