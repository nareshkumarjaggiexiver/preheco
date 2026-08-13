"""The seams the counting decisions reach the world through.

Two protocols, deliberately small. The counting logic must be usable by three
different hosts — today's runner, a future per-camera worker, a future
per-gate fusion process — and each has its own idea of what a status counter
or a stats board is. So the library states what it needs and nothing more.

WHAT IS ABSENT IS THE DESIGN. There is no ``runId`` in any signature here.
Gallery identity is bound when a port is CONSTRUCTED, not passed per call, so
a caller cannot address another run's gallery by accident — which is the kind
of mistake that merges two events' guests into one number and is unrecoverable
afterwards.
"""

from typing import Protocol


class CountingObserver(Protocol):
    """The library's write-only channel to whatever observability the host has.

    Write-only on purpose: the counting logic reports what it did, and never
    reads anything back through here. A decision that depended on an observer
    would be a decision that changed when you turned logging off.
    """

    def bump(self, key: str, by: int = 1) -> None:
        """Add to a named status counter (``gatedByWidth``, ``excludedByZone``)."""
        ...

    def observe(self, stage: str, metric: str, value: float) -> None:
        """Record one metric observation under a pipeline stage."""
        ...

    def event(self, text: str) -> None:
        """Note something this FRAME did, for the decision ledger.

        The one observer method inside the golden diff: these strings ride the
        frame record, so their wording and their ORDER are part of what a
        refactor must not change.
        """
        ...


class MatchPort(Protocol):
    """Everything the counting decisions need from the gallery.

    THE CHOSEN SEAM: a port the library CALLS, rather than intents it returns
    for the caller to perform. The decision map found four places where a
    verdict depends on the REPLY to a gallery call made mid-frame — a
    same-key split re-resolves one of two faces and splices the answer back
    into this frame's verdicts before anything reads them, and a heal must
    know whether a merge actually happened before it decrements the count.
    Returning intents would force those into a second pass and change the
    order decisions are made in, which is precisely what the extraction is
    forbidden from doing.
    """

    def match(
        self,
        *,
        embedding: list[float],
        quality: float,
        appearance: list[float] | None = None,
        site_id: str | None = None,
        exclude_keys: list[str] | None = None,
    ) -> dict:
        """Resolve one embedding against the gallery; returns the verdict."""
        ...

    def merge(self, *, doomed: str, keeper: str) -> dict:
        """Fold one identity into another; returns whether it happened."""
        ...

    def split(self, *, a: str, b: str) -> dict:
        """Assert two identities are different people (co-presence)."""
        ...

    def forget_template(self, *, template_id: int) -> dict:
        """Retract a template that turned out to belong to somebody else.

        Addressed by TEMPLATE id alone, with no person key. The template is
        the thing being retracted and the gallery already knows whose it is —
        naming the person too would let a caller retract a template from the
        wrong identity's record, which is unrecoverable. (Learned by wiring
        it: an earlier draft of this signature sent a personKey the service
        does not accept, and the runner's own test refused the wire change.)
        """
        ...


class MatchRefused(RuntimeError):
    """The gallery declined, and the DISTINCTION is a decision input.

    A 4xx means the gallery has settled the question — the pair is already
    known to be different people, the merge is impossible — and the caller
    must remember that and stop asking. A 5xx or a transport failure means
    nothing was decided, and the caller must leave its state untouched so the
    next frame retries.

    Conflating them costs both directions: treating a 4xx as transient
    re-sends a settled split on every one of the next fifty frames the pair
    share, and treating a 5xx as settled silently abandons a fold that should
    have happened. So the status code is promoted to a library type rather
    than left as an error detail.
    """

    def __init__(self, message: str, status_code: int = 0) -> None:
        """Carry the reply's status alongside its text."""
        super().__init__(message)
        self.status_code = status_code

    @property
    def settled(self) -> bool:
        """True when the gallery ANSWERED and the answer was no."""
        return 400 <= self.status_code < 500
