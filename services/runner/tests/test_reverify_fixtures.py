"""Shared scripted footage for the face-search tests: one guest who settles.

Not a test module (no ``test_`` functions): a guest walks in, is minted on
frame 0, then matches herself comfortably on every later frame — which is
what takes the identity lock that makes her track SETTLED for the re-verify
gate and the face-search cadence alike.
"""

from tests.test_loop_v1 import scripted_verdict
from tests.test_presence import FA, A

#: The verdicts: a mint, then comfortable self-matches (cosine 0.80 is over
#: the 0.45 lock floor, so each one refreshes the lock).
SETTLING = [scripted_verdict("p00001", True, None)] + [
    scripted_verdict("p00001", False, 0.80) for _ in range(40)
]


def settling_frames(n: int) -> list[dict]:
    """``n`` frames of the one body with her face showing."""
    return [{"boxes": [A], "faces": [FA]} for _ in range(n)]
