"""Boundary tests for the POC quality flag (80 / 56 px per CONTRACTS.md)."""

import pytest

from app.quality import classify_width


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (120, "ok"),
        (80, "ok"),  # canon boundary inclusive
        (79.9, "sub-canon"),
        (79, "sub-canon"),
        (64, "sub-canon"),  # POC expected range 64-85 px lands here or in ok
        (56, "sub-canon"),  # floor boundary inclusive
        (55.9, "reject"),
        (10, "reject"),
    ],
)
def test_default_thresholds(width, expected):
    assert classify_width(width) == expected


def test_env_style_overrides():
    assert classify_width(70, canon_px=60, floor_px=40) == "ok"
    assert classify_width(50, canon_px=60, floor_px=40) == "sub-canon"
    assert classify_width(30, canon_px=60, floor_px=40) == "reject"
