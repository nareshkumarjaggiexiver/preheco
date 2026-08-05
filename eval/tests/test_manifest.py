"""Tests for the manifest loader — every rule that protects a score."""

import json
import re

import pytest

from eval.manifest import KNOWN_TAGS, ManifestError, load_manifest, parse_manifest

GOOD = {
    "version": 1,
    "eventId": "evt-7",
    "clips": [
        {
            "label": "baraat-surge",
            "path": "/srv/clips/surge.mp4",
            "actualUnique": 61,
            "tags": ["surge"],
        },
        {"label": "calm", "path": "/srv/clips/calm.mp4", "actualUnique": 12},
    ],
}


def test_a_good_manifest_parses():
    """The happy path, including manifest-level planner ids."""
    m = parse_manifest(GOOD)
    assert m.labels() == ("baraat-surge", "calm")
    assert m.event_id == "evt-7"
    assert m.clips[0].tags == ("surge",)
    assert m.clips[1].tags == ()


def test_duplicate_labels_are_refused_because_they_are_the_ab_join_key():
    """Two clips sharing a label would silently pair the wrong runs."""
    raw = {**GOOD, "clips": [GOOD["clips"][0], {**GOOD["clips"][1], "label": "baraat-surge"}]}
    with pytest.raises(ManifestError, match="duplicate label"):
        parse_manifest(raw)


def test_relative_paths_are_refused():
    """Ingest opens the path, usually from inside a container with its own mounts."""
    raw = {**GOOD, "clips": [{**GOOD["clips"][0], "path": "clips/surge.mp4"}]}
    with pytest.raises(ManifestError, match="must be absolute"):
        parse_manifest(raw)


def test_a_clip_with_nobody_in_it_is_refused():
    """The headline metric is relative; zero has no denominator."""
    raw = {**GOOD, "clips": [{**GOOD["clips"][0], "actualUnique": 0}]}
    with pytest.raises(ManifestError, match="at least 1"):
        parse_manifest(raw)


def test_a_non_integer_count_of_record_is_refused():
    """'about forty' is not a count of record."""
    raw = {**GOOD, "clips": [{**GOOD["clips"][0], "actualUnique": "61"}]}
    with pytest.raises(ManifestError, match="whole number"):
        parse_manifest(raw)


def test_a_typo_in_a_tag_is_refused_rather_than_quietly_creating_a_new_one():
    """A stray 'veils' would make the per-tag breakdown lie about coverage."""
    raw = {**GOOD, "clips": [{**GOOD["clips"][0], "tags": ["veils"]}]}
    with pytest.raises(ManifestError, match="unknown tag"):
        parse_manifest(raw)


def test_every_hard_case_from_the_protocol_is_a_known_tag():
    """§5 names these five; the manifest must be able to express all of them."""
    assert {"surge", "veil", "children", "staff", "evening-light"} == KNOWN_TAGS


def test_an_empty_manifest_is_refused():
    """Nothing to score is a configuration error, not an empty success."""
    with pytest.raises(ManifestError, match="non-empty 'clips'"):
        parse_manifest({"version": 1, "clips": []})


def test_an_unsupported_version_is_refused():
    """A future format must fail loudly rather than be half-read."""
    with pytest.raises(ManifestError, match="not supported"):
        parse_manifest({"version": 99, "clips": GOOD["clips"]})


def test_load_manifest_names_the_file_in_its_errors(tmp_path):
    """An operator needs to know WHICH manifest is wrong."""
    path = tmp_path / "clips.json"
    path.write_text(json.dumps({**GOOD, "clips": [{"label": "x"}]}), encoding="utf-8")
    with pytest.raises(ManifestError, match=re.escape(str(path))):
        load_manifest(path)


def test_load_manifest_reads_a_real_file(tmp_path):
    """Round trip through the filesystem, source path recorded for the report."""
    path = tmp_path / "clips.json"
    path.write_text(json.dumps(GOOD), encoding="utf-8")
    m = load_manifest(path)
    assert m.source_path == str(path)
    assert len(m.clips) == 2


def test_the_shipped_example_manifest_is_valid():
    """The documented example must actually load, or it teaches the wrong shape."""
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "manifest.example.json"
    m = load_manifest(example)
    assert len(m.clips) == 5
    assert {t for c in m.clips for t in c.tags} == KNOWN_TAGS
