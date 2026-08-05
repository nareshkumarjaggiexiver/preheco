"""The ground-truth clip manifest — what we score against, and its count of record.

WHY A MANIFEST FILE. The clips themselves are guests' faces: they cannot live
in the repository. What CAN and MUST live here is the claim being made about
each one — its path, a stable human label, and ``actualUnique``, the number of
distinct people a human counted crossing that gate. Without that file checked
in, an "improvement" is whatever the last person remembered, and the A/B has
nothing to join on.

WHY THE VALIDATION IS STRICT. Every rule below exists because breaking it
silently corrupts a score rather than failing:

* **Labels are the join key** for the A/B comparison, so duplicates are
  refused — two clips sharing a label would silently overwrite one another's
  result and the comparison would pair the wrong runs.
* **Paths must be absolute.** The path is not opened by this harness; it is
  handed to the ingest service, which normally runs in a container with its
  own mounts. A relative path would resolve against whichever working
  directory ingest happened to have, which is not a thing anyone can reason
  about at 2 a.m.
* **``actualUnique`` must be at least 1.** The headline metric is relative
  (``(pipeline - actual) / actual``); a clip nobody walked through has no
  denominator and therefore no headline. Use a clip with at least one
  crossing.
* **Tags come from a closed set** (the hard cases §5 of the accuracy report
  names). A typo'd ``veils`` would quietly create a one-clip tag and make the
  per-tag breakdown lie about coverage.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

#: The hard cases the evaluation protocol (accuracy-and-tuning.md §5) names.
#: Closed on purpose — see the module docstring.
KNOWN_TAGS: frozenset[str] = frozenset(
    {"surge", "veil", "children", "staff", "evening-light"}
)

#: Manifest format version this loader understands.
MANIFEST_VERSION = 1


class ManifestError(ValueError):
    """A manifest is unusable; the message always names the offending clip."""


@dataclass(frozen=True)
class Clip:
    """One ground-truth clip: where it is, what it is, and the count of record."""

    label: str
    path: str
    actual_unique: int
    tags: tuple[str, ...] = ()
    notes: str | None = None

    def as_dict(self) -> dict:
        """The clip as it appears inside a result file (traceability)."""
        return {
            "clip": self.label,
            "path": self.path,
            "actualUnique": self.actual_unique,
            "tags": list(self.tags),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Manifest:
    """A validated clip set, plus the planner ids a run needs to be recorded."""

    clips: tuple[Clip, ...]
    event_id: str | None = None
    site_id: str | None = None
    source_path: str | None = None
    description: str | None = None
    unknown_keys: tuple[str, ...] = field(default=())

    def labels(self) -> tuple[str, ...]:
        """Clip labels in manifest order (the A/B join keys)."""
        return tuple(c.label for c in self.clips)


def _require(cond: bool, message: str) -> None:
    """Raise ManifestError unless ``cond`` holds (keeps the checks readable)."""
    if not cond:
        raise ManifestError(message)


def parse_manifest(raw: dict, source_path: str | None = None) -> Manifest:
    """Validate a decoded manifest document and return a :class:`Manifest`.

    Split from :func:`load_manifest` so the validation rules are testable
    without touching a filesystem — the rules are the part that protects a
    score, and they get tested properly.
    """
    _require(isinstance(raw, dict), "manifest must be a JSON object")
    version = raw.get("version", MANIFEST_VERSION)
    _require(
        version == MANIFEST_VERSION,
        f"manifest version {version!r} is not supported (this harness reads "
        f"version {MANIFEST_VERSION})",
    )
    clips_raw = raw.get("clips")
    _require(isinstance(clips_raw, list) and clips_raw, "manifest needs a non-empty 'clips' list")

    clips: list[Clip] = []
    seen: set[str] = set()
    for i, item in enumerate(clips_raw):
        where = f"clips[{i}]"
        _require(isinstance(item, dict), f"{where} must be an object")
        label = item.get("label")
        _require(
            isinstance(label, str) and label.strip() != "",
            f"{where} needs a non-empty 'label' — it is the A/B join key",
        )
        _require(
            label not in seen,
            f"{where}: duplicate label {label!r}; labels must be unique because "
            "the A/B comparison joins clips on them",
        )
        seen.add(label)

        path = item.get("path")
        _require(
            isinstance(path, str) and path.strip() != "",
            f"{where} ({label}) needs a non-empty 'path'",
        )
        _require(
            path.startswith("/"),
            f"{where} ({label}): path {path!r} must be absolute AS THE INGEST "
            "SERVICE SEES IT — ingest opens it, and it usually runs in a "
            "container with its own mounts",
        )

        actual = item.get("actualUnique")
        _require(
            isinstance(actual, int) and not isinstance(actual, bool),
            f"{where} ({label}): actualUnique must be a whole number — the human "
            "count of record for this clip",
        )
        _require(
            actual >= 1,
            f"{where} ({label}): actualUnique must be at least 1; the headline "
            "metric is a relative error and a clip with nobody in it has no "
            "denominator",
        )

        tags_raw = item.get("tags", [])
        _require(
            isinstance(tags_raw, list) and all(isinstance(t, str) for t in tags_raw),
            f"{where} ({label}): tags must be a list of strings",
        )
        unknown = [t for t in tags_raw if t not in KNOWN_TAGS]
        _require(
            not unknown,
            f"{where} ({label}): unknown tag(s) {unknown!r}; known tags are "
            f"{sorted(KNOWN_TAGS)} — a typo would silently split the per-tag "
            "breakdown",
        )
        notes = item.get("notes")
        _require(
            notes is None or isinstance(notes, str),
            f"{where} ({label}): notes must be a string when present",
        )
        clips.append(
            Clip(
                label=label,
                path=path,
                actual_unique=actual,
                tags=tuple(tags_raw),
                notes=notes,
            )
        )

    event_id = raw.get("eventId")
    site_id = raw.get("siteId")
    _require(
        event_id is None or (isinstance(event_id, str) and event_id.strip() != ""),
        "eventId, when present, must be a non-empty string",
    )
    _require(
        site_id is None or (isinstance(site_id, str) and site_id.strip() != ""),
        "siteId, when present, must be a non-empty string",
    )
    known_top = {"version", "clips", "eventId", "siteId", "description"}
    return Manifest(
        clips=tuple(clips),
        event_id=event_id,
        site_id=site_id,
        source_path=source_path,
        description=raw.get("description"),
        unknown_keys=tuple(sorted(set(raw) - known_top)),
    )


def load_manifest(path: str | Path) -> Manifest:
    """Read and validate a manifest file (JSON), naming the file in any error."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest {p} is not valid JSON: {exc}") from exc
    try:
        return parse_manifest(raw, source_path=str(p))
    except ManifestError as exc:
        raise ManifestError(f"{p}: {exc}") from None


def missing_local_paths(manifest: Manifest) -> list[str]:
    """Clip paths that do not exist ON THIS MACHINE.

    Only meaningful when ingest shares this filesystem (a bare-metal run); a
    dockerised ingest sees different mounts, which is why the runner treats
    this as an opt-in check rather than a precondition.
    """
    return [c.path for c in manifest.clips if not Path(c.path).exists()]
