"""The shared-frame transport, and the properties it is chosen FOR."""

import os

import numpy as np
import pytest

from heco_common import frameref as fr


@pytest.fixture
def shared(tmp_path, monkeypatch):
    monkeypatch.setenv(fr.FRAMES_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(fr.KEEP_ENV, "3")
    return tmp_path


def _img(w=200, h=120):
    return np.random.randint(0, 255, (h, w, 3), np.uint8)


def test_a_frame_survives_the_round_trip_byte_for_byte(shared):
    """The whole point: the consumer gets the SAME pixels, with no codec."""
    img = _img()
    ref = fr.write_frame(img, 1)
    assert ref == "f1_200x120.bgr"
    assert np.array_equal(fr.read_frame(ref), img)


def test_the_ref_carries_the_shape_so_there_is_no_second_source_of_truth(shared):
    """A sidecar saying 1920x1080 over bytes that are 3840x2160 is a silent
    reshape into garbage. The name IS the metadata, so they cannot disagree."""
    ref = fr.write_frame(_img(64, 48), 7)
    assert ref == "f7_64x48.bgr"
    assert fr.read_frame(ref).shape == (48, 64, 3)


def test_frames_past_the_keep_window_are_retired(shared):
    """Unbounded would be a tmpfs (RAM) filling at 24 MB per 4K frame."""
    old = fr.write_frame(_img(), 1)
    for seq in range(2, 8):
        fr.write_frame(_img(10, 10), seq)
    assert fr.read_frame(old) is None
    assert not (shared / old).exists()


def test_a_reader_holding_the_file_is_not_disturbed_by_retirement(shared):
    """THE property the naming scheme exists for.

    A ring of reused slots would need leasing and a fence to stop a producer
    overwriting a frame mid-read. Here each frame is its own inode, and POSIX
    keeps an unlinked inode alive until the last fd closes — so retiring a
    frame somebody is reading cannot corrupt that read. This test is what
    makes 'torn reads are impossible' a claim rather than a hope.
    """
    img = _img()
    ref = fr.write_frame(img, 100)
    with open(shared / ref, "rb") as fh:
        for seq in range(101, 120):        # retire it out from under the reader
            fr.write_frame(_img(10, 10), seq)
        assert not (shared / ref).exists()
        data = fh.read()                   # ... and read it anyway
    assert np.frombuffer(data, np.uint8).reshape(img.shape).tobytes() == img.tobytes()


def test_only_names_we_wrote_are_ever_opened(shared):
    """The ref arrives over the wire from another service. It names a file
    this process will open, so it is validated as a NAME WE MINT and never
    used as a path — otherwise the transport is a file-read primitive."""
    (shared.parent / "secret").write_text("not a frame")
    assert fr.read_frame("../secret") is None
    assert fr.read_frame("/etc/passwd") is None
    assert fr.read_frame("whatever.bgr") is None
    assert fr.read_frame("") is None


def test_a_truncated_frame_reads_as_absent_not_as_pixels(shared):
    """Length is checked against the shape in the name: a short file is a
    fallback to JPEG, never a reshape error or a frame of stale memory."""
    ref = fr.write_frame(_img(), 3)
    (shared / ref).write_bytes(b"\x00" * 10)
    assert fr.read_frame(ref) is None


def test_the_whole_path_is_off_unless_configured(monkeypatch):
    """No mount, no behaviour change: every producer still sends imageB64,
    so an unconfigured or mixed-version deploy is the old pipeline exactly."""
    monkeypatch.delenv(fr.FRAMES_DIR_ENV, raising=False)
    assert fr.frames_dir() is None
    assert fr.write_frame(_img(), 1) is None
    assert fr.read_frame("f1_200x120.bgr") is None


def test_a_missing_directory_is_off_rather_than_an_error(tmp_path, monkeypatch):
    """A typo'd mount point must degrade to JPEG, not crash a live count."""
    monkeypatch.setenv(fr.FRAMES_DIR_ENV, str(tmp_path / "nope"))
    assert fr.frames_dir() is None
    assert fr.write_frame(_img(), 1) is None


def test_a_non_contract_array_declines_rather_than_writing_something_odd(shared):
    """Greyscale or float frames are not the BGR uint8 contract; the writer
    says no and the JPEG path carries them, instead of writing bytes whose
    shape the name would misdescribe."""
    assert fr.write_frame(np.zeros((10, 10), np.uint8), 1) is None
    assert fr.write_frame(np.zeros((10, 10, 3), np.float32), 1) is None


def test_partial_writes_are_never_visible(shared):
    """Write-then-rename: a consumer polling the directory sees a frame only
    once all of it is there. The temp name is dot-prefixed so it cannot match
    the sweeper's pattern either."""
    fr.write_frame(_img(), 5)
    assert [p.name for p in shared.iterdir()] == ["f5_200x120.bgr"]
    assert not any(p.name.startswith(".") for p in shared.iterdir())


def test_the_sweeper_only_touches_frames(shared):
    """It unlinks by pattern in a shared directory. A pattern that could match
    anything else is a sweeper that deletes somebody else's file."""
    keeper = shared / "not-a-frame.txt"
    keeper.write_text("leave me")
    for seq in range(1, 20):
        fr.write_frame(_img(10, 10), seq)
    assert keeper.exists()


def test_write_then_read_survives_a_gap_in_sequence_numbers(shared):
    """The runner drops frames by design on a live source, so sequence
    numbers arrive with holes. Retention is relative to the NEWEST seq, not a
    count of files, or a burst of drops would retire frames still in flight."""
    ref = fr.write_frame(_img(), 1000)
    assert fr.read_frame(ref) is not None
    fr.write_frame(_img(10, 10), 1002)
    assert fr.read_frame(ref) is not None, "still within keep=3 of the newest"
    fr.write_frame(_img(10, 10), 1010)
    assert fr.read_frame(ref) is None, "now well behind"


def test_the_directory_can_be_passed_explicitly_for_tests_and_tools(tmp_path, monkeypatch):
    """The env is the deployment's answer; an explicit directory is what lets
    a test or a one-off tool use this without setting process-wide state."""
    monkeypatch.delenv(fr.FRAMES_DIR_ENV, raising=False)
    img = _img(32, 32)
    ref = fr.write_frame(img, 1, directory=tmp_path)
    assert ref is not None
    assert np.array_equal(fr.read_frame(ref, directory=tmp_path), img)


def test_frames_are_written_to_the_configured_directory(shared):
    """Sanity: the bytes land where the mount is, not in a cwd nobody shares."""
    fr.write_frame(_img(16, 16), 42)
    assert (shared / "f42_16x16.bgr").exists()
    assert os.environ[fr.FRAMES_DIR_ENV] == str(shared)


def test_a_full_directory_can_recover(shared, monkeypatch):
    """REGRESSION (2026-09-24). The sweep ran only AFTER a successful write,
    so a full tmpfs failed the write, skipped the sweep, and could never
    retire anything again — the mount sat at 98% holding 21 frames against a
    keep of 8, and every frame from then on fell back to JPEG silently.

    Simulated here by failing the write once: the sweep must still have run,
    because making room is a precondition, not a reward.
    """
    for seq in range(1, 12):
        fr.write_frame(_img(10, 10), seq)
    assert len(list(shared.glob("f*.bgr"))) <= 4, "keep=3 plus the newest"

    real_open = open
    calls = {"n": 0}

    def _failing_open(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("No space left on device")
        return real_open(*args, **kwargs)

    # Only the FIRST open fails; undo() is deliberately not used, because it
    # would also revert the fixture's HECO_FRAMES_DIR and the next write would
    # return None for an entirely different reason.
    monkeypatch.setattr("builtins.open", _failing_open)
    assert fr.write_frame(_img(10, 10), 99) is None, "the failed write reports failure"
    # and the sweep ran anyway, so the next write has room
    assert fr.write_frame(_img(10, 10), 100) is not None
    assert len(list(shared.glob("f*.bgr"))) <= 4
