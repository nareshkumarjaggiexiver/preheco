

def test_a_frame_is_written_to_the_shared_transport_once_per_seq(monkeypatch, tmp_path):
    """REGRESSION (2026-09-24). The runner polls /frame at 50 Hz waiting for
    the seq to advance, and every poll returned the same frame — so writing
    on each one put 24 MB into tmpfs fifty times a second for ONE frame.

    The JPEG encode had been hiding it: at 13.9 ms it throttled the polling.
    Remove the encode and the waste ran free, and the 'faster' ref-only path
    measured 2.32 fps against the 5.14 of the JPEG it replaced.
    """
    from app import main as m

    monkeypatch.setattr(m, "_last_ref", (-1, None))
    writes = []

    def _spy(img, seq, directory=None):
        writes.append(seq)
        return f"f{seq}_1x1.bgr"

    monkeypatch.setattr(m.frameref, "write_frame", _spy)
    img = object()
    assert m._cached_ref(img, 7) == "f7_1x1.bgr"
    for _ in range(50):                    # the poll storm
        assert m._cached_ref(img, 7) == "f7_1x1.bgr"
    assert writes == [7], "one write for one frame, however many times it is asked for"
    m._cached_ref(img, 8)
    assert writes == [7, 8], "and a new seq is a new frame"
