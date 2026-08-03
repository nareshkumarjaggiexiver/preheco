"""One-shot generator for decode_golden.json (checked in; rerun only on purpose).

Implements YOLOX grid decoding with explicit per-anchor loops — deliberately
independent of the vectorised `app.postprocess.decode_predictions` — so the
golden numbers are derived twice by different code. Seeded, hence reproducible.
Run: `.venv/bin/python tests/fixtures/generate_decode_golden.py` from the
service root.
"""

import json
import math
from pathlib import Path

import numpy as np


def main() -> None:
    """Write decode_golden.json next to this script."""
    rng = np.random.default_rng(42)
    img_size = (32, 32)
    strides = [8, 16, 32]
    cells = []
    for s in strides:
        for gy in range(img_size[0] // s):
            for gx in range(img_size[1] // s):
                cells.append((gx, gy, s))
    preds = rng.normal(0.0, 1.0, size=(len(cells), 7)).round(4)  # 5 + 2 classes: tiny file
    expected = []
    for row, (gx, gy, s) in zip(preds, cells, strict=True):
        out = list(row)
        out[0] = (row[0] + gx) * s
        out[1] = (row[1] + gy) * s
        out[2] = math.exp(row[2]) * s
        out[3] = math.exp(row[3]) * s
        expected.append(out)
    payload = {"img_size": list(img_size), "preds": preds.tolist(), "expected": expected}
    Path(__file__).with_name("decode_golden.json").write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
