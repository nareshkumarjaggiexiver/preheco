"""Convert contract face JSON into the 15-float row cv2.FaceRecognizerSF expects.

`FaceRecognizerSF.alignCrop` consumes a YuNet-format detection row:
[x, y, w, h, then five (x, y) landmarks in YuNet order — right eye, left eye,
nose tip, right mouth corner, left mouth corner — then score]. The faces
service emits landmarks in exactly that order, so this is a pure reshape.
"""

import numpy as np


def face_to_row(face: dict) -> np.ndarray:
    """Build the (15,) float32 row for alignCrop from a contract face dict.

    Requires `box` {x, y, w, h} and `landmarks` as five [x, y] pairs; `conf`
    is optional (defaults to 1.0). Raises ValueError on malformed input.
    """
    box = face.get("box")
    landmarks = face.get("landmarks")
    if not isinstance(box, dict) or not all(k in box for k in ("x", "y", "w", "h")):
        raise ValueError("face.box must have x, y, w, h")
    if (
        not isinstance(landmarks, (list, tuple))
        or len(landmarks) != 5
        or any(len(p) != 2 for p in landmarks)
    ):
        raise ValueError("face.landmarks must be five [x, y] pairs (YuNet order)")
    row = np.empty(15, dtype=np.float32)
    row[0:4] = [box["x"], box["y"], box["w"], box["h"]]
    for i, (lx, ly) in enumerate(landmarks):
        row[4 + 2 * i] = lx
        row[5 + 2 * i] = ly
    row[14] = float(face.get("conf", 1.0))
    return row
