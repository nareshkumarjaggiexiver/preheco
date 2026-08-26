"""Convert contract face JSON into the 15-float row cv2.FaceRecognizerSF expects.

`FaceRecognizerSF.alignCrop` consumes a YuNet-format detection row:
[x, y, w, h, then five (x, y) landmarks in YuNet order — right eye, left eye,
nose tip, right mouth corner, left mouth corner — then score]. The faces
service emits landmarks in exactly that order, so this is a pure reshape.
"""

import numpy as np


def validate_landmarks(face: dict) -> list:
    """Enforce the contract's landmark shape — shared by BOTH embedder families.

    Five [x, y] pairs or ValueError. The sface path always enforced this
    (via face_to_row); the arcface path used np.reshape, which happily
    re-pairs ANY nesting totalling ten floats into wrong (x, y) points and
    embeds a garbage crop with 200 OK — a silent accuracy collapse that only
    shows up as bad match rates (review finding). One helper, one sentence,
    so malformed landmarks are the same ValueError -> 400 everywhere.
    """
    landmarks = face.get("landmarks")
    if (
        not isinstance(landmarks, (list, tuple))
        or len(landmarks) != 5
        or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in landmarks)
    ):
        raise ValueError("face.landmarks must be five [x, y] pairs (YuNet order)")
    return list(landmarks)


def face_to_row(face: dict) -> np.ndarray:
    """Build the (15,) float32 row for alignCrop from a contract face dict.

    Requires `box` {x, y, w, h} and `landmarks` as five [x, y] pairs; `conf`
    is optional (defaults to 1.0). Raises ValueError on malformed input.
    """
    box = face.get("box")
    if not isinstance(box, dict) or not all(k in box for k in ("x", "y", "w", "h")):
        raise ValueError("face.box must have x, y, w, h")
    landmarks = validate_landmarks(face)
    row = np.empty(15, dtype=np.float32)
    row[0:4] = [box["x"], box["y"], box["w"], box["h"]]
    for i, (lx, ly) in enumerate(landmarks):
        row[4 + 2 * i] = lx
        row[5 + 2 * i] = ly
    row[14] = float(face.get("conf", 1.0))
    return row
