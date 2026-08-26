"""Five-point face alignment — the piece cv2.FaceRecognizerSF kept private.

Doc 06 said it plainly: "Stage 5 is not a models.lock row" — SFace ships
with `alignCrop` bundled, so swapping the embedder means owning the
alignment ourselves, and "misalignment is one of the largest silent
accuracy losses in face recognition". This module is that ownership, pure
and testable:

* THE TEMPLATE is the canonical ArcFace 112x112 five-point destination —
  the coordinates every ArcFace-lineage model was trained against (right
  eye, left eye, nose tip, right mouth corner, left mouth corner: the
  YuNet/scrfd landmark order, which is the service contract's order).
* THE TRANSFORM is the similarity (rotation + uniform scale + translation)
  that best maps the detected landmarks onto the template, solved in closed
  form (Umeyama) rather than with an iterative estimator — deterministic,
  no RANSAC seed, no cv2 dependency in the maths.

The functions take and return plain numpy; the warp itself uses cv2 (the
one import worth taking — a hand-rolled bilinear warp would be slower and
buggier than the battle-tested one).
"""

from __future__ import annotations

import cv2
import numpy as np

#: The canonical ArcFace 112x112 destination landmarks, in the contract's
#: landmark order. These exact constants are what ArcFace-lineage training
#: pipelines warp to; changing them changes what an embedding means.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],   # right eye
        [73.5318, 51.5014],   # left eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # right mouth corner
        [70.7299, 92.2041],   # left mouth corner
    ],
    dtype=np.float32,
)

#: The template's frame.
TEMPLATE_SIZE = 112

#: Landmark-variance floor (px^2) below which alignment is refused. A real
#: gate-passing face (>= 56 px wide, inter-eye distance >= ~24 px) has
#: landmark variance in the HUNDREDS of px^2, so this only catches the
#: degenerate sentinels — an all-zeros "no landmarks" placeholder, five
#: copy-pasted points — which would otherwise warp an arbitrary frame window
#: (for all-zeros: the top-left corner) into a confident, matchable
#: "face" embedding with nothing in any log.
MIN_LANDMARK_VAR = 1.0


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve the 2x3 similarity matrix mapping src points onto dst (Umeyama).

    Closed form: centre both point sets, take the SVD of the covariance,
    fix any reflection (a mirror is not a rotation — a reflected face is a
    different face), scale by the variance ratio. Deterministic for a given
    input, which is what makes alignment testable at all.

    (Near-)coincident src points are a ValueError, not a fallback: a zero
    covariance has no meaningful rotation and the old scale=1.0 branch
    silently embedded whatever pixels sat under the landmark point.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    src_var = (src_c ** 2).sum() / len(src)
    if src_var < MIN_LANDMARK_VAR:
        raise ValueError(
            "degenerate landmarks: five points (near-)coincident — cannot align"
        )
    cov = dst_c.T @ src_c / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    diag = np.diag([1.0, d])
    rotation = u @ diag @ vt
    scale = float(np.trace(np.diag(s) @ diag) / src_var)
    matrix = np.empty((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = dst_mean - scale * rotation @ src_mean
    return matrix.astype(np.float32)


def align_face(img: np.ndarray, landmarks) -> np.ndarray:
    """Warp out the 112x112 aligned crop an ArcFace-lineage embedder consumes.

    `landmarks` are the contract's five [x, y] pairs in source pixels, in
    the contract order — the same five every face-detector family emits.
    """
    src = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    matrix = similarity_transform(src, ARCFACE_TEMPLATE)
    return cv2.warpAffine(
        img, matrix, (TEMPLATE_SIZE, TEMPLATE_SIZE), flags=cv2.INTER_LINEAR
    )
