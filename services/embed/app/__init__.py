"""embed — SFace face-embedding microservice (heco-pipeline, port 7105).

Implements the CONTRACTS.md `embed` service: POST /embed over a base64-JPEG
frame plus detected faces (box + 5-point landmarks from the faces service),
returning one 128-float SFace embedding per face after landmark alignment.
Weights arrive via `make models` (repo-root models.lock); never committed.
"""

__version__ = "0.1.0"
