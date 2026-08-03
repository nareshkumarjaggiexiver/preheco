"""persons — YOLOX-nano person-detection microservice (heco-pipeline, port 7102).

Implements the CONTRACTS.md `persons` service: POST /detect over base64-JPEG
frames returning person boxes, plus GET /health. Model weights are downloaded
by `make models` (see repo-root models.lock); nothing model-shaped is committed.
"""

__version__ = "0.1.0"
