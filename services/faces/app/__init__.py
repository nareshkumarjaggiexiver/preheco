"""faces — YuNet face-detection microservice (heco-pipeline, port 7104).

Implements the CONTRACTS.md `faces` service: POST /detect over base64-JPEG
frames (optionally scoped to person boxes via `within`) returning face boxes,
5-point landmarks and the POC quality flag (ok / sub-canon / reject). Weights
arrive via `make models` (repo-root models.lock); never committed.
"""

__version__ = "0.1.0"
