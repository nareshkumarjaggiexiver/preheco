"""heco_common — shared library for the HECO pipeline services.

Owns the wire contract as code (pydantic schemas for every inter-service
message in CONTRACTS.md), base64-JPEG imaging helpers, config-from-env, and
the planner ingest client. Installed editable into each service's venv.
"""

__version__ = "0.1.0"
