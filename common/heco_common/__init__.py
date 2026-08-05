"""heco_common — shared library for the HECO pipeline services.

Owns the SHARED half of the wire contract as code (pydantic schemas for the
messages more than one service speaks — see heco_common.schemas for what that
does and does not cover), base64-JPEG imaging helpers, config-from-env, run
logging, and the planner ingest client. Installed editable into each service's
venv.
"""

__version__ = "0.1.0"
