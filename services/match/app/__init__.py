"""heco match service — SQLite-backed identity gallery (stage 6 of the pipeline).

Decides, for each face embedding, whether the person has been seen before in
this run (cosine search against a per-run gallery) or is new (unique count +1).
See CONTRACTS.md at the repo root for the wire contract.
"""
