"""heco runner service — the pipeline conductor (port 7100).

Drives frame -> persons -> tracker -> faces -> quality gate -> embed -> match
over HTTP against the other six services, times every stage, aggregates the
measured metrics, and reports the whole run into the site-planner console.
See CONTRACTS.md at the repo root for the wire contract.
"""
