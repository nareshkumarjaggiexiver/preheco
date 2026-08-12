"""The counting decisions, separable from whatever process hosts them.

See README.md for why this is not part of ``heco_common`` (short version: the
match service installs heco_common, and the count must have exactly one
writer) and for the per-camera / per-gate / per-venue scope split that a
future worker and fusion process divide along.

Nothing here talks to a network, a camera or a framework.
"""
