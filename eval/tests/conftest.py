"""Put the repo root on sys.path so tests import the harness as ``eval.*``.

The harness is run from the repo root (``python -m eval.run``) but its tests
run from ``eval/`` with its own venv, like every other directory here. This is
the one line that reconciles the two.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
