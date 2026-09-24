"""What this package is NOT allowed to know about.

The counting decisions have to be hostable by three different processes — the
runner today, a per-camera worker and a per-gate fusion process next. Each of
those brings its own transport, its own framework and its own idea of a
status counter. The moment this library reaches for one of them, all three
inherit the runner's whole world just to ask whether two faces are two people.

So the boundary is asserted rather than described.
"""

import subprocess
import sys

#: Importing the counting library must not drag any of these in. httpx and
#: fastapi are the runner's transport and framework; a counting library that
#: needs either has stopped being a library.
FORBIDDEN = ("httpx", "fastapi", "uvicorn", "starlette")


def test_importing_the_library_pulls_in_no_transport_or_framework():
    """Run in a FRESH interpreter, because this process has already imported
    the world — asserting against sys.modules here would pass trivially."""
    code = (
        "import sys, importlib;"
        "importlib.import_module('heco_counting');"
        "importlib.import_module('heco_counting.gate');"
        "importlib.import_module('heco_counting.zones');"
        "importlib.import_module('heco_counting.config');"
        "importlib.import_module('heco_counting.ports');"
        "importlib.import_module('heco_counting.association');"
        "importlib.import_module('heco_counting.appearance');"
        "importlib.import_module('heco_counting.face_search');"
        f"bad=[m for m in {FORBIDDEN!r} if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", (
        f"heco_counting imported {out.stdout.strip()} — the library has grown a "
        "transport, and every host now inherits it"
    )


def test_no_module_imports_the_runner():
    """The library must never reach back into the process hosting it.

    A single `from app...` here would make the package unusable by the worker
    and the fusion process it exists to serve, and the import would only fail
    at THEIR startup, not in this repo's tests.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "heco_counting"
    offenders = [
        f"{p.name}:{n}"
        for p in sorted(root.rglob("*.py"))
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if line.startswith(("from app", "import app"))
    ]
    assert not offenders, f"the library imports its host: {offenders}"
