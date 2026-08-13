#!/bin/sh
# Refuse to deploy weights that are missing, wrong, or a DIRECTORY.
#
# THE FAILURE THIS EXISTS TO STOP, which happened on a real box:
#
#   A deploy shipped the source tree with `tar --exclude='*.onnx'`. The
#   compose file bind-mounts each service's models directory, and Docker
#   materialises a missing bind-mount source as an EMPTY DIRECTORY rather
#   than failing. The stack came up, every healthcheck passed, /health served
#   the correct manifest, person detection worked — and the face models were
#   dead, so it counted zero faces for hours. The only evidence was a line
#   deep in a container log ("[Errno 21] Is a directory").
#
# Weights are deliberately never committed and never baked into the images
# (.dockerignore excludes **/models), so the bind mount is their ONLY source
# and nothing verified it at deploy time — even though models.lock has
# carried pinned sha256 for every artefact all along. This closes that gap:
# the pins are checked before anything starts, and a mismatch stops the
# deploy rather than producing a stack that looks healthy and cannot count.
#
#   ./scripts/verify-models.sh          # verify every row in models.lock
#
# Exit 0 = every pinned artefact is a real file with the right hash.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="$ROOT/models.lock"
[ -f "$LOCK" ] || { echo "verify-models: no models.lock at $LOCK" >&2; exit 1; }

fail=0
rows=0

# Rows are: <service> <filename> <sha256> <url>; comments and blanks skipped.
while read -r service file sha _url; do
    case "$service" in ''|'#'*) continue ;; esac
    [ -n "${file:-}" ] && [ -n "${sha:-}" ] || continue
    rows=$((rows + 1))
    path="$ROOT/services/$service/models/$file"

    if [ -d "$path" ]; then
        # The exact shape of the bug: Docker's placeholder for a bind mount
        # whose source was absent. Named explicitly because "not found" would
        # send someone looking for a download problem instead of a deploy one.
        echo "MISSING (a DIRECTORY, not a file — Docker made a placeholder): services/$service/models/$file" >&2
        fail=1
        continue
    fi
    if [ ! -f "$path" ]; then
        echo "MISSING: services/$service/models/$file — run 'make models-all'" >&2
        fail=1
        continue
    fi
    if [ ! -s "$path" ]; then
        echo "EMPTY: services/$service/models/$file (0 bytes)" >&2
        fail=1
        continue
    fi
    if ! echo "$sha  $path" | sha256sum --check --status -; then
        echo "SHA MISMATCH: services/$service/models/$file does not match its pin in models.lock" >&2
        fail=1
        continue
    fi
    echo "ok  $service/$file"
done < "$LOCK"

[ "$rows" -gt 0 ] || { echo "verify-models: models.lock named no artefacts — is it truncated?" >&2; exit 1; }

# The restricted tier (models-restricted.lock) is opt-in: an UNFETCHED weight
# is a choice, not a fault — but a fetched one that is wrong, empty or a
# Docker placeholder directory is exactly as fatal as a main-tier one, and a
# DIRECTORY for an unfetched one is still the bind-mount trap and fatal.
RLOCK="$ROOT/models-restricted.lock"
if [ -f "$RLOCK" ]; then
    while read -r service file sha _url; do
        case "$service" in ''|'#'*) continue ;; esac
        [ -n "${file:-}" ] && [ -n "${sha:-}" ] || continue
        path="$ROOT/services/$service/models/$file"
        if [ -d "$path" ]; then
            echo "MISSING (a DIRECTORY, not a file — Docker made a placeholder): services/$service/models/$file (restricted tier)" >&2
            fail=1
        elif [ ! -f "$path" ]; then
            echo "skip $service/$file (restricted tier, not fetched — 'make models-restricted' to opt in)"
        elif [ ! -s "$path" ]; then
            echo "EMPTY: services/$service/models/$file (restricted tier, 0 bytes)" >&2
            fail=1
        elif ! echo "$sha  $path" | sha256sum --check --status -; then
            echo "SHA MISMATCH: services/$service/models/$file vs models-restricted.lock" >&2
            fail=1
        else
            echo "ok  $service/$file (restricted tier)"
        fi
    done < "$RLOCK"
fi

if [ "$fail" -ne 0 ]; then
    echo "" >&2
    echo "Refusing to deploy: at least one pinned model artefact is missing or wrong." >&2
    echo "A container started against these would report Up (healthy) and count nothing." >&2
    exit 1
fi

echo "all $rows pinned model artefacts verified"
