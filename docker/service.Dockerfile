# Thin per-service image on the shared base.
#
# Parameterised so every service uses the same Dockerfile (docker-compose
# passes SERVICE and PORT as build args):
#   docker build -f docker/service.Dockerfile \
#     --build-arg SERVICE=match --build-arg PORT=7106 -t heco-match .
#
# Build context is the repo root.  The repo layout is preserved inside the
# image (/srv/common + /srv/services/<name>) so each service's editable
# `-e ../../common` requirement resolves exactly as it does on a dev machine.
#
# Model weights are NOT baked in: bind-mount services/<name>/models (populated
# by `make models`) to /srv/services/<name>/models — see docker-compose.yml.
ARG BASE_IMAGE=heco-pipeline-base:latest
FROM ${BASE_IMAGE}

ARG SERVICE
ARG PORT=8000

COPY common /srv/common
# The counting decisions (gate, zones, association, folds, the ledger). The
# runner's requirements.txt resolves `-e ../../counting` against this path
# exactly as it does on a dev machine. Copied into every service image for the
# same reason common is — one Dockerfile serves them all — but only the
# runner's requirements install it, so no other service can import it.
COPY counting /srv/counting
# The capability manifest the runner serves from /health. It lives at the repo
# root and the runner resolves it as parents[3] of its own main.py — /srv here.
# Without this COPY the file exists on every dev checkout and in NO container,
# so /health silently reports "pipeline": null and the planner registers a
# manifest-less pipeline. Copied into every service image (it is ~1 KB); only
# the runner reads it.
COPY pipeline.json /srv/pipeline.json
WORKDIR /srv/services/${SERVICE}

# Requirements first (layer-cached): mostly satisfied by the base already, so
# this usually installs only heco-common (editable) and small dev extras
# (pytest/ruff ride along at POC — trim if image size ever matters).
COPY services/${SERVICE}/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/${SERVICE}/app ./app

# THE TRAP THIS CLOSES: a green test suite and an image whose runner cannot
# import. The tests run against the dev checkout's editable installs, so a
# missing COPY above passes every test on the machine and fails at the first
# frame in the container — the same shape as the bind-mount placeholder that
# once left the face models dead behind a healthy healthcheck. Import at BUILD
# time, where it costs a failed build instead of a failed event.
RUN python -c "import importlib, sys; \
    mods = ['heco_common']; \
    mods += ['heco_counting', 'heco_counting.gate', 'heco_counting.appearance'] \
            if '${SERVICE}' == 'runner' else []; \
    [importlib.import_module(m) for m in mods]; \
    print('imports ok:', ', '.join(mods))"

ENV HECO_PORT=${PORT}
EXPOSE ${PORT}

# Shell form so ${HECO_PORT} expands at runtime.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${HECO_PORT}
