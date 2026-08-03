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
WORKDIR /srv/services/${SERVICE}

# Requirements first (layer-cached): mostly satisfied by the base already, so
# this usually installs only heco-common (editable) and small dev extras
# (pytest/ruff ride along at POC — trim if image size ever matters).
COPY services/${SERVICE}/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/${SERVICE}/app ./app

ENV HECO_PORT=${PORT}
EXPOSE ${PORT}

# Shell form so ${HECO_PORT} expands at runtime.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${HECO_PORT}
