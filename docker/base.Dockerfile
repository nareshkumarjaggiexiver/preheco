# heco-pipeline shared base image
#
# One heavy layer for all seven services: OpenCV (headless) + onnxruntime +
# the FastAPI stack, on python:3.12-slim, CPU-only (POC hard rule).  Each
# service then builds a thin layer on top (see docker/service.Dockerfile), so
# the native wheels are downloaded and stored once.
#
# Build (from the repo root):
#   docker build -f docker/base.Dockerfile -t heco-pipeline-base:latest .
#
# Licences: opencv-python-headless (Apache-2.0), onnxruntime (MIT),
# fastapi/uvicorn/httpx (MIT/BSD), numpy (BSD) — all ship-clean.
FROM python:3.12-slim

# Build-time only: debconf must not attempt an interactive frontend.
ARG DEBIAN_FRONTEND=noninteractive
# Root pip inside a single-purpose image is the intended layout, and the pip
# version is irrelevant to the pinned installs — silence both warnings.
ENV PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libglib is the one system library the headless OpenCV wheel still links.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the SAME versions the per-service requirements pin, so a service
# pip install on top of this base is a no-op for these packages.
RUN pip install --no-cache-dir \
        numpy==2.5.1 \
        opencv-python-headless==5.0.0.93 \
        onnxruntime==1.28.0 \
        pydantic==2.13.4 \
        fastapi==0.141.1 \
        uvicorn==0.52.1 \
        httpx==0.28.1

# Fail the build, not the deploy, if the native stack does not import.
RUN python -c "import cv2, onnxruntime, numpy; print(cv2.__version__, onnxruntime.__version__)"

WORKDIR /srv
