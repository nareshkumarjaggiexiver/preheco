# syntax=docker/dockerfile:1
# Hardware-decode ingest image (L6): the plain ingest service image plus the
# ffmpeg binary that INGEST_DECODER=nvdec|vaapi|ffmpeg runs as a subprocess.
#
#   docker compose -f docker-compose.yml build ingest
#   docker build -f docker/ingest-hwdec.Dockerfile --build-arg BASE=heco-ingest \
#     -t heco-ingest:hwdec .
#   INGEST_DECODER=nvdec docker compose -f docker-compose.yml \
#     -f docker-compose.hwdec.yml up -d --force-recreate ingest
#
# BUILD MANUALLY, never via `compose build` with the overlay: the overlay sets
# only `image:`, so a compose build would build the plain Dockerfile and tag
# it :hwdec — an image with no ffmpeg, which /health `device` then reports as
# requested=nvdec active=cpu (the CUDA arm's lesson, 2026-08-26).
#
# Built FROM the already-built service image, so the Python inside is
# byte-identical to the plain arm: a comparison between the two is a decoder
# comparison, never a code one.
#
# Nothing CUDA is installed here. Debian's ffmpeg reaches NVDEC through the
# ffnvcodec loader, which dlopens libcuda/libnvcuvid at run time; the driver's
# libraries arrive from the host (the overlay's gpus + /usr/lib/wsl mount).
# VA-API comes from libva + Mesa's drivers (mesa-va-drivers carries the d3d12
# driver WSL needs); vainfo is here to diagnose it.
ARG BASE=heco-ingest
FROM ${BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg mesa-va-drivers vainfo \
    && rm -rf /var/lib/apt/lists/*

# FAIL THE BUILD, not the run, if this ffmpeg cannot do the one thing the
# image exists for: an ffmpeg without the cuda/vaapi hwaccels would decode on
# the CPU and every "nvdec" number would be a CPU number.
RUN ffmpeg -hide_banner -hwaccels | tee /tmp/hwaccels \
    && grep -qx cuda /tmp/hwaccels && grep -qx vaapi /tmp/hwaccels \
    && ffmpeg -hide_banner -filters | grep -q hwdownload \
    && rm /tmp/hwaccels
