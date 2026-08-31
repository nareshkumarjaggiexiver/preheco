# iGPU-capable persons image — the Intel accelerator variant, IN the repo.
#
# The 2026-08-12 box-local experiment on thinkcenter002 proved the pattern
# (+40% end-to-end on the same stream — offload, not acceleration: YOLOX-nano
# barely runs faster on the UHD 620, but moving it there hands four CPU cores
# back to decode, YuNet, SFace and the tracker). That Dockerfile lived only
# on the box; this one is repo-owned so the arm is reproducible, exactly as
# persons-cuda.Dockerfile did for the CUDA arm.
#
# BUILD MANUALLY, never via `compose build` with the overlay — the overlay
# sets only `image:`, so a compose build would build the CPU dockerfile and
# tag it :igpu, silently clobbering this image (bitten live on the CUDA arm,
# 2026-08-26; /health device truth is what catches it):
#
#   docker compose -f docker-compose.yml build persons
#   docker build -f docker/persons-igpu.Dockerfile --build-arg BASE=heco-persons \
#     -t heco-persons:igpu .
#   HECO_DEVICE=GPU docker compose -f docker-compose.yml -f docker-compose.igpu.yml \
#     up -d --force-recreate persons
#
# Built FROM the already-built CPU service image, so the source inside is
# byte-identical to the CPU arm — a comparison between the two is a device
# comparison, never a code one.
ARG BASE=heco-persons
FROM ${BASE}

USER root

# The Intel compute runtime, from vendored release .debs — placed in
# docker/igpu-debs/ before building (they are ~146MB and never committed;
# fetch them from Intel's GitHub releases or copy them from a box that has
# them, e.g. thinkcenter002's ~/heco-igpu/debs). Two dead ends made it debs:
#
#   1. Debian trixie (this base) DROPPED intel-opencl-icd; bookworm has it,
#      trixie does not, so apt-get cannot supply the driver here.
#   2. Mounting the HOST's driver fails at kernel-compile time with
#      CL_OUT_OF_HOST_MEMORY — which looks like a GPU fault and is not one:
#      a host libigc built against a newer glibc (2.43) cannot load on this
#      base (2.41), so clBuildProgram has no compiler to build with.
#
# The versions are a PAIRING, not a choice: the UHD 620 is Gen9.5, which
# current NEO dropped — 24.35.30872.36 is the legacy line and it pins IGC
# 1.0.17537.24; a mismatched compiler reproduces dead end 2. SHA256SUMS is
# the repo's pin on all four files, checked fail-closed before anything
# installs.
COPY docker/igpu-debs/ /tmp/debs/
RUN cd /tmp/debs && sha256sum -c SHA256SUMS \
    && apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 clinfo \
    && dpkg -i /tmp/debs/*.deb \
    && rm -rf /tmp/debs /var/lib/apt/lists/*

# onnxruntime's OpenVINO execution provider replaces the CPU wheel (both
# register the same module name, so the uninstall must come first — the CUDA
# arm's rule). Only persons gets this: faces (YuNet) and embed (SFace) run
# through OpenCV, whose current build blocks their OpenCL path, so an
# OpenVINO EP in their images would be weight they never load. No source is
# patched — heco_common/ort.py already speaks OpenVINO natively: any
# HECO_DEVICE that is not CPU/CUDA becomes the EP's device_type, and /health
# serves requested-vs-active truth.
RUN pip uninstall -y onnxruntime \
    && pip install --no-cache-dir onnxruntime-openvino
