# syntax=docker/dockerfile:1
# CUDA-capable persons image — the accelerator variant, IN the repo.
#
# The iGPU work proved the pattern (same source, device chosen by env, the
# ACTIVE provider logged and served from /health) but its Dockerfile stayed
# box-local and was lost with the box. This one is repo-owned so the CUDA
# arm is reproducible on any box with the nvidia container runtime:
#
#   docker build -f docker/persons-cuda.Dockerfile --build-arg BASE=heco-persons \
#     -t heco-persons:cuda .
#   HECO_DEVICE=CUDA PERSONS_MODEL=yolox_s.onnx \
#     docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d
#
# Built FROM the already-built CPU service image, so the source inside is
# byte-identical to the CPU arm — a comparison between the two is a device
# comparison, never a code one (the igpu overlay's own rule).
#
# CUDA libraries come from NVIDIA's pip wheels rather than a nvidia/cuda
# base image: the driver arrives via the container runtime (`gpus:` in the
# overlay), and the toolkit userspace (cublas/cudnn/nvrtc) via pip keeps
# this a thin layer on the SAME base as every other service instead of a
# parallel image tree. onnxruntime-gpu replaces the CPU wheel; both register
# the same module name, so the uninstall must come first.
ARG BASE=heco-persons
FROM ${BASE}

USER root

# CUDA 13 userspace, not cu12: onnxruntime-gpu 1.28 links
# libcublasLt.so.13 and cuDNN 9-for-13 — the cu12 wheels install cleanly
# and then the EP fails to dlopen at session creation, which the /health
# device truth catches as requested=CUDA active=[CPU] (bitten live on the
# .94 first build). NAMING, verified empirically on that box: the CUDA-13
# generation DROPPED the -cu13 suffix for the toolkit wheels (nvidia-cublas,
# nvidia-cuda-runtime, nvidia-cufft, nvidia-curand — they land under
# site-packages/nvidia/cu13/lib) while cuDNN keeps it (nvidia-cudnn-cu13,
# under nvidia/cudnn/lib).
# PINNED to the set the .94 box ran green on 2026-09-24 (onnxruntime-gpu 1.30.0
# listing TensorRT/CUDA/CPU providers, cu13 userspace). Unpinned, every code
# change re-resolved these, so a routine redeploy could pull a different
# runtime than the one last proven — the night before an event is when that
# surfaces. The pip cache mount keeps the ~2 GB of wheels on the build host:
# this layer re-runs on EVERY code change (it sits on top of the service
# image), and six downloads per deploy was the whole rebuild time.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip uninstall -y onnxruntime \
    && pip install \
        onnxruntime-gpu==1.30.0 \
        nvidia-cuda-runtime==13.4.92 \
        nvidia-cublas==13.8.0.4 \
        nvidia-cudnn-cu13==9.26.0.51 \
        nvidia-cufft==12.4.0.43 \
        nvidia-curand==10.4.4.72
# TensorRT, for HECO_DEVICE=TRT (the fp16 arm; heco_common/ort.py). The
# TensorRT EP inside onnxruntime-gpu 1.30.0 links libnvinfer.so.10 and
# libnvonnxparser.so.10 against libcudart.so.13 (ldd, 2026-09-24), so the
# match is TensorRT 10.x built for CUDA 13: tensorrt-cu13-libs 10.x. The 11.x
# wheels ship libnvinfer.so.11 and would leave the EP exactly as dead as no
# TensorRT at all. Its OWN layer, after the CUDA one, so this ~2 GB rides the
# same pip cache and a TensorRT bump never re-resolves the proven CUDA set.
# Without these libs the EP fails to dlopen and ORT drops the session to
# CUDA+CPU — /health shows requested=TRT with no TensorRT in `active`.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install tensorrt-cu13-libs==10.16.1.11
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/tensorrt_libs
