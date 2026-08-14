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

RUN pip uninstall -y onnxruntime \
    && pip install --no-cache-dir \
        onnxruntime-gpu \
        nvidia-cuda-runtime-cu12 \
        nvidia-cublas-cu12 \
        nvidia-cudnn-cu12 \
        nvidia-cufft-cu12 \
        nvidia-curand-cu12

# ORT dlopens the CUDA userspace at session creation; the pip wheels land
# under site-packages/nvidia/*/lib and are not on the default search path.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/site-packages/nvidia/curand/lib
