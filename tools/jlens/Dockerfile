# SPDX-License-Identifier: MIT
# Multi-target CUDA image for tools/jlens:
#   docker build --target inkling2 -f tools/jlens/Dockerfile -t <registry>/nagaya-jlens:inkling2 .
#   docker build --target qwen38   -f tools/jlens/Dockerfile -t <registry>/nagaya-jlens:qwen38 .
#
# Build from the repository root so tools/jlens is copied cleanly.

ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY tools/jlens/requirements.txt /workspace/tools/jlens/requirements.txt
RUN pip install -r /workspace/tools/jlens/requirements.txt

COPY tools/jlens/*.py /workspace/tools/jlens/
# Make `python3 -m tools.jlens.*` importable without copying the rest of the repo.
RUN touch /workspace/tools/__init__.py

# Target 1: Qwen3.8-Flash-Next (transformers 5.17.0, matches tools/qwen38_reference/requirements.txt)
FROM base AS qwen38
ARG QWEN38_TRANSFORMERS=transformers==5.17.0
RUN pip install "${QWEN38_TRANSFORMERS}"
ENTRYPOINT ["python3", "-m", "tools.jlens.fit"]

# Target 2: Inkling-2-Mini (transformers commit 046bdb8cc2, matches tools/inkling2_golden.sh)
FROM base AS inkling2
ARG INKLING2_TRANSFORMERS=git+https://github.com/huggingface/transformers.git@046bdb8cc2
RUN pip install "${INKLING2_TRANSFORMERS}"
ENTRYPOINT ["python3", "-m", "tools.jlens.fit"]
