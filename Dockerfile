# ================================================================
# STAGE 1: builder
#
# Purpose: compile all C/C++ extensions into Python wheels once.
# Neither training nor inference runs this stage in production —
# they just copy the pre-built wheels from it.
#
# Why ubuntu:22.04 (not nvidia/cuda):
#   Day 2 infra = local KVM/QEMU, no GPU passthrough configured.
#   CUDA base adds 3GB for zero benefit on this hardware.
#   GPU upgrade path: swap these two FROM lines to:
#   FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS builder
#   Nothing else in this file changes.
#
# Why match ubuntu:22.04 to your VM base image:
#   Your Terraform cloud-init uses Ubuntu 22.04.
#   Same OS = same glibc version = wheels compiled here will
#   run there without "wrong ELF class" or glibc errors.
# ================================================================
FROM ubuntu:22.04 AS builder

# Prevents apt from asking interactive questions during build
# e.g. "What timezone are you in?" — breaks automated builds
ENV DEBIAN_FRONTEND=noninteractive

# System packages needed to COMPILE Python extensions:
# build-essential : gcc, g++, make — required by XGBoost, scipy
# python3.10-dev  : Python C headers — required to build any .so file
# libgomp1        : OpenMP — XGBoost uses this for CPU parallelism
# curl + git      : needed by DVC for remote operations (Day 4)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
# ↑ rm -rf /var/lib/apt/lists/* : deletes apt cache after install
#   Without this, every image carries ~50MB of package index files

# Pin pip version — pip's resolver behavior changed between versions
# Unpinned pip = dependency resolution works differently on rebuild
RUN python3.10 -m pip install --upgrade pip==23.3.1

WORKDIR /build

# COPY requirements first — BEFORE any source code.
# Docker builds in layers. If requirements.txt hasn't changed,
# Docker reuses the cached wheel-build layer on next build.
# This makes rebuilds after code changes take seconds not minutes.
COPY requirements.txt .
COPY requirements-inference.txt .

# Compile all training + inference packages into wheel files
# --wheel-dir /wheels : write compiled .whl files here
# --no-cache-dir      : don't cache downloads (keeps image lean)
# Both requirements files share many packages (numpy, xgboost etc)
# Building both together deduplicates the compilation work
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt
RUN pip wheel --no-cache-dir --wheel-dir /wheels \
    -r requirements-inference.txt


# ================================================================
# STAGE 2: training
#
# Purpose: throughput-heavy batch job.
# - Reads gigabytes of rideshare logs from /app/data
# - Burns CPU cycles fitting XGBoost model
# - Writes model.pkl to /models
# - Then EXITS — this container is not long-running
#
# Data enters via volume mount at runtime:
#   docker run -v ~/data:/app/data -v ~/models:/models rideshare:training
#
# Why ubuntu:22.04 (not python:3.10-slim):
#   Training needs libgomp1 and git for DVC.
#   Slim images strip these out and they're painful to add back.
# ================================================================
FROM ubuntu:22.04 AS training

ENV DEBIAN_FRONTEND=noninteractive

# Runtime-only system deps (NO build-essential — compilation is done)
# libgomp1 : XGBoost needs this at RUNTIME even on CPU
# git      : DVC needs git to track data versions (Day 4)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels from builder stage
# This is the key multi-stage move — no recompilation here
COPY --from=builder /wheels /wheels
COPY requirements.txt .

# Install from local wheels — fast, no internet needed
# --no-index        : only use /wheels, don't hit PyPI
# --find-links      : look for wheels in /wheels directory
RUN pip install --no-cache-dir --no-index \
    --find-links=/wheels \
    -r requirements.txt \
    && rm -rf /wheels requirements.txt
# ↑ Clean up wheels after install — they served their purpose

WORKDIR /app

# Copy source code — ONLY src/, never data/ or models/
# Data enters at runtime via volume mount (see VOLUME below)
# This keeps the image size independent of your dataset size
COPY src/ ./src/

# Declare volume mount points as metadata
# Actual data flows in at: docker run -v host_path:container_path
# /app/data  : rideshare CSVs flow IN  (read by training)
# /models    : model.pkl flows OUT (written by training, read by inference)
VOLUME ["/app/data", "/models"]

# Security: never run as root inside a container
# If a vulnerability is exploited, attacker gets user 'mlops'
# not root — cannot escape the container or modify system files
RUN useradd -m -u 1000 mlops \
    && mkdir -p /app/data /models \
    && chown -R mlops:mlops /app /models
USER mlops

# ENTRYPOINT vs CMD — the most important Dockerfile concept for MLOps:
#
# ENTRYPOINT: the fixed verb. Always runs. Cannot be overridden.
#             Think of it as: "this container IS a training job"
#
# CMD:        the default arguments. Can be overridden at runtime.
#             Think of it as: "with these default settings"
#
# Together they enable:
#   docker run rideshare:training                    ← uses CMD defaults
#   docker run rideshare:training --data-path /v2   ← overrides CMD
#
# This same pattern lets Kubernetes pass different args to the
# same image for different training runs (Day 17+)
ENTRYPOINT ["python3.10", "-m", "src.train"]
CMD ["--data-path", "/app/data", "--model-output", "/models/model.pkl"]


# ================================================================
# STAGE 3: inference
#
# Purpose: latency-sensitive, always-on prediction server.
# - Starts once, loads model.pkl into memory
# - Accepts prediction requests 24/7
# - Must respond in milliseconds
# - Lives for months between redeployments
#
# Why python:3.10-slim (not ubuntu:22.04):
#   Slim = stripped Ubuntu with only Python runtime.
#   ~200MB smaller than ubuntu base.
#   Smaller image = faster Kubernetes pod startup (cold start)
#   Smaller image = smaller attack surface (fewer CVEs to patch)
#
# Why no CUDA here:
#   XGBoost CPU inference is fast enough for ridesharing demand.
#   A single prediction takes ~1ms on CPU.
#   GPU would add 3GB of image size for <0.1ms improvement.
# ================================================================
FROM python:3.10-slim-bullseye AS inference

# libgomp1: XGBoost needs OpenMP at runtime even for CPU inference
# slim images strip it — we add it back explicitly
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY requirements-inference.txt .

RUN pip install --no-cache-dir --no-index \
    --find-links=/wheels \
    -r requirements-inference.txt \
    && rm -rf /wheels requirements-inference.txt

WORKDIR /app
COPY src/ ./src/

# MODEL_PATH: the single contract between training and inference.
#
# Training writes:  /models/model.pkl  (on host disk)
# Inference reads:  /models/model.pkl  (via volume mount)
#
# This env var is the ONLY thing that changes between:
#   - Local dev:  MODEL_PATH=/models/model.pkl
#   - Production: MODEL_PATH=s3://bucket/models/latest/model.pkl (Day 17+)
#
# The image never needs to be rebuilt when the model is retrained.
# Just point MODEL_PATH at the new artifact and restart.
# That's what makes blue-green deployment (Day 17) trivial.
ENV MODEL_PATH="/models/model.pkl"

VOLUME ["/models"]

RUN useradd -m -u 1000 mlops \
    && mkdir -p /models \
    && chown -R mlops:mlops /app /models
USER mlops

ENTRYPOINT ["python3.10", "-m", "src.serve"]
CMD []

# Document the port — Day 16 wires FastAPI to this port
# Kubernetes uses this as a hint for service routing
EXPOSE 8000
