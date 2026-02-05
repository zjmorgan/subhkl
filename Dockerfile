FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS build 

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt update \
    && apt install -y \
       curl git make build-essential cmake pkg-config \
       libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
       libjpeg-dev zlib1g-dev libffi-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Use repo root as build context so versioningit can access git metadata
WORKDIR /build

# Copy project files (include .git so versioningit can compute versions)
COPY . /build/

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade packaging tools and install the package
RUN uv pip install -U pip setuptools wheel toml \
    && uv pip install -e .

FROM python:3.10-slim AS tool

# Copy the virtual environment from build stage
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app
