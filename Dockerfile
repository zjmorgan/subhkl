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
	&& uv build

FROM python:3.10-slim AS tool

RUN apt-get update
RUN apt-get install -y libgl1 libxcb1

# Copy the virtual environment from build stage
COPY --from=build /build/dist/subhkl-0.1.0-py3-none-any.whl subhkl-0.1.0-py3-none-any.whl
#ENV PATH="/opt/venv/bin:$PATH"

RUN python -m pip install subhkl-0.1.0-py3-none-any.whl
RUN rm subhkl-0.1.0-py3-none-any.whl

# Set working directory
WORKDIR /app
