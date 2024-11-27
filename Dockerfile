FROM python:3.10-slim as build 

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PDM_VERSION=2.11.2 \
    PDM_HOME=/usr/local
# uncomment to allow prereleases to be installed
#ENV PDM_PRERELEASE=1
ENV PATH="/root/.local/bin:$PATH"

RUN apt update \
    && apt install -y curl git make \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python -

WORKDIR /build

COPY pyproject.toml pdm.lock README.md /build/
RUN pdm install -G:all --no-lock --no-self

ADD . /build
RUN pdm sync --dev -G:all

RUN pdm build
RUN pdm run pytest tests

FROM python:3.10-slim as tool

COPY --from=build /build/dist/subhkl-*-py3-none-any.whl .
RUN python -m pip install "$(find . -maxdepth 1 -name *.whl)"

