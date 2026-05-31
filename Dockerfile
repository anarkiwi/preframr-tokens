# syntax=docker/dockerfile:1
# CI gate as a Docker build: installs the package + dev tools (pyright[nodejs] bundles the node
# runtime so pyright needs no network fetch) and runs the full run_tests.sh gate (black, pylint,
# pyright, pytest + coverage>=85) during build, so `docker build` IS the test gate.
# PIP_OPTS optionally points pip at a PyPI cache (proxpi) for fast local rebakes (set via a
# gitignored .env -> build.sh); empty in CI -> upstream PyPI.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}
ARG PIP_OPTS=""
ENV PIP_OPTS=$PIP_OPTS

WORKDIR /tok
COPY . /tok

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install $PIP_OPTS --upgrade pip \
 && pip install $PIP_OPTS -e ".[dev]" "pyright[nodejs]"

RUN sh run_tests.sh
