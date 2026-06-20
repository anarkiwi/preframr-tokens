# syntax=docker/dockerfile:1
# Test image: installs the package + dev tools (pyright[nodejs] bundles the node runtime so pyright
# needs no network fetch). The full run_tests.sh gate (black, pylint, pyright, pytest + coverage>=85)
# runs via `docker run` -- NOT during build -- so the register-dump fixtures (rendered beforehand in
# the headlessvice container, which cannot run inside a `docker build`) can be mounted in at
# /tok/tests/test_fixtures. PIP_OPTS optionally points pip at a PyPI cache (proxpi) for fast local
# rebakes (set via a gitignored .env -> build.sh); empty in CI -> upstream PyPI.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}
ARG PIP_OPTS=""
ENV PIP_OPTS=$PIP_OPTS

WORKDIR /tok
COPY . /tok

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install $PIP_OPTS --upgrade pip \
 && pip install $PIP_OPTS -e ".[dev]" "pyright[nodejs]"

CMD ["sh", "run_tests.sh"]
