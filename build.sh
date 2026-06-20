#!/bin/bash
# Build the test image and run the run_tests.sh gate against it. The register-dump fixtures are
# rendered beforehand (headlessvice container) and MOUNTED into the container -- they cannot be
# produced inside a `docker build`, and are not baked into the image (.dockerignore). Optional
# gitignored local config sets PIP_OPTS to a PyPI cache (proxpi) for fast rebakes: a per-host
# .env.<hostname> is preferred (checked first; the NFS repo dir is shared across build hosts on
# different subnets), falling back to .env. Neither present -> PIP_OPTS empty -> upstream PyPI
# (slower, still works). See .env.example. Usage: ./build.sh [python-version]
set -e
ENV_FILE=".env.$(hostname -s)"
[ -f "$ENV_FILE" ] || ENV_FILE=".env"
[ -f "$ENV_FILE" ] && . "./$ENV_FILE"
PIP_OPTS="${PIP_OPTS:-}"
PYTHON_VERSION="${1:-3.12}"

python3 tests/_dump_fixture.py

DOCKER_BUILDKIT=1 docker build \
    --build-arg PIP_OPTS="$PIP_OPTS" \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    -t anarkiwi/preframr-tokens-test:"$PYTHON_VERSION" .

docker run --rm -v "$PWD/tests/test_fixtures:/tok/tests/test_fixtures" \
    anarkiwi/preframr-tokens-test:"$PYTHON_VERSION"
