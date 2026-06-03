#!/bin/sh
set -e
black --check preframr_tokens tests
pylint preframr_tokens tests
pyright preframr_tokens
# Run the suite under pytest-xdist. -n auto adapts to the host (24 cores locally,
# 2-4 in CI); --dist worksteal lets idle workers steal queued tests so a few long
# property tests don't pin the wall-clock. pytest-cov writes a per-worker
# .coverage.<id> and combines them, so --cov-fail-under still gates the union.
# Override the worker count with PYTEST_WORKERS (e.g. PYTEST_WORKERS=1 for serial).
pytest -n "${PYTEST_WORKERS:-auto}" --dist worksteal \
    --cov=preframr_tokens --cov-report=term-missing --cov-fail-under=85
