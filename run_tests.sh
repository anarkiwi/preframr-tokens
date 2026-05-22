#!/bin/sh
set -e
black --check preframr_tokens tests
pylint preframr_tokens tests
pyright preframr_tokens
pytest --cov=preframr_tokens --cov-report=term-missing --cov-fail-under=80
