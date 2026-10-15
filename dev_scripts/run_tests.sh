#!/bin/bash
# Runs the test suite with the active environment's Python.
# Usage: ./run_tests.sh [--slow] [pytest arguments...]
#   --slow   also run the slow tests (CITRUS_SLOW_TESTS=1: full-resolution library builds, needs XrayPhysics)
# Tests that need CUDA, Triton, MBIRTORCH or XrayPhysics skip themselves when those are unavailable.
# `python -m pytest` (not bare `pytest`) runs pytest from the active environment regardless of PATH order;
# -ra prints the short summary of skipped / failed tests.

cd "$(dirname "$0")/.." || exit 1
if [ "$1" == "--slow" ]; then
    export CITRUS_SLOW_TESTS=1
    shift
fi
python -m pytest -ra -q tests "$@"
