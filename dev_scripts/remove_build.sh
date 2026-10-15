#!/bin/bash
# Cleans the repository of build artifacts, Python caches and test outputs (keeps demo/input and demo/output).
# Usage: ./remove_build.sh   (or: source remove_build.sh)

NAME="citrus"

pushd "$(dirname "${BASH_SOURCE[0]}")/.." > /dev/null || exit 1
/bin/rm -rf docs/build dist build "$NAME.egg-info" .pytest_cache &> /dev/null
find . -name "__pycache__" -type d -prune -exec /bin/rm -rf {} + &> /dev/null
popd > /dev/null || exit 1
echo "Removed build artifacts and Python caches."
