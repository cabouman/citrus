#!/bin/bash
# Removes the repository's build artifacts AND the user caches that citrus and its tools create: the cross-section
# library cache (~/.cache/citrus or $CITRUS_CACHE_DIR), the Triton kernel cache (~/.triton) and pip's cache.
# The next run rebuilds the library (needs XrayPhysics) and recompiles the kernels.
# Usage: ./deep_clean.sh

cd "$(dirname "$0")" || exit 1
source ./remove_build.sh

CACHE_DIR="${CITRUS_CACHE_DIR:-$HOME/.cache/citrus}"
echo "Removing $CACHE_DIR, $HOME/.triton/cache and the pip cache"
rm -rf "$CACHE_DIR" "$HOME/.triton/cache"
pip cache purge &> /dev/null
