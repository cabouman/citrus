#!/bin/bash
# Purges the documentation build and rebuilds the html pages (needs: pip install ".[docs]").
# Usage: ./build_docs.sh   (from any directory)

cd "$(dirname "$0")/../docs" || exit 1
/bin/rm -rf build source/generated

python -m sphinx -b html source build/html
status=$?

echo ""
echo "*** The html documentation is at citrus/docs/build/html/index.html ***"
echo ""
exit $status
