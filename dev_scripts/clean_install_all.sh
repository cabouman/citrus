#!/bin/bash
# Removes the build artifacts and reinstalls citrus (editable, with the test and docs extras) into the active
# environment. To start from a fresh conda environment, first run
#   source create_conda_test_environment.sh

cd "$(dirname "$0")" || exit 1
source ./remove_build.sh

cd ..
pip uninstall -y citrus &> /dev/null
pip install -e ".[test,docs]"
cd dev_scripts || exit 1

echo ""
echo "*** citrus reinstalled. Optional extras: pip install -e \".[physics,gpu]\" (XrayPhysics, Triton) ***"
echo ""
