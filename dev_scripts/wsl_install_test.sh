#!/bin/bash
# End-to-end installation test of citrus from GitHub in a fresh conda environment (Linux / WSL), without sudo:
# the C++ toolchain, CMake and git that XrayPhysics needs come from conda-forge. Logs every step.
# Usage: bash wsl_install_test.sh [install-dir]   (default ~/Purdue/citrus-install-test)
set -o pipefail
ROOT="${1:-$HOME/Purdue/citrus-install-test}"
ENV_NAME="citrus-test"
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

source "$HOME/miniconda3/etc/profile.d/conda.sh" || { echo "conda not found"; exit 1; }
step "conda environment $ENV_NAME (python 3.11, git, cmake, make, C/C++ compilers from conda-forge)"
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1
conda create -n "$ENV_NAME" -y -c conda-forge python=3.11 git cmake make cxx-compiler c-compiler || exit 1
conda activate "$ENV_NAME" || exit 1
python --version; git --version; cmake --version | head -1; ${CXX:-g++} --version | head -1

step "clone https://github.com/cabouman/citrus"
mkdir -p "$ROOT" && cd "$ROOT" || exit 1
rm -rf citrus
git clone -q https://github.com/cabouman/citrus.git || exit 1
cd citrus && git log --oneline -1

step "pip install torch (PyPI Linux wheel with CUDA)"
pip install -q torch || exit 1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

step "pip install -e .[test,physics,gpu]   (XrayPhysics builds from source here)"
# XrayPhysics declares cmake_minimum_required(VERSION 3.0), which CMake 4 rejects without this policy setting.
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install -e ".[test,physics,gpu]" 2>&1 | tail -15
echo "pip exit: ${PIPESTATUS[0]}"
pip list 2>/dev/null | grep -i -E "^(torch|triton|mbirtorch|xrayphysics|citrus|numpy|scipy) "

step "dev_scripts/check_cuda.sh"
bash dev_scripts/check_cuda.sh
echo "check_cuda exit: $?"

step "pytest (CITRUS_SLOW_TESTS=1 builds the real cross-section library)"
CITRUS_SLOW_TESTS=1 python -m pytest -q -ra tests 2>&1 | tail -25
echo "pytest exit: ${PIPESTATUS[0]}"

step "demo/demo_scatter_solver.py"
python demo/demo_scatter_solver.py 2>&1 | tail -15
echo "demo_scatter_solver exit: ${PIPESTATUS[0]}"

step "demo/demo_citrus.py (quick setting: short scan, 36 views, 1 cm, 10 outer iterations)"
python demo/demo_citrus.py --views 36 --delta-voxel 1 --outer-iterations 10 --scatter-view-stride 1 --scatter-every 1 2>&1 | tail -20
echo "demo_citrus exit: ${PIPESTATUS[0]}"
ls demo/output 2>/dev/null
step "done"
