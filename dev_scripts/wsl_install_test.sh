#!/bin/bash
# End-to-end installation test of citrus from GitHub in a fresh conda environment (Linux / WSL), without sudo and
# without compilers: every dependency installs from wheels; the NIST XCOM data are downloaded on first use. Logs every
# step.
# Usage: bash wsl_install_test.sh [install-dir]   (default ~/Purdue/citrus-install-test)
set -o pipefail
ROOT="${1:-$HOME/Purdue/citrus-install-test}"
ENV_NAME="citrus-test"
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

source "$HOME/miniconda3/etc/profile.d/conda.sh" || { echo "conda not found"; exit 1; }
step "conda environment $ENV_NAME (python 3.11, git from conda-forge)"
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1
conda create -n "$ENV_NAME" -y -c conda-forge python=3.11 git || exit 1
conda activate "$ENV_NAME" || exit 1
python --version; git --version

step "clone https://github.com/cabouman/citrus"
mkdir -p "$ROOT" && cd "$ROOT" || exit 1
rm -rf citrus
git clone -q https://github.com/cabouman/citrus.git || exit 1
cd citrus && git log --oneline -1

step "pip install torch (PyPI Linux wheel with CUDA)"
pip install -q torch || exit 1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

step "pip install -e .[test,gpu]"
pip install -e ".[test,gpu]" 2>&1 | tail -15
echo "pip exit: ${PIPESTATUS[0]}"
pip list 2>/dev/null | grep -i -E "^(torch|triton|mbirtorch|xraylib|citrus|numpy|scipy) "

step "NIST XCOM data (downloaded into the citrus cache on first use)"
python -c "from citrus.scatter_solver.photon_data import resolve_backend; print(resolve_backend().reason)"
echo "photon data exit: $?"

step "dev_scripts/check_cuda.sh"
bash dev_scripts/check_cuda.sh
echo "check_cuda exit: $?"

step "pytest (CITRUS_SLOW_TESTS=1 builds the real cross-section library)"
CITRUS_SLOW_TESTS=1 python -m pytest -q -ra tests 2>&1 | tail -25
echo "pytest exit: ${PIPESTATUS[0]}"

step "demo/demo_1_scatter_solver.py"
MPLBACKEND=Agg python demo/demo_1_scatter_solver.py 2>&1 | tail -15
echo "demo_1_scatter_solver exit: ${PIPESTATUS[0]}"

step "demo/demo_2_citrus.py (180 views over a short scan, 0.5 cm, 80 outer iterations)"
MPLBACKEND=Agg python demo/demo_2_citrus.py 2>&1 | tail -20
echo "demo_2_citrus exit: ${PIPESTATUS[0]}"

step "benchmarks/citrus_study.py --help"
python benchmarks/citrus_study.py --help > /dev/null
echo "citrus_study exit: $?"
ls demo/output 2>/dev/null
step "done"
