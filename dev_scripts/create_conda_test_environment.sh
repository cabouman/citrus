#!/bin/bash
# Creates (or recreates) the conda environment "citrus" with Python 3.11 and installs citrus with the test extras.
# Usage: source create_conda_test_environment.sh   (so that the environment stays active in your shell)
# Install PyTorch first if you need a specific CUDA build, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu126

NAME="citrus"
PYTHON_VERSION="3.11"

# conda activate needs the shell hook in a non-interactive shell
eval "$(conda shell.bash hook)"

while [ ${#CONDA_DEFAULT_ENV} -gt 0 ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; do
    conda deactivate
done

conda env remove --name "$NAME" -y &> /dev/null
conda create --name "$NAME" python="$PYTHON_VERSION" -y
conda activate "$NAME"

cd "$(dirname "${BASH_SOURCE[0]}")/.." || return 1
pip install -e ".[test]"
# the NIST XCOM photon data: downloaded once into the citrus cache (or found through $CITRUS_XCOM_ARCHIVE)
python -c "from citrus.scatter_solver.photon_data import resolve_backend; print(resolve_backend().reason)"
cd dev_scripts || return 1

echo
echo "Use 'conda activate $NAME' to activate the test environment."
echo
