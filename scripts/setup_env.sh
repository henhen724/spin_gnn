#!/bin/bash
# One-time environment setup for spin_gnn.
#
# Sherlock policy: never pip install / run python on the login node. Run this
# from an interactive allocation first, e.g.:
#   sh_dev -c 4 -m 16G
#   bash scripts/setup_env.sh
#
# Venv lives on $SCRATCH, not $HOME or $GROUP_HOME: both were near quota
# (HOME 86%, GROUP_HOME 92%) when this was written. $SCRATCH purges files
# unmodified for 90 days, but the venv is fully reproducible -- just rerun
# this script if it's ever gone.
#
# Module versions were verified with `ml spider` at the time this script was
# written (py-pytorch/2.4.1_py312 has no separate module-level CUDA version
# listed by spider, so the torch_geometric extension wheels below are matched
# to the *loaded* torch build at install time rather than hardcoded -- if
# `ml spider py-pytorch` later shows a different default/newer version,
# update the `module load` line and this will still resolve correctly).
set -euo pipefail

module purge
module load devel math \
    python/3.12.1 \
    py-pytorch/2.4.1_py312 \
    py-numpy/1.26.3_py312 \
    py-scipy/1.12.0_py312
# jpeg headers/libs for building Pillow from source: pip has no manylinux
# wheel for the newer Pillow versions matplotlib now pulls in that's
# compatible with this system's glibc, so it falls back to compiling --
# which fails without this (verified via `ml spider libjpeg-turbo`, per
# cluster policy: never guess module names/versions).
module load system libjpeg-turbo/3.1.2

VENV_DIR="${SCRATCH}/spin_gnn_venv"
python3 -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Sherlock's `module load` puts each loaded module's site-packages on
# PYTHONPATH *ahead of* the venv's own -- normally desirable (it's how this
# venv reuses the module's numpy/torch/scipy builds instead of reinstalling
# them), but it silently shadows anything ALSO pip-installed here with an
# older version. Confirmed case: py-pytorch bundles a typing_extensions too
# old for jupyter_client's TypedDict usage, which breaks `jupyter` entirely
# until the venv's own (newer, pip-installed) copy is given priority.
export PYTHONPATH="${VENV_DIR}/lib/python3.12/site-packages:${PYTHONPATH:-}"

export PIP_CACHE_DIR="${SCRATCH}/pip_cache"
export XDG_CACHE_HOME="${SCRATCH}/xdg_cache"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"

pip install --upgrade pip

TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_TAG=$(python3 -c "import torch; print('cu' + torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu')")
echo "Detected loaded torch ${TORCH_VER} (${CUDA_TAG})"

pip install torch_geometric pytest matplotlib pandas jupyter nbconvert ipykernel jinja2

# Optional accelerated scatter/sparse kernels; torch_geometric works without
# them (falls back to native PyTorch ops), so don't fail the whole setup if
# no matching prebuilt wheel exists for this torch/CUDA combo.
pip install torch_scatter torch_sparse \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html" \
    || echo "torch_scatter/torch_sparse not installed (no matching wheel) -- torch_geometric will use its native fallback."

pip install -e "$(dirname "$0")/.."

echo ""
echo "Setup complete. In future sessions, activate with:"
echo "  module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 system libjpeg-turbo/3.1.2"
echo "  source ${VENV_DIR}/bin/activate"
echo "  export PYTHONPATH=\"${VENV_DIR}/lib/python3.12/site-packages:\${PYTHONPATH:-}\""
