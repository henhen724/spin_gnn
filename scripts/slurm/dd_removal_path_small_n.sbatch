#!/bin/bash
#SBATCH -J spin_gnn_dd_rp_smalln
#SBATCH -p sganguli
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# The DD reference curve for the transition plot, at transformer-reachable N.
#
# Hypothesis: rho_min and the shape of T(rho) are already close to their
# large-N values at N=24..996, so the deduction-time divergence that defines
# the hint-density transition is present at the size the transformer trains
# on, and no scale-up to ~1700 tokens is needed to look for it.
#
# Supporting evidence from a 5-seed smoke run: N=24 gives rho_min=0.4583 with
# T rising ~7x toward the floor; N=48 gives 0.4542 and ~9x. Large-N reference
# (existing sudoku_dd_results): rho_min = 0.4756..0.4778 for N = 996..49992.
#
# CPU only -- removal_path is numpy/numba, no GPU involved.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_dd_removal_small_n_2026-10-19"
mkdir -p "$OUT_DIR"

python -m scripts.dd_removal_path_small_n \
    --sizes 24 48 96 192 384 996 \
    --degree 5 --n-colors 4 \
    --n-seeds 200 \
    --out-csv "${OUT_DIR}/removal_path_T_vs_rho.csv"
