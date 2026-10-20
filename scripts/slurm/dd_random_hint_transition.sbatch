#!/bin/bash
#SBATCH -J spin_gnn_dd_rhint
#SBATCH -p sganguli
#SBATCH --array=0-4
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH -o %x-%A_%a.out

# Ground-truth DD recovery curve under INDEPENDENT RANDOM hint sampling, the
# reference curve the transformer work is aiming at. Distinct from
# sudoku_sweep.py / sudoku_transition_sweep.py, which build instances via
# removal_path (resolvable by construction at every density -> no solvability
# transition exists in that distribution at all).
#
# Motivation: a pilot at N=24 (the transformer's current size) showed
# mean_n_rounds peaking at only 1.7 and resolved_frac lifting off only above
# rho~0.7 -- i.e. at N=24 the DD cascade barely runs and there is no
# transition to learn. This maps how the curve sharpens with N, to pick the
# smallest N that shows a real transition (and therefore the sequence length
# the transformer must handle).
#
# One array task per size so the largest N doesn't serialize behind the rest.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
# Slurm stdout is block-buffered, so progress lines otherwise stay invisible
# for the whole run and cannot be polled.
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

# All satisfy planted_regular_graph's constraints at q=4, d=5:
# n_colors | n_nodes, and n_colors*(n_colors-1)=12 | degree*n_nodes.
SIZES=(24 48 96 192 384)
N=${SIZES[${SLURM_ARRAY_TASK_ID}]}

OUT_DIR="${SCRATCH}/spin_gnn/dd_random_hint_transition_2026-10-19"
mkdir -p "$OUT_DIR"

echo "task ${SLURM_ARRAY_TASK_ID}: N=${N}"

python -m scripts.dd_random_hint_transition \
    --sizes "$N" \
    --degree 5 \
    --n-colors 4 \
    --rho-min 0.0 \
    --rho-max 1.0 \
    --n-rho 41 \
    --n-instances 300 \
    --out-csv "${OUT_DIR}/N${N}.csv"
