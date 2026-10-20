#!/bin/bash
#SBATCH -J sudoku2_transition
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# Trains once (checkpointing a handful of epochs) then sweeps solve_rate
# over a fixed rho grid for each checkpoint -- traces how the easy (high
# rho) <-> hard (low rho) solve-accuracy curve shifts over training.
# See spin_gnn/sudoku_transition_sweep.py. Writes a CSV consumed by
# notebooks/sudoku_transition.ipynb.
# Usage: sbatch sudoku2_transition_sweep.sbatch [extra args]

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

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_transition"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_transition_sweep \
    --n-nodes 24 \
    --degree 5 \
    --n-colors 4 \
    --rho 0.3 \
    --rho-max 0.95 \
    --d-model 64 \
    --num-layers 2 \
    --nhead 1 \
    --n-train-instances 2000 \
    --n-val-instances 200 \
    --epochs 50 \
    --checkpoint-epochs 0,5,10,20,30,40,50 \
    --rho-grid 0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99,1.0 \
    --n-per-bin 50 \
    --out-csv "${OUT_DIR}/sweep_${SLURM_JOB_ID}.csv" \
    "$@"
