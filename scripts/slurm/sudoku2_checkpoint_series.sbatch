#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_series
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# Saves a 16-snapshot checkpoint series to disk from ONE training run, so
# "when during training does DD appear" can be asked within a single run
# rather than across separately-seeded models. Records solve_rate at each
# snapshot on a fixed instance pool, so the emergence of decodable DD state
# (spin_gnn.sudoku_probes, spin_gnn.sudoku_sae) can be lined up against the
# capability curve: does DD structure appear before, with, or after
# solve_rate moves?
#
# Config matches the current best (d_model=128, nhead=4, L=12, ff=1024, 900
# epochs; Slurm 48513926_1, agg_solve_rate=0.767) so the series describes the
# model whose behavior all the other measurements characterize.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_checkpoint_series_2026-10-19"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_checkpoint_series \
    --n-nodes 24 --degree 5 --n-colors 4 \
    --d-model 128 --nhead 4 --num-layers 12 --dim-feedforward 1024 \
    --n-train-instances 5000 --n-val-instances 300 --batch-size 32 \
    --epochs 900 \
    --checkpoint-epochs 1,2,5,10,20,35,50,75,100,150,200,300,450,600,750,900 \
    --n-solve-instances 150 \
    --out-dir "$OUT_DIR"
