#!/bin/bash
#SBATCH -J sudoku2_smoke
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
cd "${SLURM_SUBMIT_DIR}"

python -m spin_gnn.train_sudoku_transformer \
    --n-nodes 24 --degree 5 --n-colors 4 --rho 0.3 --rho-max 0.95 \
    --d-model 64 --num-layers 2 --nhead 1 \
    --n-train-instances 500 --n-val-instances 100 --batch-size 32 --epochs 10 \
    --out-dir "${SCRATCH}/spin_gnn/_sudoku2_smoke"
