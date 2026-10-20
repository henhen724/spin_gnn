#!/bin/bash
#SBATCH -J sudoku2_train
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# Pretrains SATReasoningTransformer (reused from the k-SAT v2 work) on the
# event-based DD-Sudoku reasoning-trace encoding (spin_gnn.sudoku_tokens /
# sudoku_trace). See spin_gnn/train_sudoku_transformer.py.
# Usage: sbatch sudoku2_train_transformer.sbatch [extra args passed to train_sudoku_transformer.py]

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

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_checkpoints/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

python -m spin_gnn.train_sudoku_transformer \
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
    --batch-size 32 \
    --epochs 50 \
    --out-dir "$OUT_DIR" \
    "$@"
