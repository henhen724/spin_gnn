#!/bin/bash
#SBATCH -J spin_gnn_sparsify
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -o %x-%j.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Random-sparsification sweep: trains one SpinGNN per edge keep-probability
# against the same fixed disorder realizations, writes a comparison CSV.
# Usage: sbatch sparsify_sweep.sbatch [extra args passed to sparsify.py]

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"

OUT_DIR="${SCRATCH}/spin_gnn/sparsify_sweep/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sparsify \
    --n-spins 200 \
    --context-length 4 \
    --beta 0.5 \
    --hidden-dim 32 \
    --num-layers 2 \
    --n-realizations 8 \
    --trajectory-length 512 \
    --epochs 50 \
    --keep-probs 1.0 0.5 0.25 0.1 0.05 0.02 0.01 \
    --out-dir "$OUT_DIR" \
    "$@"
