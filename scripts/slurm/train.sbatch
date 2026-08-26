#!/bin/bash
#SBATCH -J spin_gnn_train
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -o %x-%j.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Self-supervised next-spin-prediction training.
# Usage: sbatch train.sbatch [extra args passed to train.py]

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"

OUT_DIR="${SCRATCH}/spin_gnn/checkpoints/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

python -m spin_gnn.train \
    --n-spins 200 \
    --context-length 4 \
    --beta 0.5 \
    --hidden-dim 32 \
    --num-layers 2 \
    --keep-prob 1.0 \
    --n-realizations 8 \
    --trajectory-length 512 \
    --epochs 50 \
    --out-dir "$OUT_DIR" \
    "$@"
