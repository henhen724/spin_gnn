#!/bin/bash
#SBATCH -J spin_gnn_sat_train
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Next-token-prediction training of SATTransformer over random-k-SAT
# unit-propagation solver traces.
# Usage: sbatch sat_train_transformer.sbatch [extra args passed to train_sat_transformer.py]

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"

OUT_DIR="${SCRATCH}/spin_gnn/sat_checkpoints/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

python -m spin_gnn.train_sat_transformer \
    --n-vars 24 \
    --k 3 \
    --alpha 4.2 \
    --d-model 64 \
    --num-layers 2 \
    --nhead 1 \
    --n-train-instances 256 \
    --n-val-instances 32 \
    --batch-size 8 \
    --epochs 50 \
    --out-dir "$OUT_DIR" \
    "$@"
