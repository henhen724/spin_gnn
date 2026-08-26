#!/bin/bash
#SBATCH -J spin_gnn_scale_xf
#SBATCH -p sganguli
#SBATCH --nice=1000000
#SBATCH --array=0-39
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH -o %x-%A_%a.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Transformer-baseline counterpart to scale_sweep.sbatch: same grid, same
# power-law(1) disorder, same held-out-test-pool methodology, but trains
# SpinTransformer (no access to the true coupling graph) instead of SpinGNN,
# with d_model tuned per context_length so total parameter count matches
# SpinGNN's at that context_length (see transformer_baseline.find_matching_d_model).
#
# Same sganguli / --nice=1000000 low-priority convention as scale_sweep.sbatch.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"

K_VALUES=(2 4 8 16 32)
NSEQ_VALUES=(1 2 4 8 16 32 64 128)
NUM_NSEQ=${#NSEQ_VALUES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
K_IDX=$(( IDX / NUM_NSEQ ))
N_IDX=$(( IDX % NUM_NSEQ ))
K=${K_VALUES[$K_IDX]}
NSEQ=${NSEQ_VALUES[$N_IDX]}

OUT_DIR="${SCRATCH}/spin_gnn/scale_sweep/transformer_power1"
mkdir -p "$OUT_DIR"

echo "task ${IDX}: context_length=${K} n_train_sequences=${NSEQ}"

python -m spin_gnn.transformer_scale_sweep \
    --n-spins 32 \
    --gamma-kind power_law \
    --gamma-param 1.0 \
    --context-length "$K" \
    --n-train-sequences "$NSEQ" \
    --n-test-realizations 8 \
    --trajectory-length 512 \
    --epochs 50 \
    --gnn-hidden-dim 32 \
    --gnn-num-layers 2 \
    --num-layers 2 \
    --out-dir "$OUT_DIR" \
    "$@"
