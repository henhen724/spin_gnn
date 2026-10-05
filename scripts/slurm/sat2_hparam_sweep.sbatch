#!/bin/bash
#SBATCH -J spin_gnn_sat2_hp
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-71
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# First hyperparameter+LR scan for the v2 event-based encoding
# (spin_gnn.ksat2/sat_transformer2), at a fixed easy problem size: n_vars=16,
# alpha=1.25 -> m_clauses=20 ("O(20) clauses" as requested), no hints.
# Sequences here are ~200-300 tokens (vs. thousands in the old v1 vector
# encoding), so this should be tractable at small model scale. Grid: nhead
# x num_layers x d_model x lr = 4 x 3 x 3 x 2 = 72 configs. Goal: find any
# config that reaches solve_rate ~1.0 (free-generation from hints+statement
# alone reaches the verified-correct answer), the actual target metric, not
# token-level accuracy.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "${SLURM_SUBMIT_DIR}"

H_VALUES=(1 2 4 8)
L_VALUES=(1 2 4)
D_VALUES=(32 64 128)
LR_VALUES=(1e-3 3e-4)

N_L=${#L_VALUES[@]}
N_D=${#D_VALUES[@]}
N_LR=${#LR_VALUES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
LR_IDX=$(( IDX % N_LR )); IDX=$(( IDX / N_LR ))
D_IDX=$(( IDX % N_D )); IDX=$(( IDX / N_D ))
L_IDX=$(( IDX % N_L )); IDX=$(( IDX / N_L ))
H_IDX=$(( IDX ))

NHEAD=${H_VALUES[$H_IDX]}
NUM_LAYERS=${L_VALUES[$L_IDX]}
D_MODEL=${D_VALUES[$D_IDX]}
LR=${LR_VALUES[$LR_IDX]}

if (( D_MODEL % NHEAD != 0 )); then
    echo "skipping invalid combo d_model=${D_MODEL} nhead=${NHEAD} (not divisible)"
    exit 0
fi

OUT_DIR="${SCRATCH}/spin_gnn/sat2_hparam_sweep"
mkdir -p "$OUT_DIR"

echo "task ${SLURM_ARRAY_TASK_ID}: d_model=${D_MODEL} nhead=${NHEAD} num_layers=${NUM_LAYERS} lr=${LR}"

python -m spin_gnn.sat2_hparam_sweep \
    --n-vars 16 \
    --alpha 1.25 \
    --d-model "$D_MODEL" \
    --nhead "$NHEAD" \
    --num-layers "$NUM_LAYERS" \
    --lr "$LR" \
    --n-train-instances 3000 \
    --n-val-instances 300 \
    --batch-size 64 \
    --epochs 60 \
    --out-dir "$OUT_DIR" \
    "$@"
