#!/bin/bash
#SBATCH -J spin_gnn_sat2_prenorm
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-15
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Tests whether pre-norm (norm_first=True) fixes the depth-instability seen
# in sat2_layer_scan.sbatch: at lr=1e-3, num_layers in {12,16} completely
# collapsed under post-norm (0% well-formed output). Pre-norm keeps the
# residual stream itself un-normalized end to end (x = x + Sublayer(LN(x))
# instead of post-norm's x = LN(x + Sublayer(x))), the standard fix for
# exactly this failure mode in every modern deep decoder-only LM. Same grid
# as before (num_layers x lr, fixed d_model=128, nhead=2) so results are
# directly comparable to the post-norm numbers already on record, plus two
# even-deeper points (24, 32 layers) since pre-norm often keeps paying off
# well past where post-norm would have already failed.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "${SLURM_SUBMIT_DIR}"

L_VALUES=(2 3 4 6 8 12 16 24)
LR_VALUES=(3e-4 1e-3)
N_LR=${#LR_VALUES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
LR_IDX=$(( IDX % N_LR ))
L_IDX=$(( IDX / N_LR ))
NUM_LAYERS=${L_VALUES[$L_IDX]}
LR=${LR_VALUES[$LR_IDX]}

OUT_DIR="${SCRATCH}/spin_gnn/sat2_hparam_sweep"
mkdir -p "$OUT_DIR"

echo "task ${SLURM_ARRAY_TASK_ID}: norm_first=True num_layers=${NUM_LAYERS} lr=${LR} (d_model=128, nhead=2 fixed)"

python -m spin_gnn.sat2_hparam_sweep \
    --n-vars 16 \
    --alpha 1.25 \
    --d-model 128 \
    --nhead 2 \
    --num-layers "$NUM_LAYERS" \
    --norm-first \
    --lr "$LR" \
    --n-train-instances 3000 \
    --n-val-instances 300 \
    --batch-size 64 \
    --epochs 150 \
    --out-dir "$OUT_DIR" \
    "$@"
