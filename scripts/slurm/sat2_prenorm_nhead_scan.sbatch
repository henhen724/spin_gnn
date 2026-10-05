#!/bin/bash
#SBATCH -J spin_gnn_sat2_nh_pn
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-5
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Head-count scan at the new best point found by sat2_prenorm_depth_scan.sbatch:
# d_model=128, num_layers=16, lr=3e-4, norm_first=True (solve_rate=12.3% at
# nhead=2, the only head count tried so far at this depth/LR). Holds
# everything else fixed and sweeps nhead over every divisor of d_model=128
# in a reasonable range.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "${SLURM_SUBMIT_DIR}"

H_VALUES=(1 2 4 8 16 32)
NHEAD=${H_VALUES[$SLURM_ARRAY_TASK_ID]}

OUT_DIR="${SCRATCH}/spin_gnn/sat2_hparam_sweep"
mkdir -p "$OUT_DIR"

echo "task ${SLURM_ARRAY_TASK_ID}: nhead=${NHEAD} (d_model=128, num_layers=16, lr=3e-4, norm_first=True fixed)"

python -m spin_gnn.sat2_hparam_sweep \
    --n-vars 16 \
    --alpha 1.25 \
    --d-model 128 \
    --nhead "$NHEAD" \
    --num-layers 16 \
    --norm-first \
    --lr 3e-4 \
    --n-train-instances 3000 \
    --n-val-instances 300 \
    --batch-size 64 \
    --epochs 150 \
    --out-dir "$OUT_DIR" \
    "$@"
