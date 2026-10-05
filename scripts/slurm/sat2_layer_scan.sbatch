#!/bin/bash
#SBATCH -J spin_gnn_sat2_layers
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-13
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Follow-up to sat2_hparam_sweep.sbatch: that 72-config grid found nhead
# barely matters (1/2/4/8 all similar) but num_layers=1 is clearly too
# shallow (20/24 configs got solve_rate=0) and num_layers=4 did WORSE than
# num_layers=2 on average -- ambiguous whether that's a real depth penalty
# or just undertraining at the same 60-epoch budget deeper nets need more
# of. This fixes the grid's other best-found axes (d_model=128, nhead=2)
# and scans num_layers much further up {2,3,4,6,8,12,16}, doubling the
# epoch budget (150 vs. 60) so deeper configs get a fair shot at
# converging, not just a fixed epoch count inherited from the shallow
# regime. Also keeps both competitive learning rates (1e-3 and 3e-4) from
# the prior sweep rather than committing to just one.
# Goal: find any (num_layers) that reaches solve_rate >= 0.50.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "${SLURM_SUBMIT_DIR}"

L_VALUES=(2 3 4 6 8 12 16)
LR_VALUES=(3e-4 1e-3)
N_LR=${#LR_VALUES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
LR_IDX=$(( IDX % N_LR ))
L_IDX=$(( IDX / N_LR ))
NUM_LAYERS=${L_VALUES[$L_IDX]}
LR=${LR_VALUES[$LR_IDX]}

OUT_DIR="${SCRATCH}/spin_gnn/sat2_hparam_sweep"
mkdir -p "$OUT_DIR"

echo "task ${SLURM_ARRAY_TASK_ID}: num_layers=${NUM_LAYERS} lr=${LR} (d_model=128, nhead=2 fixed)"

python -m spin_gnn.sat2_hparam_sweep \
    --n-vars 16 \
    --alpha 1.25 \
    --d-model 128 \
    --nhead 2 \
    --num-layers "$NUM_LAYERS" \
    --lr "$LR" \
    --n-train-instances 3000 \
    --n-val-instances 300 \
    --batch-size 64 \
    --epochs 150 \
    --out-dir "$OUT_DIR" \
    "$@"
