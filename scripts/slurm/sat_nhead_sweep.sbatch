#!/bin/bash
#SBATCH -J spin_gnn_sat_nhead
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-2
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Follow-up to sat_hparam_sweep.sbatch's (d_model, nhead) grid: that sweep
# only tried nhead in {1,2,4} at each width, and only d_model=256 showed any
# real lift over the majority-class baseline (92.6-93.4% vs 89.05%) -- but
# even there, val_satisfying_completion_rate stayed at 0%. Hypothesis to
# test: single/few-head attention was the bottleneck, not model width.
# This scan fixes d_model=256 (the previously-best width) and extends the
# head-count axis to 8/16/32 (all valid divisors of 256), isolating the
# head-count effect. Writes into the SAME sat_hparam_sweep directory as the
# original grid (distinct tag d0256_h08/h16/h32, no collision), so
# aggregate_sat_sweep.py's existing output already gives the full
# nhead in {1,2,4,8,16,32} @ d_model=256 comparison once this finishes.
#
# Same partition/resource convention as sat_hparam_sweep.sbatch (sganguli,
# no --nice -- this lab's own GPU allocation, not backgrounded).

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"

H_VALUES=(8 16 32)
NHEAD=${H_VALUES[$SLURM_ARRAY_TASK_ID]}
D_MODEL=256

OUT_DIR="${SCRATCH}/spin_gnn/sat_hparam_sweep"
mkdir -p "$OUT_DIR"

echo "task ${SLURM_ARRAY_TASK_ID}: d_model=${D_MODEL} nhead=${NHEAD}"

python -m spin_gnn.sat_hparam_sweep \
    --n-vars 24 \
    --k 3 \
    --alpha 4.2 \
    --d-model "$D_MODEL" \
    --nhead "$NHEAD" \
    --num-layers 2 \
    --n-train-instances 10000 \
    --n-val-instances 500 \
    --batch-size 8 \
    --epochs 15 \
    --max-trace-tokens 2000 \
    --out-dir "$OUT_DIR" \
    "$@"
