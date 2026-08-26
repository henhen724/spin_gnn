#!/bin/bash
#SBATCH -J spin_gnn_sat_hsweep
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-11
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# SATTransformer hyperparameter sweep over (d_model, nhead) at a much larger
# instance count (10,000 train / 500 val) than the earlier smoke run, per
# request after the 5-epoch/40-instance smoke run's accuracy plateaued --
# does scaling model width/heads and data actually move accuracy, and does
# any config reach >=90%?
#
# -p sganguli (this lab's own GPU allocation, currently far less contended
# than the shared `gpu` partition -- see personal Sherlock notes) with no
# --nice: this is the group's own new work, not backgrounded behind anyone
# else's, per the account's usual sganguli-preference guidance.
#
# --max-trace-tokens 2000: trace length is heavy-tailed (self-attention is
# O(L^2)), so an uncapped 10,000-instance pool is near-certain to include
# outliers long enough to blow up batch memory. This does skew training away
# from the hardest (deepest-backtracking) instances -- report the drop
# fraction sat_hparam_sweep.py's train_sat_model prints alongside any result.
#
# Array index -> (d_model, nhead) via the lookup table below: 4 widths x 3
# head counts = 12 grid points (all combinations valid: 32/64/128/256 all
# divide evenly by 1/2/4).

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"

D_VALUES=(32 64 128 256)
H_VALUES=(1 2 4)
NUM_H=${#H_VALUES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
D_IDX=$(( IDX / NUM_H ))
H_IDX=$(( IDX % NUM_H ))
D_MODEL=${D_VALUES[$D_IDX]}
NHEAD=${H_VALUES[$H_IDX]}

OUT_DIR="${SCRATCH}/spin_gnn/sat_hparam_sweep"
mkdir -p "$OUT_DIR"

echo "task ${IDX}: d_model=${D_MODEL} nhead=${NHEAD}"

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
