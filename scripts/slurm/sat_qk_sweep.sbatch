#!/bin/bash
#SBATCH -J spin_gnn_sat_qk
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-19
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Follow-up to sat_mlp_sweep.sbatch: that scan found num_layers=4,
# dim_feedforward=1024 the best MLP config at d_model=256/nhead=8 (94.3%
# accuracy, NLL ~5.0 -- the best of any sweep so far), but
# val_satisfying_completion_rate stayed exactly 0% across all 16 points,
# unlike the earlier nhead-only sweep where nhead=16/32 (worse accuracy)
# were the only configs to ever solve anything.
#
# This sweep fixes the winning MLP config (d_model=256, num_layers=4,
# dim_feedforward=1024) and scans TWO attention-shape axes at once: nhead
# (number of heads) and qk_dim (the *total*, pre-head-split Q/K projection
# size -- decoupled from nhead/d_model via spin_gnn.sat_transformer's
# LowRankSelfAttention; V and the output projection always stay at full
# d_model=256, only Q/K rank is being restricted/expanded here).
# nhead in {1,2,4,8,16}, qk_dim in {64,128,256,512} -- qk_dim=256 is
# "standard" full rank (=d_model); 64/128 are reduced rank; 512 is an
# over-complete Q/K space larger than d_model itself. 5 x 4 = 20 grid
# points, all valid (every qk_dim here is a power of 2 and divides evenly
# by every nhead here).
#
# Same partition/resource convention as the prior two follow-ups (sganguli,
# no --nice -- this lab's own GPU allocation). Writes into the SAME
# sat_hparam_sweep directory; the tag now also encodes qk_dim, so there's no
# collision with any earlier sweep generation (see sat_hparam_sweep.py).

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"

H_VALUES=(1 2 4 8 16)
QK_VALUES=(64 128 256 512)
NUM_QK=${#QK_VALUES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
H_IDX=$(( IDX / NUM_QK ))
QK_IDX=$(( IDX % NUM_QK ))
NHEAD=${H_VALUES[$H_IDX]}
QK_DIM=${QK_VALUES[$QK_IDX]}
D_MODEL=256
NUM_LAYERS=4
DIM_FF=1024

OUT_DIR="${SCRATCH}/spin_gnn/sat_hparam_sweep"
mkdir -p "$OUT_DIR"

echo "task ${IDX}: d_model=${D_MODEL} nhead=${NHEAD} num_layers=${NUM_LAYERS} dim_feedforward=${DIM_FF} qk_dim=${QK_DIM}"

python -m spin_gnn.sat_hparam_sweep \
    --n-vars 24 \
    --k 3 \
    --alpha 4.2 \
    --d-model "$D_MODEL" \
    --nhead "$NHEAD" \
    --num-layers "$NUM_LAYERS" \
    --dim-feedforward "$DIM_FF" \
    --qk-dim "$QK_DIM" \
    --n-train-instances 10000 \
    --n-val-instances 500 \
    --batch-size 8 \
    --epochs 15 \
    --max-trace-tokens 2000 \
    --out-dir "$OUT_DIR" \
    "$@"
