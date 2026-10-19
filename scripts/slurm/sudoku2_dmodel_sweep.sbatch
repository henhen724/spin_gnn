#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_dmodel
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out

# sudoku2_nhead_sweep.sbatch (Slurm 48296020) found nhead=4 (head_dim=16) is
# a peak -- more heads at fixed d_model=64 (smaller head_dim) monotonically
# hurt both solve_rate and attention-targeting enrichment. This tests the
# other direction: keep nhead=4 fixed and scale d_model up (so head_dim
# grows: 64->128->256 means head_dim 16->32->64), giving each head more room
# for id-matching without fragmenting into more heads. dim_feedforward
# stays fixed at 1024 (not scaled with d_model) to isolate d_model's effect
# on per-head capacity specifically. d_model=64 baseline already exists
# (d0064_h04_L12_ff01024_lr1e-03_pre, mean_o1_solve_rate=0.980).

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
cd "${SLURM_SUBMIT_DIR}"

D_MODEL_VALUES=(128 256)
D_MODEL=${D_MODEL_VALUES[${SLURM_ARRAY_TASK_ID}]}
NHEAD=4
NUM_LAYERS=12
DIM_FF=1024

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_hparam_sweep"
mkdir -p "$OUT_DIR"

echo "task ${SLURM_ARRAY_TASK_ID}: d_model=${D_MODEL} nhead=${NHEAD} num_layers=${NUM_LAYERS} dim_feedforward=${DIM_FF} epochs=300"

python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 24 \
    --degree 5 \
    --n-colors 4 \
    --d-model "$D_MODEL" \
    --nhead "$NHEAD" \
    --num-layers "$NUM_LAYERS" \
    --dim-feedforward "$DIM_FF" \
    --n-train-instances 5000 \
    --n-val-instances 300 \
    --batch-size 32 \
    --epochs 300 \
    --n-per-bin 100 \
    --out-dir "$OUT_DIR" \
    "$@"
