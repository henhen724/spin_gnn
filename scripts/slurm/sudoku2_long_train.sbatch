#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_long
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out

# Hypothesis: the attention-targeting enrichment plateau (~1.25x, flat from
# epoch ~75 on per Slurm 48287024) and the agg_solve_rate=0.72 ceiling are
# training-budget-limited, not architecture-limited -- the best config's
# val_accuracy was still slowly rising at epoch 300 (0.56->0.57). Triples the
# budget to 900 epochs, changing ONLY epochs vs. the existing 300-epoch runs.
#
# Task 0 (d_model=64):  the current best config (mean_o1=0.980, agg=0.72 at
#   300 epochs, Slurm 48162371). Does more training raise agg_solve_rate /
#   enrichment, or is the ceiling real?
# Task 1 (d_model=128): tests the d_model-sweep's open question (Slurm
#   48446043) -- at 300 epochs it had agg_solve_rate UP (0.773) but
#   mean_o1_solve_rate/k0 DOWN (0.896/0.55) and val_nll worse (1.67). If that
#   was undertraining at larger width, 900 epochs should recover mean_o1
#   toward ~0.98 while keeping the better agg_solve_rate.
#
# Writes to a NEW dated out-dir: sudoku_hparam_sweep's checkpoint dir tag
# (d{d}_h{h}_L{L}_ff{ff}_lr{lr}_{norm}) does NOT include the epoch count, so
# reusing the old out-dir would overwrite the 300-epoch checkpoints.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
# Slurm stdout is block-buffered, so progress lines otherwise stay invisible
# for the whole run and cannot be polled.
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

D_MODEL_VALUES=(64 128)
D_MODEL=${D_MODEL_VALUES[${SLURM_ARRAY_TASK_ID}]}
NHEAD=4
NUM_LAYERS=12
DIM_FF=1024
EPOCHS=900

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_hparam_sweep_2026-10-19_ep${EPOCHS}"
mkdir -p "$OUT_DIR"

echo "task ${SLURM_ARRAY_TASK_ID}: d_model=${D_MODEL} nhead=${NHEAD} num_layers=${NUM_LAYERS} dim_feedforward=${DIM_FF} epochs=${EPOCHS}"

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
    --epochs "$EPOCHS" \
    --n-per-bin 100 \
    --out-dir "$OUT_DIR" \
    "$@"
