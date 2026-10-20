#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_nhead
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-3
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out

# sudoku_attention_evolution.py (Slurm 48287024) found attention-targeting
# enrichment rises to ~1.35-1.38 by epoch 50-75 then plateaus/regresses to
# ~1.22-1.26 for the rest of training while val_accuracy keeps slowly
# climbing -- the attention head's content-addressing of "which neighbor's
# color is this" saturates early. Testing whether more heads (more
# independent id-matching subspaces) raises that ceiling: sweeps nhead in
# {2,8,16,32} at the fixed d_model=64/num_layers=12/dim_feedforward=1024
# config that already hits mean_o1_solve_rate=0.980 at nhead=4 (round 2,
# Slurm 48162371) -- same training budget, nhead is the only thing varied.

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

NHEAD_VALUES=(2 8 16 32)
NHEAD=${NHEAD_VALUES[${SLURM_ARRAY_TASK_ID}]}
D_MODEL=64
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
