#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_mlp
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-15
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Follow-up to the checkpoint-vs-difficulty transition sweep (STATUS.md):
# solve_rate plateaus at ~0.28-0.34 even at rho=1.0 (the fully trivial,
# zero-reasoning-round case) despite outcome_accuracy/well_formed_rate
# already at 1.0 -- a model-capacity/precision ceiling on exact 24-node
# copying, not a difficulty-positioning issue. Current budget was
# d_model=64/2-layer/2000-instance/50-epoch.
#
# This scans num_layers ("MLP depth" -- more stacked transformer blocks,
# each with its own attention + feedforward sub-layer) x dim_feedforward
# (each block's own feedforward/MLP hidden width) at fixed d_model=64,
# nhead=4, mirroring sat_mlp_sweep.sbatch's grid pattern for the earlier
# k-SAT v2 work. Also bumps epochs 50->150 and n_train_instances
# 2000->5000 so deeper/wider configs get a fair shot at converging rather
# than being judged on an epoch budget sized for the smallest config.
#
# Target metric: mean_o1_solve_rate (spin_gnn.sudoku_hparam_sweep) --
# solve_rate averaged over k=0..4 uncolored nodes (a fixed small constant,
# not scaling with n_nodes=24), i.e. the near-trivial end of the problem.
# Goal: find the smallest config that reaches ~1.0 there.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
cd "${SLURM_SUBMIT_DIR}"

L_VALUES=(2 4 6 8)
FF_VALUES=(128 256 512 1024)
NUM_FF=${#FF_VALUES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
L_IDX=$(( IDX / NUM_FF ))
FF_IDX=$(( IDX % NUM_FF ))
NUM_LAYERS=${L_VALUES[$L_IDX]}
DIM_FF=${FF_VALUES[$FF_IDX]}
D_MODEL=64
NHEAD=4

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_hparam_sweep"
mkdir -p "$OUT_DIR"

echo "task ${IDX}: d_model=${D_MODEL} nhead=${NHEAD} num_layers=${NUM_LAYERS} dim_feedforward=${DIM_FF}"

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
    --epochs 150 \
    --n-per-bin 100 \
    --out-dir "$OUT_DIR" \
    "$@"
