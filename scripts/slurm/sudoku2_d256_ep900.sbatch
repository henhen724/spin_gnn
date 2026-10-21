#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_d256ep900
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH -o %x-%j.out

# Does d_model=256 keep improving past 300 epochs?
#
# First, a correction this run is built on: the project has been quoting
# agg_solve_rate = 0.767 (d0128 @900ep, 48513926_1) as the best result, but
# d0256_h04_L12_ff01024_lr1e-03_pre @300ep in sudoku_hparam_sweep/sweep.csv
# scores 0.8467 on the SAME eval (--n-per-bin 100, --eval-seed-start default
# 20000000000). 0.847 is the real incumbent. The 0.767 figure appears to be a
# stale ranking from when mean_o1_solve_rate was the headline metric (d0064
# wins that at 0.980); nothing was re-ranked when the metric changed.
#
# Changes exactly ONE thing vs that 0.847 run: epochs 300 -> 900.
#
# Hypothesis: it will NOT beat 0.847, and may come in below it. Two reasons.
# (1) The same epochs-300->900 change at d_model=128 moved agg_solve_rate
#     0.773 -> 0.767, i.e. nothing, while val_nll rose 1.67 -> 2.16.
# (2) The checkpoint series (48580229, analysed 2026-10-21) showed solve_rate
#     saturating by epoch 50 with val_nll rising monotonically after, so the
#     last ~850 epochs are pure overfitting at d_model=128. d_model=256 is
#     already at val_nll 2.41 by epoch 300 -- worse than d0128 reaches at 900
#     -- so it is overfitting harder and earlier, and more epochs should hurt.
#
# This is therefore a falsification run, not a hopeful one: it was explicitly
# requested, and it settles whether the epoch budget is the binding constraint
# at this width. The companion job sudoku2_d256_data4x.sbatch tests what I
# actually expect to be the binding constraint (training-set size).
#
# If agg_solve_rate DOES exceed 0.847 here, my overfitting reading is wrong
# and the d_model=128 series does not generalise to d_model=256.
#
# Runtime: the 300-epoch version fit inside 1h30 (sudoku2_dmodel_sweep.sbatch),
# so 3x the epochs needs ~4.5h; 8h leaves margin without being extravagant.

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

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_hparam_sweep_2026-10-21_d256_ep900"
mkdir -p "$OUT_DIR"

echo "d_model=256 nhead=4 num_layers=12 dim_feedforward=1024 epochs=900 n_train=5000"

python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 24 \
    --degree 5 \
    --n-colors 4 \
    --d-model 256 \
    --nhead 4 \
    --num-layers 12 \
    --dim-feedforward 1024 \
    --n-train-instances 5000 \
    --n-val-instances 300 \
    --batch-size 32 \
    --epochs 900 \
    --n-per-bin 100 \
    --out-dir "$OUT_DIR" \
    "$@"
