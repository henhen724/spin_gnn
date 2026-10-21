#!/bin/bash
#SBATCH --job-name=sud2d384x4
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#
# Combine the two wins of 2026-10-21, one change from the new best.
#
# 48854233 raised n_train 5000 -> 20000 at d_model=256 and took agg_solve_rate
# 0.847 -> 0.973, by far the largest single improvement in 34 runs. It also
# raised solve_rate_k0 0.24 -> 0.60 and mean_o1 0.842 -> 0.918, and dropped
# val_nll 2.413 -> 1.265. Training set size had been pinned at 5000 for all 31
# previous runs, so the binding constraint was never varied until today.
#
# 48862411 raised d_model 256 -> 384 at n_train=5000 and matched the incumbent
# on agg (0.847) while taking mean_o1 0.842 -> 0.994 and solve_rate_k0
# 0.24 -> 0.99. The agg-vs-o1 trade-off that 48854232 made concrete is not a
# trade for width: d384 fixed the rho=1.0 weakness without giving up agg.
#
# This job changes exactly ONE thing against 48854233, the current best:
# d_model 256 -> 384. Everything else -- epochs 300, nhead 4, 12 layers,
# ff 1024, lr 1e-3, batch 32, n_train 20000, n_val 300, --n-per-bin 100 --
# is 48854233's value, so the comparison is 0.973 against this number on the
# identical eval.
#
# PRE-REGISTERED PREDICTION. agg above 0.973 but by much less than the data
# jump was worth, because agg has little room left; the real target is the
# residual weakness. 48854233's remaining soft spot is k0 = 0.60, which is
# exactly what width fixed at 5000 instances (0.24 -> 0.99), so I predict
# k0 >= 0.95 and mean_o1 >= 0.97 here. If instead k0 stays near 0.60, the
# width effect on k0 was contingent on the small training set and the two
# interventions are not additive, which would be the more informative result
# and would argue for n_train 80000 at d256 as the next step rather than more
# width.
#
# Falsifiable either way, and worth stating plainly: a tie at 0.973 would mean
# agg has saturated at this eval's resolution. The standard error on a 0.973
# rate over 300 instances is about 0.009, so anything inside 0.973 +/- 0.02 is
# a tie and must be reported as one, not as an improvement.
#
# 4h: 48854233 (d256, 20000) took 1:24 and 48862411 (d384, 5000) took 0:30, so
# ~2h is the estimate at d384 x 20000. One GPU, 2 cores (instance construction
# is single-threaded and happens once up front), 32GB for the held instance
# list, same as 48854233.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_hparam_sweep_2026-10-21_d384_data4x"
mkdir -p "$OUT_DIR"

echo "d_model=384 nhead=4 num_layers=12 dim_feedforward=1024 epochs=300 n_train=20000"

python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 24 \
    --degree 5 \
    --n-colors 4 \
    --d-model 384 \
    --nhead 4 \
    --num-layers 12 \
    --dim-feedforward 1024 \
    --n-train-instances 20000 \
    --n-val-instances 300 \
    --batch-size 32 \
    --epochs 300 \
    --n-per-bin 100 \
    --out-dir "$OUT_DIR" \
    "$@"
