#!/bin/bash
#SBATCH --job-name=sud2n48base
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=%x_%j.out
#
# First training run at N=48. Everything in this project so far is N=24.
#
# WHY N MATTERS HERE. N=24 gives the deduction depth almost no room: over the
# training range rho in [0.3, 0.95] the mean DD depth is T=4.25 with max 8,
# and the finite-size analysis (notebooks/sudoku_finite_size_2026-10-24)
# shows T grows as N^a at every density with no critical point to sit at, so
# depth only arrives by raising N. Measured on 20 instances per size:
#
#     N=24   204 tokens   mean T 4.25   max T  8
#     N=48   402 tokens   mean T 6.10   max T 14
#     N=96   787 tokens   mean T 7.45   max T 15
#
# N=48 buys a 1.4x deeper mean chain and nearly 2x the max for 4x the compute
# per instance (2x the sequence, quadratic attention). This run establishes
# that the architecture transfers at all before anything is spent at N=96.
#
# ONE CHANGE. Identical to the current best recipe (d_model=256, nhead=4,
# num_layers=12, ff=1024, lr=1e-3, pre-norm, 300 epochs) except n_nodes
# 24 -> 48. n_train is 20000 rather than 80000 so the cost matches the 5.5 h
# that 80000 took at N=24: 1/4 the instances at 4x the cost each.
#
# PRE-REGISTERED PREDICTIONS.
#  (a) agg solve_rate at N=48 lands in [0.85, 0.98] -- below the 0.9847 that
#      this recipe reached at N=24 with the same n_train=20000, because
#      sequences double and mean depth rises 4.25 -> 6.10. Falsifier: below
#      0.70, meaning the recipe does not transfer and N=48 needs its own
#      data/epoch budget before any depth scan is worth running there.
#  (b) The k bins stay at ceiling: solve_rate_k2..k4 >= 0.98, since those are
#      the shallowest instances at any N. If they are not, the problem is
#      optimization, not depth.
#  (c) Depth eval at fixed rho shows a STEEPER decline in solve_rate with T
#      than N=24 did (N=24: 0.980 -> 0.931 over T=1..4 at rho=0.75), because
#      N=48 reaches T=14 where N=24 stops at 8.
#
# Note the shared pool (seed_start 5e10) is an N=24 instance set, so it does
# NOT apply here; N=48 needs its own pool and this run is scored only against
# its own eval. Cross-N solve_rate comparisons are not like-for-like and are
# not claimed.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_n48_baseline_2026-10-24"
mkdir -p "$OUT_DIR"

echo "N=48 d_model=256 nhead=4 num_layers=12 ff=1024 epochs=300 n_train=20000"
python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 \
    --degree 5 \
    --n-colors 4 \
    --d-model 256 \
    --nhead 4 \
    --num-layers 12 \
    --dim-feedforward 1024 \
    --n-train-instances 20000 \
    --n-val-instances 300 \
    --batch-size 32 \
    --epochs 300 \
    --n-per-bin 100 \
    --out-dir "$OUT_DIR"

echo
echo "=== solve_rate by DD depth at fixed density (the depth axis, confound-controlled) ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${OUT_DIR}"/d0256_h04_L12_ff01024_lr1e-03_pre \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT_DIR}/depth_eval.csv"
