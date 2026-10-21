#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_d256data4x
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -o %x-%j.out

# Is the model data-limited rather than epoch-limited?
#
# Baseline: d0256_h04_L12_ff01024_lr1e-03_pre @300ep, 5000 train instances,
# agg_solve_rate = 0.8467 (sudoku_hparam_sweep/sweep.csv) -- the real
# incumbent best, see the header of sudoku2_d256_ep900.sbatch for why the
# project had been quoting 0.767 instead.
#
# Changes exactly ONE thing vs that run: --n-train-instances 5000 -> 20000.
# Epochs stay at 300, width stays at 256, lr/batch/architecture all unchanged.
#
# Hypothesis: this is the change that helps, and the epoch-budget change in
# the companion job is not. Reasoning: every diagnostic so far says the models
# overfit rather than underfit -- val_nll rises monotonically after epoch ~50
# while solve_rate is flat (48580229), d0256 reaches val_nll 2.41 by epoch 300
# (worse than d0128 at 900), and tripling epochs at d0128 bought 0.773 ->
# 0.767. A 9.5M-parameter transformer seeing only 5000 instances of a
# combinatorial task is the textbook setup for that, and there is no dropout
# or weight-decay flag in sudoku_hparam_sweep, so training-set size is the
# only regularization lever available. Instances are generated procedurally,
# so the extra 15000 cost nothing but CPU time to build.
#
# Prediction: val_nll at epoch 300 comes in well below 2.41, and
# agg_solve_rate exceeds 0.8467. If val_nll drops but agg_solve_rate does not
# move, then 0.847 is a genuine capacity//task ceiling rather than an
# overfitting artifact, which is also worth knowing and would redirect effort
# to architecture instead of data.
#
# Resources: 4x the data at the same epoch count is ~4x the gradient steps of
# a run that fit in 1h30, so ~6h; 10h leaves margin. Instance generation was
# timed at 6.1s/200 on a login node, so the 20000-graph DD solve costs ~10 min
# up front and is single-threaded -- no reason to ask for extra cores. Memory
# is 32GB rather than the usual 16 only because the instance list is held for
# the whole run (~100MB of arrays, so this is already generous).

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

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_hparam_sweep_2026-10-21_d256_data4x"
mkdir -p "$OUT_DIR"

echo "d_model=256 nhead=4 num_layers=12 dim_feedforward=1024 epochs=300 n_train=20000"

python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 24 \
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
    --out-dir "$OUT_DIR" \
    "$@"
