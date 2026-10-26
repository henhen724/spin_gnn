#!/bin/bash
#SBATCH --job-name=sud2n96stk
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#
# DO THE TWO N=96 LEVERS STACK? Both moved the ceiling on their own, measured
# against the same control (49524030, 0.8267) on the same pool:
#   cell A  cosine lr at 20000   0.8740   +0.0473  p=3.5e-05
#   cell B  constant lr at 80000 0.9273   +0.1007  p=3e-20
# Neither has been run together. This cell is ONE CHANGE from cell B
# (--lr-schedule cosine added) and it completes a 2x2 over {20000, 80000} x
# {constant, cosine}, so the interaction term is computable exactly rather than
# inferred from two separate deltas.
#
# EVERYTHING ELSE IS CELL B: N=96, degree=5, n_colors=4, rho=0.3, rho_max=0.95,
# removal hints, d_model=256, nhead=4, L=12, ff=1024, pre-norm, lr=1e-3,
# batch=32, 150 epochs, n_train=80000, n_val=300, seed=0. --lr-min-factor stays
# at its 0.01 default, the value cell A used.
#
# TAG COLLISION, INTENTIONAL. This cell is tagged
# d0256_h04_L12_ff01024_lr1e-03_pre_cos -- the same string as cell A, which is a
# cosine cell at 20000 (n_train is not in the tag). They live in different out
# dirs and sudoku_agg_eval appends " [run_dir]" to same-tag checkpoints from
# different runs in both the CSV and the npz names, so all four cells of the 2x2
# stay addressable. Same mechanism the two N=48 L=12 cells rely on.
#
# WHY ALL FOUR CELLS GO IN ONE POOL. The pool is regenerated per problem size,
# so the four cells are only paired-testable against each other if they are
# scored in the same agg_eval call or at the same seed_start; they are at
# seed_start=50000000000 either way, but re-scoring all four here costs ~0.9 h
# per checkpoint and buys exact paired McNemar on every edge of the 2x2 plus
# the interaction, which is the question. Cell B goes first so vs_reference
# reads against the current best N=96 cell.
#
# PRE-REGISTERED:
#  (a) Cosine at 80000 beats cell B (0.9273) on the shared pool, paired, at
#      p<0.05. Falsifier: no resolvable gain or a loss, which would mean the
#      levers are not independent and cosine's +0.0473 was specific to the
#      underfitting regime at 20000 rather than a property of the schedule.
#  (b) The gain is SMALLER than the +0.0473 cosine bought at 20000 -- the levers
#      stack sub-additively. Reason: at 20000 the model never fit its training
#      set (train ended 0.0289 BELOW val) and a decaying lr had slack to take
#      up, whereas at 80000 it already reaches val 1.3943 with a normal +0.0239
#      gap, and 0.9273 leaves less room than 0.8267 did.
#      LOW CONFIDENCE, AND SAID SO IN ADVANCE: this is the same genre of
#      inference -- reading the train/val gap to predict a lever's size -- that
#      produced pre-registration (b) of 49565433, which was not merely wrong but
#      wrong by 2.1x in the opposite direction. My record on this style of
#      argument is 0 for 1. Falsifier: a gain >= +0.0473, meaning the levers
#      stack at least additively, the N=96 recipe has more headroom than the
#      ceiling language in STATUS implies, and the gap heuristic should be
#      dropped outright rather than used with a caveat.
#  (c) Every rho=0.60 bin still clears 0.90. Cell B clears all six but its
#      weakest is T=9 at 0.907 on n=43, so an aggregate gain is compatible with
#      a bin slipping under. Falsifier: a bin below 0.90, which would mean an
#      improvement in the aggregate can cost L*-measurability -- the per-bin
#      gate, not the aggregate, is what unblocked the layer scan (49639312).
#  (d) The train-val gap widens beyond cell B's +0.0239, because a lr decaying
#      to 1e-5 fits the training set tighter at the end. Falsifier: the gap
#      shrinks or goes negative while the solve_rate still rises, which would
#      sever the gap from the solve_rate at N=96 the way finding 4 already
#      severed val_nll from it across models.
#
# COST. Cell B's own training was ~8 h (49565433 ran 14:41:55 total: ~1 min
# smoke, ~2 h at 20000, ~8 h at 80000, ~4:40 of eval). Here: ~8 h training,
# ~1 h for a 1-checkpoint depth eval, ~3.6 h for a 4-checkpoint pool. ~13 h
# expected against a 24 h limit. 128G/4 cores is what the 80000-instance N=96
# build needed in cell B (sganguli caps memory at 32G per core).
#
# This is the third job of the 10-26 set and the only one that is not on the
# critical path of the L*(N) series: 49639312 (N=96 layer scan) and 49639313
# (N=48 data control) answer the standing plan question, this one improves the
# N=96 model itself. If the partition is contended, the other two come first.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

PLAT="${SCRATCH}/spin_gnn/sudoku_n96_plateau_2026-10-26"
CTRL="${SCRATCH}/spin_gnn/sudoku_n96_baseline_2026-10-25/d0256_h04_L12_ff01024_lr1e-03_pre"
OUT="${SCRATCH}/spin_gnn/sudoku_n96_stack_2026-10-26"
mkdir -p "$OUT"

echo "=== N=96 L=12, n_train=80000, COSINE lr (one change from cell B) ==="
python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 96 --degree 5 --n-colors 4 \
    --d-model 256 --nhead 4 --num-layers 12 --dim-feedforward 1024 \
    --lr-schedule cosine \
    --n-train-instances 80000 --n-val-instances 300 --batch-size 32 \
    --epochs 150 --n-per-bin 100 \
    --out-dir "$OUT"

echo
echo "=== does the stacked cell hold 0.90 in every rho=0.60 bin? (pre-reg c) ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${OUT}/d0256_h04_L12_ff01024_lr1e-03_pre_cos" \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT}/depth_n96_stacked.csv"

echo
echo "=== the full 2x2 on one pool: {20000,80000} x {constant,cosine} ==="
# Cell B first: it is the current best N=96 cell and the reference for (a).
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "${PLAT}/d0256_h04_L12_ff01024_lr1e-03_pre" \
                 "${OUT}/d0256_h04_L12_ff01024_lr1e-03_pre_cos" \
                 "${PLAT}/d0256_h04_L12_ff01024_lr1e-03_pre_cos" \
                 "$CTRL" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500.csv"
