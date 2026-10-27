#!/bin/bash
#SBATCH --job-name=sud2n24LFdata
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=%x_%j.out
#
# IS THE LEAK-FREE N=24 TASK DATA-BOUND AT 80000?
#
# HYPOTHESIS. Yes, and this is the prerequisite for everything else.
# 49659502 put the clean N=24 baseline at 0.8540 [0.8352, 0.8710] -- BELOW the
# 0.90 gate that defines L*. It also, against my own pre-registration, measured
# a memorization gap of +0.1000 at n_train=80000 (0.9567 on a recoverable
# prefix of its own training set vs 0.8567 held out, p=2.6e-09): the model fits
# what it has seen and the shortfall is coverage, not capacity or optimization.
#
# WHY THIS RUN AND NOT THE L*(T) SERIES. The L*(T) series at N=24 is the run
# STATUS calls highest-value, and it is -- but L* is threshold-defined, so a
# series run at a budget where even L=12 scores 0.854 would return L*(T)
# UNDEFINED in most bins and cost five cells to say nothing. That is exactly
# the error of finding 15 at N=96 (L*(96) undefined because the control was
# data-starved, diagnosed as an optimization ceiling, and wrong by 2.1x). One
# cell that establishes a budget which clears the gate makes the five-cell
# series well-posed; if it does NOT clear it, the series was saved.
#
# DESIGN. Two cells, both one change from the 49659502 config (N=24, degree=5,
# q=4, rho=0.3, rho_max=0.95, removal hints, d_model=256, nhead=4, L=12,
# ff=1024, pre-norm, lr=1e-3 constant, batch=32, seed=0, --shuffle-adj):
#
#   A  n_train=240000, 300 epochs.  3x the data at 3x the compute. Answers
#      "can leak-free N=24 clear 0.90 at all", which is the question that
#      gates the L*(T) series.
#   B  n_train=240000, 100 epochs.  3x the data at the SAME number of gradient
#      steps as the 0.8540 run (240000x100 = 80000x300 = 24M samples).
#      Separates data from compute: if B also beats 0.8540 then distinct
#      instances are worth more than repeats of the same ones, which is the
#      claim, and A's gain is not merely 3x more training.
#
# All three cells (these two plus 49659502's) are `_shuf` at the same N,
# degree, q, rho and rho_max, so they share the n=1500 pool and are
# paired-testable. n_train is not an INSTANCE_KEY and not in the checkpoint
# tag, so the three live in separate run dirs and agg_eval disambiguates them
# by the " [run_dir]" suffix.
#
# PRE-REGISTERED:
#  (a) Cell A clears 0.90 on the shared pool, i.e. resolvably above 0.8540.
#      Reason: the +0.1000 own-training gap says the function is learnable
#      from what it has seen and the held-out shortfall is coverage.
#      Moderate confidence. Falsifier: A ties 0.8540, which would mean 0.854
#      is a capacity or optimization ceiling at L=12/ff=1024 and the gate has
#      to be moved rather than reached.
#  (b) Cell B also beats 0.8540, by less than A. Reason: coverage is the
#      binding constraint, so new instances buy something even at fixed
#      compute, but 100 epochs undertrains relative to 300. Falsifier worth
#      watching: B >= A, which would say the extra epochs do nothing and the
#      record's 300 were already wasted.
#  (c) The memorization gap SHRINKS at 240000, below +0.1000. This is the
#      mechanism check on (a): if A clears the gate while the gap stays at
#      +0.1000, then whatever improved is not coverage and (a) was right for
#      the wrong reason.
#  (d) THE CONTROL THAT MATTERS MOST. The depth cost survives. 49659502's
#      headline is that solve_rate falls monotonically with DD depth at fixed
#      realized density (rho=0.60: 0.946/0.913/0.871/0.827 over T=2..5,
#      r=-0.998) -- the first clean evidence in this project that reasoning
#      depth is what costs. If that slope FLATTENS at 240000 then the depth
#      signal was a data artefact, the whole finding-17 depth claim weakens,
#      and the L*(T) series has nothing to measure. Prediction: more data
#      lifts every depth bin by roughly the same amount and the slope
#      persists, because finding 13 showed the depth cost is serial
#      computation rather than coverage (at matched parameters L=4/ff=256
#      beats L=2/ff=1024 by +0.49, and no amount of data fixes a model that
#      is too shallow). Falsifier: r at rho=0.60 rises above -0.5.
#
# My record on mechanism predictions in this project is 3 for 5, and the two
# misses -- finding 15's "no overfitting so data will not help" and finding
# 17d's "no gap at 80000" -- were BOTH about data budgets, in the same
# direction: I have twice underestimated how much data this task wants. (a)
# and (b) are written with that on the record.
#
# COST. Cell A is 3x the record's training (~4.5 h) = ~13.5 h; cell B matches
# the record's step count = ~4.5 h; the evals are ~2 h. ~20 h against a 30 h
# limit. A runs first so a timeout truncates the less informative cell.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT="${SCRATCH}/spin_gnn/sudoku_n24_data_2026-10-26"
PREV="${SCRATCH}/spin_gnn/sudoku_n24_leakfree_2026-10-26"
TAG="d0256_h04_L12_ff01024_lr1e-03_pre_shuf"
mkdir -p "$OUT"

COMMON=(--n-nodes 24 --degree 5 --n-colors 4 --shuffle-adj
        --d-model 256 --nhead 4 --num-layers 12 --dim-feedforward 1024
        --n-val-instances 300 --batch-size 32 --n-per-bin 100)

echo "=== cell A: n_train=240000, 300 epochs (3x data, 3x compute) ==="
python -m spin_gnn.sudoku_hparam_sweep "${COMMON[@]}" \
    --n-train-instances 240000 --epochs 300 \
    --out-dir "${OUT}/data240k_ep300"

echo
echo "=== cell B: n_train=240000, 100 epochs (3x data, SAME gradient steps) ==="
python -m spin_gnn.sudoku_hparam_sweep "${COMMON[@]}" \
    --n-train-instances 240000 --epochs 100 \
    --out-dir "${OUT}/data240k_ep100"

echo
echo "=== pre-reg (d): does the depth cost survive? ==="
for cell in data240k_ep300 data240k_ep100; do
    python -m spin_gnn.sudoku_depth_eval \
        --checkpoint "${OUT}/${cell}/${TAG}" \
        --rho 0.75 0.60 --n-pool 600 --min-per-bin 40 \
        --out-csv "${OUT}/depth_${cell}.csv"
done

echo
echo "=== pre-reg (a)+(b): the three-cell shared pool, 0.8540 reference first ==="
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "${PREV}/${TAG}" \
                 "${OUT}/data240k_ep300/${TAG}" \
                 "${OUT}/data240k_ep100/${TAG}" \
    --n-instances 1500 --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500.csv"

echo
echo "=== pre-reg (c): has the memorization gap shrunk? ==="
for cell in data240k_ep300 data240k_ep100; do
    python -m spin_gnn.sudoku_memorization_eval \
        --checkpoint "${OUT}/${cell}/${TAG}" \
        --n-memorization 600 --n-ordering 0 \
        --out-csv "${OUT}/memorization_${cell}.csv"
done
