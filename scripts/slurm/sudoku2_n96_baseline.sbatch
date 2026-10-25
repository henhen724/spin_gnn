#!/bin/bash
#SBATCH --job-name=sud2n96base
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --dependency=afterok:49523504
#
# Can L* even be MEASURED at N=96? One cell, to gate the 36 h scan.
#
# WHY A PROBE AND NOT THE SCAN. sudoku2_n96_layer_scan.sbatch has been written
# and unsubmitted for two days behind a gate I wrote into it: "If N=24 and N=48
# agree that L*(T) is FLAT, finding 2 is confirmed twice and a third point buys
# nothing. Do not run this." The two arms DO agree -- flat in T at both sizes.
# So that gate says don't run it, and I am not going to quietly reinterpret my
# own stopping rule because a 36 h job is sitting there written.
#
# What has changed is a DIFFERENT question, and it deserves to be stated as one
# rather than smuggled in. The N=24 scan put L* at 3; the N=48 scan puts it at
# 4 or at 3 (49523504 decides). Both arms are flat in T. If L*(48)=4, then L* is
# flat in T but RISING IN N, which is a claim about graph size rather than about
# reasoning depth, and it rests on two points -- the minimum from which a trend
# can be asserted at all. N=96 would be its first test. That is a real reason to
# spend the compute, and it is not the reason the old gate was adjudicating.
#
# THE BLOCKER THAT GATE DID NOT CONSIDER, which is why this job exists. L*(T)
# is DEFINED as the smallest num_layers reaching solve_rate >= 0.90 on depth-T
# instances. If no configuration at N=96 clears 0.90, L* is not merely large,
# it is UNDEFINED, and a 36 h scan returns a column of sub-threshold numbers
# with no L* in it. That is not a hypothetical margin: at N=48 the best cell
# clears the bar by 0.05 (L=12 at 0.9533, L=8 the same, L=4 at 0.9033), and
# N=96 roughly doubles mean DD depth again (7.45 vs 6.10 vs 4.25) on sequences
# of ~787 tokens against 402. The scan's own budget makes it worse: it proposed
# n_train=10000 and 100 epochs, half the data and two thirds the epochs of the
# N=48 arm, which by finding 6 -- training-set size is the binding constraint
# and is not saturated even at 80000 -- is the one lever you least want to cut
# on a harder problem. A scan run at that budget could report "L* is undefined
# at N=96" when what it measured was its own data starvation.
#
# ONE CHANGE. This cell is 49509466's L=12 cell with N moved 48 -> 96 and
# nothing else touched: d_model=256, nhead=4, num_layers=12, ff=1024, lr=1e-3,
# pre-norm, n_train=20000, 150 epochs, degree=5, n_colors=4. So the comparison
# against that cell's 0.9533 is attributable to N alone. 150 epochs rather than
# the N=48 baseline's 300 is deliberate and is the same choice: finding 14
# showed 300 epochs scoring BELOW 150 at N=48 because save_checkpoint keeps the
# final epoch and the longer run overfits further, so 150 is both the matched
# budget and the better one.
#
# PRE-REGISTERED:
#  (a) The recipe clears 0.90 aggregate at N=96 on the shared pool, so L* is
#      measurable and the scan is worth its 36 h. Falsifier: it lands below
#      0.90, in which case the scan is NOT submitted as written -- the next
#      move is data (n_train 40000/80000 at N=96), not layers, because a scan
#      whose ceiling is under the threshold cannot report an L* at all. If it
#      lands below 0.70 as well, N=96 is out of reach at this project's budget
#      and the cross-N claim rests on N=24 vs N=48; say that instead of
#      spending a day proving it.
#  (b) Mean DD depth at N=96 is ~7.4 and rho=0.60 spans T=4..9, widening the
#      window again (N=24: T=2..5, N=48: T=3..7). Falsifier: the span does not
#      widen, which would mean N=96 adds cost without adding the depth range
#      that is the entire point of going there.
#  (c) The decline with T continues: solve_rate at the deepest bin is below the
#      shallowest by more than 0.05, as at both smaller sizes (0.980->0.931 at
#      N=24 rho=0.75, 0.943->0.894 at N=48 rho=0.60). The realized-density
#      confound is reported in hints, not as r, per the N=24 fix.
#
# Cost: 49509466 measured 0.116 GPU-h per layer-unit at N=48 / n_train=20000 /
# 150 ep. Per-token cost goes as d^2 + d*s, so total goes as s*(d^2+d*s):
# 787*(65536+201472) / 402*(65536+102912) = 3.1x. Twelve layer-units at 3.1x is
# ~4.3 h. The depth eval with 787-token autoregressive generation is the real
# unknown -- that is what the remaining margin is for, not the training.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT="${SCRATCH}/spin_gnn/sudoku_n96_baseline_2026-10-25"
CKPT="${OUT}/d0256_h04_L12_ff01024_lr1e-03_pre"

echo "=== N=96, the N=48 recipe with N as the only change ==="
python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 96 \
    --degree 5 \
    --n-colors 4 \
    --d-model 256 \
    --nhead 4 \
    --num-layers 12 \
    --dim-feedforward 1024 \
    --n-train-instances 20000 \
    --n-val-instances 300 \
    --batch-size 32 \
    --epochs 150 \
    --n-per-bin 100 \
    --out-dir "$OUT"

echo
echo "=== solve_rate by DD depth: does the T window widen again? ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "$CKPT" \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT}/depth_n96.csv"

echo
echo "=== metric of record: shared pool at n=1500 (fresh pool, new signature) ==="
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "$CKPT" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500.csv"
