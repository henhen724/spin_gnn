#!/bin/bash
#SBATCH --job-name=sud2n24ff256
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#
# THE LAST CELL OF THE L* vs (N, ff) GRID: L*(24, ff=256).
#
# HYPOTHESIS, stated before the run. The MLP-width factor on L* GROWS with N,
# so at N=24 it is at its smallest -- 1.0, i.e. L*(24, ff=256) = 3 = L*(24,
# ff=1024), and narrowing the MLP 4x costs no layers at the small end.
#
# WHY ONE CELL IS ENOUGH. The grid already reads 1.5x at N=48 (finding 14:
# 4 -> 6) and > 1.5x at N=96 (finding 25: 8 -> above 12). At N=24 the wide
# column is measured (finding 11, job 49433454: L=3 is the first to clear all
# four rho=0.60 bins) and the THIN column is already measured at L=4 and L=6 --
# job 49510197 trained them for finding 13's matched-parameter test at exactly
# this recipe (20000 instances, 150 epochs, d_model 256, nhead 4, lr 1e-3
# constant, pre-norm, batch 32) and scored them through the same
# `--rho 0.75 0.60 --n-pool 600 --min-per-bin 40` depth eval, so the bins are
# identical row by row (rho=0.60: T=2..5 at n=92/195/163/98):
#
#     rho=0.60          T=2     T=3     T=4     T=5    clears 0.90?
#     L=2 ff=1024      0.359   0.379   0.368   0.449   no
#     L=3 ff=1024      0.989   0.974   0.963   0.949   YES  <- L*(24, ff=1024)
#     L=4 ff=1024      0.978   0.995   0.988   0.990   yes
#     L=4 ff=0256      1.000   0.985   0.975   0.959   YES  <- thin knee <= 4
#     L=6 ff=0256      1.000   0.995   0.994   0.990   yes
#
# So L*(24, ff=256) is 3 or 4 and nothing else, and the ONLY cell that can
# decide it is L=3/ff=256. L=2/ff=256 is added as the ladder floor (~7 min of
# GPU) so the thin column has a measured failure below its knee rather than an
# assumed one -- without it, "the thin knee is 3" rests on never having checked
# that the thin column fails anywhere.
#
# ONE CHANGE from the cells it is compared against: --dim-feedforward 256
# against the ff=1024 ladder, and --num-layers against the thin cells. Data
# budget, epochs, lr, schedule, norm, batch, seed and the instance distribution
# are the ladder's. shuffle_adj stays False, so these are DECODING scores in the
# same leaked-order task as findings 11, 13, 14, 22 and 25 -- the point is to
# extend one ladder, not to start a second.
#
# PRE-REGISTERED:
#  (a) L=3/ff=256 clears all four rho=0.60 bins, so L*(24, ff=256) = 3 and the
#      width factor at N=24 is 1.0. The reasoning: at N=24 narrowing the MLP
#      cost L=4 nothing measurable (0.959-1.000 thin vs 0.978-0.995 wide), and
#      finding 11 reads the N=24 knee as a fixed SERIAL cost paid before any
#      chaining (L=2 collapses to 0.36 at every T), which MLP width should not
#      move. Falsifier: it misses a bin -- most likely T=5, where the wide L=3
#      sits only 0.049 above the gate -- giving a factor of 1.33 and making the
#      width cost present at EVERY N and monotone, 1.33 / 1.5 / >1.5.
#  (b) On the n=1500 shared pool, L=3/ff=256 is resolvably WORSE than
#      L=3/ff=1024 even if it clears every bin: the four-bin gate is a coarse
#      instrument (finding 25) and the pool caught 7.1 points of width cost at
#      N=96. Falsifier: a tie (McNemar p > 0.05), meaning MLP width buys
#      literally nothing at the small end and the super-multiplicative story has
#      a clean zero to anchor it.
#  (c) L=2/ff=256 clears no bin and does not beat L=2/ff=1024 (0.359-0.449)
#      resolvably: below the knee the deficit is serial, not capacity.
#      Falsifier: the thin L=2 beats the wide L=2 -- narrowing the MLP would
#      then HELP below the knee, which would reframe finding 13's margins as
#      partly an anti-width effect rather than a pro-depth one.
#  (d) well_formed_rate stays >= 0.95 in both new cells, i.e. finding 25's
#      generative collapse (L=12/ff=256 at N=96 fell to 0.853 while its
#      CONDITIONAL solve rate stayed near the wide knee's) is specific to
#      deep-thin stacks at large N and not a property of ff=256. Falsifier: it
#      collapses here too, which would make "a thin MLP breaks the output
#      grammar" general and would retro-explain the N=48 thin knee as well.
#
# COST. Job 49433454 trained 36 layers at N=24/ff=1024 plus a 7-checkpoint depth
# eval in 2:52:14, and job 48999377 scored an n=1500 AND an n=6000 pool at N=24
# in 00:10:27. Here: 5 layers at ff=256 (~15 min), a 2-checkpoint depth eval
# (~10 min) and a 5-checkpoint n=1500 pool (~20 min) -- under an hour, against a
# 4 h limit. 32G/2 cores is what the other N=24 scans used.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

WIDE="${SCRATCH}/spin_gnn/sudoku_layer_scan_2026-10-24_n24"
THIN="${SCRATCH}/spin_gnn/sudoku_ff_depth_tradeoff_2026-10-25_n24"
OUT="${SCRATCH}/spin_gnn/sudoku_n24_ff256_2026-10-28"
mkdir -p "$OUT"

# L=3 first: it is the cell that decides the knee, so if anything goes wrong
# later the expensive question is already answered.
for L in 3 2; do
    echo
    echo "=== N=24 num_layers=${L}, ff=256 (one change from the ff=1024 ladder) ==="
    python -m spin_gnn.sudoku_hparam_sweep \
        --n-nodes 24 --degree 5 --n-colors 4 \
        --d-model 256 --nhead 4 --num-layers "$L" --dim-feedforward 256 \
        --n-train-instances 20000 --n-val-instances 300 --batch-size 32 \
        --epochs 150 --n-per-bin 100 \
        --out-dir "$OUT"
done

echo
echo "=== the two new rungs, on the same bins as every other N=24 column ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${OUT}/d0256_h04_L03_ff00256_lr1e-03_pre" \
                 "${OUT}/d0256_h04_L02_ff00256_lr1e-03_pre" \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT}/depth_n24_ff256.csv"

echo
echo "=== metric of record: the thin column against the cell that defines L*(24) ==="
# L=3/ff=1024 goes first so vs_reference reads against L*(24, ff=1024) = 3.
# L=4 and L=6 at ff=256 are 49510197's checkpoints, included because they cost
# nothing here and make the thin column a complete ladder on one pool.
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "${WIDE}/d0256_h04_L03_ff01024_lr1e-03_pre" \
                 "${OUT}/d0256_h04_L03_ff00256_lr1e-03_pre" \
                 "${OUT}/d0256_h04_L02_ff00256_lr1e-03_pre" \
                 "${THIN}/d0256_h04_L04_ff00256_lr1e-03_pre" \
                 "${THIN}/d0256_h04_L06_ff00256_lr1e-03_pre" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500.csv"
