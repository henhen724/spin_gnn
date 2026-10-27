#!/bin/bash
#SBATCH --job-name=sud2n96ff256
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#
# DOES THE MLP WIDTH MOVE L* AT N=96 THE WAY IT DID AT N=48?
#
# HYPOTHESIS, stated before the run. Narrowing dim_feedforward 4x (1024 -> 256)
# raises L*(96) by roughly the same factor it raised L*(48), so L*(96, ff=256)
# is about 12.
#
# The evidence it rests on is finding 14: at N=48, L*(ff=1024) = 4, but
# L=4/ff=256 scores 0.7113 and clears 0.90 in NONE of the five rho=0.60 bins
# while L=6/ff=256 clears all five -- the knee moves 4 -> 6, a factor of 1.5.
# Finding 22 has just put L*(96, ff=1024) = 8. If the factor is a property of
# the architecture rather than of N=48, the thin knee at N=96 is near 12.
#
# WHY THIS CELL AND NOT ANOTHER. This is the missing entry of the grid the
# project has been filling in: L* vs (N, ff). ff=1024 is now measured at all
# three sizes (3, 4, 8 at N = 24, 48, 96); ff=256 is measured only at N=48.
# Without N=96 at ff=256 there is one point of comparison, which cannot
# distinguish "a multiplicative width cost" from "something that happened at
# N=48".
#
# L GRID {12, 8} at ff=256, in that order, n_train=80000. Two cells, ordered by
# information: L=12 decides whether a knee exists inside the range at all, and
# L=8 decides whether narrowing the MLP costs anything at N=96. L=16 is
# deliberately NOT included -- it would push training past the eval stage's
# start and a truncation there would cost the evals, which is the whole point of
# the job. If both cells fail, L*(96, ff=256) > 12 is itself the answer and the
# follow-up is one more cell.
#
# EVERYTHING ELSE IS HELD AT THE ff=1024 LADDER'S VALUES: N=96, degree=5,
# n_colors=4, d_model=256, nhead=4, lr=1e-3 CONSTANT, pre-norm, batch 32,
# 150 epochs, n_train=80000, seed=0. The depth eval uses the same --n-pool 600
# --min-per-bin 40 and the same generator stream, so the T bins come out
# identical to depth_by_layers_n96.csv (rho=0.60: T=4..9 at n=86/146/143/69/
# 68/43; rho=0.75: T=2..5) and the two columns are directly comparable bin by
# bin. The shared pool puts L=8/ff=1024 first so vs_reference reads against the
# cell that defined L*(96).
#
# PARAMETERS, which is the other reason to want this. At N=96, L=12/ff=256 is
# ~5.02M against L=8/ff=1024's 6.58M -- 0.76x. So if the thin-deep cell clears
# every bin it does so with a QUARTER FEWER parameters than the wide knee, and
# finding 13's "depth is the cheaper currency" replicates at the largest size.
#
# PRE-REGISTERED:
#  (a) L=8/ff=256 does NOT clear all six rho=0.60 bins -- the width cost of
#      finding 14 is present at N=96. Falsifier: it clears everywhere, meaning
#      4x of MLP width buys nothing at N=96 and the N=48 knee shift was either
#      N-specific or noise. That would retire the "at ff=1024" caveat that
#      every L* in STATUS.md currently carries.
#  (b) L=12/ff=256 DOES clear all six, i.e. L*(96, ff=256) <= 12 and the width
#      factor is at most 1.5. Falsifier: it fails too, putting the thin knee
#      above 12 -- a super-multiplicative width cost, which no cell in this
#      project has shown and which would need its own replication.
#  (c) On the n=1500 shared pool, L=12/ff=256 is at least as good as
#      L=8/ff=1024 (0.9113) at 0.76x the parameters. Falsifier: resolvably
#      worse, meaning the parameter-efficiency of depth does not survive to
#      N=96 and finding 13 is a small-N phenomenon.
#  (d) Inside whichever thin column clears, L*(T) RISES with T as it did in the
#      wide column (finding 22: L=6/ff=1024 clears T=4,5,6 and fails T=7,8,9,
#      CA p=0.055). Falsifier: flat in T, which would make finding 22's rise a
#      property of the wide column rather than of the problem.
#
# READ EVERY NUMBER THIS PRODUCES AS A DECODING SCORE. shuffle_adj defaults to
# False, so this run sits in the same leaked-order task as findings 14 and 22,
# which is the point -- it extends one ladder rather than starting a second.
# The leak-free counterpart cannot be run yet: at N=24 leak-free even L=12 at
# n_train=80000 scores 0.854, below the 0.90 gate, so no L* is defined there at
# all until 49705064 reports a working budget.
#
# COST. 49639312 trained L=6+4+8 at ff=1024 (18 layers) in ~12 h, i.e. ~0.67 h
# per layer, and spent ~5.5 h on a 3-checkpoint depth eval plus a 4-checkpoint
# n=1500 pool. Here: 20 layers at ff=256, where the MLP is 4x smaller but
# attention at 801 tokens dominates, so ~0.55 h per layer -> ~11 h training,
# plus ~4 h for a 2-checkpoint depth eval and a 3-checkpoint pool. ~15 h
# expected against a 24 h limit. 128G/4 cores is what the 80000-instance N=96
# build needs (sganguli caps memory at 32G per core).

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

WIDE="${SCRATCH}/spin_gnn/sudoku_n96_layer_scan_2026-10-26/d0256_h04_L08_ff01024_lr1e-03_pre"
OUT="${SCRATCH}/spin_gnn/sudoku_n96_ff256_2026-10-27"
mkdir -p "$OUT"

for L in 12 8; do
    echo
    echo "=== N=96 num_layers=${L}, ff=256 (one change from the ff=1024 ladder) ==="
    python -m spin_gnn.sudoku_hparam_sweep \
        --n-nodes 96 --degree 5 --n-colors 4 \
        --d-model 256 --nhead 4 --num-layers "$L" --dim-feedforward 256 \
        --n-train-instances 80000 --n-val-instances 300 --batch-size 32 \
        --epochs 150 --n-per-bin 100 \
        --out-dir "$OUT"
done

echo
echo "=== L*(T) in the thin column, on the same bins as the wide one ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${OUT}/d0256_h04_L08_ff00256_lr1e-03_pre" \
                 "${OUT}/d0256_h04_L12_ff00256_lr1e-03_pre" \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT}/depth_by_layers_n96_ff256.csv"

echo
echo "=== metric of record: thin cells against the wide knee on one pool ==="
# The wide knee goes first so vs_reference reads against L*(96, ff=1024) = 8.
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "$WIDE" \
                 "${OUT}/d0256_h04_L08_ff00256_lr1e-03_pre" \
                 "${OUT}/d0256_h04_L12_ff00256_lr1e-03_pre" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500.csv"
