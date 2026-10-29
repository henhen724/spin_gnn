#!/bin/bash
#SBATCH --job-name=sud2n96L07
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=%x_%j.out
#
# THE LAST UNRESOLVED RUNG OF THE WIDE LADDER: L=7 at ff=1024, N=96.
#
# WHY THIS CELL AND NOT L=10. STATUS said, in finding 22's caveat and in next
# step 2, that "L=10 is not run, so the threshold is in (6, 8]". That is
# incoherent and it is fixed in the same commit as this file. The wide ladder
# at N=96 reads, on the six rho=0.60 bins (49639312 / finding 22):
#
#     L=4   0 of 6
#     L=6   3 of 6   <- misses the three DEEPEST bins
#     L=8   6 of 6   <- L* is read off here
#     L=12  ties L=8 on the n=1500 pool (0.9273 vs 0.9113, p=0.091)
#
# So L*(96, ff=1024) is 7 or 8 and nothing else. L=10 sits ABOVE the knee and
# cannot narrow (6, 8] by a single rung; L=7 is the only cell that can. This
# is a bug in my own notes, caught on 10-28 while writing up finding 26, and
# it cost nothing because the L=10 job was never submitted.
#
# WHY IT IS WORTH ~6 GPU-HOURS. Every width factor in the completed L* grid is
# a ratio against the wide knee at that N:
#
#     N=24   3 -> 3     1.00x   (finding 26, 50007138)
#     N=48   4 -> 6     1.50x   (finding 14)
#     N=96   8 -> >12   >1.50x  (finding 25, 49816217)
#
# The N=96 entry is the one that carries the "super-multiplicative in N"
# claim, and its denominator is exactly the number this job resolves. If L=7
# clears all six bins the factor becomes >12/7 = >1.71x and the claim gets
# STRONGER; if it does not, L*=8 is confirmed by measurement rather than by
# the absence of a measurement, which is what it rests on today.
#
# ONE CHANGE from 49639312's cells: num_layers = 7. Everything else is pinned
# to the recipe that arm used, because a knee is only comparable within one
# recipe: n_train=80000, 150 epochs, batch 32, d_model=256, nhead=4, ff=1024,
# lr=1e-3 CONSTANT (not cosine -- cell B, the reference, is a constant-lr
# cell), pre-norm, seed 0, N=96, degree=5, n_colors=4. shuffle_adj=False, so
# this stays in the same leaked-order task as the rest of the ladder: it
# EXTENDS one ladder rather than starting a second.
#
# NO WARMUP HERE, deliberately. Job 50008605 is testing --warmup-epochs 5 on
# the THIN column, where finding 25 found a grammar collapse that looks like
# an early-training failure. The wide column has never shown that failure
# (well_formed_rate 0.97-1.00 everywhere), so adding warmup here would be a
# second change against the ladder this cell has to join. If 50008605 shows
# warmup moves a wide-column number, this cell is re-run with it; that is a
# consequence to record, not a reason to pre-empt it.
#
# PRE-REGISTERED PREDICTIONS, with falsifiers.
#
# (a) L=7 clears 4 or 5 of the six rho=0.60 bins, so L*(96, ff=1024) = 8
#     STANDS. Reason: the ladder is roughly linear in L near the knee at
#     ~1.5 bins per layer (0 -> 3 -> 6 over L=4 -> 6 -> 8), which puts L=7 at
#     ~4.5 of 6. FALSIFIER: 6 of 6, i.e. L* = 7, and then the N=96 width
#     factor in the grid must be restated as >1.71x and every "L*(96)=8" in
#     STATUS and in the finding 22/25/26 decks is wrong by one rung.
#     The other falsifier is 0-3 of 6, which would make the ladder NON-
#     monotone in L and indict the whole gate.
#
# (b) The bins L=7 misses, if any, are the DEEPEST ones -- same shape as
#     L=6's failure. Tested by Cochran-Armitage over the ordered bins.
#     FALSIFIER: it misses shallow bins and clears deep ones, which would
#     mean the gate is not reading reasoning depth at all at this size.
#
# (c) On the n=1500 shared pool L=7 lands strictly BETWEEN L=6 and L=8 and is
#     resolvably below L=8. FALSIFIER: it ties or beats L=8 (p > 0.05 or
#     delta >= 0), which would say the pool cannot separate adjacent rungs and
#     that the gate, not the pool, is doing all the work in this ladder.
#
# (d) well_formed_rate >= 0.95 in every bin. The wide column has never lost
#     the output grammar and L=7 is not a deep-thin cell. FALSIFIER: a
#     collapse like finding 25's 0.853, which would mean the grammar failure
#     is about DEPTH alone and not about depth-at-narrow-width, and would put
#     a confound under the whole wide ladder rather than just the thin one.
#
# COST. 49639312 trained 18 layer-units (L = 6, 4, 8) plus three depth evals
# and a four-checkpoint pool in 17:45:48. Scaling the training portion gives
# ~5.5 h for 7 layer-units, plus one depth eval (~1 h at N=96's 787-token
# generation) and one three-checkpoint n=1500 pool (~2-3 h, the dominant
# unknown -- 49816217's pool stage was the long pole there too). 16:00:00 is
# sized for the pool being twice as dear as expected, not for the training.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

# 49639312's output directory: the ladder this cell joins. Read-only here.
SCAN="${SCRATCH}/spin_gnn/sudoku_n96_layer_scan_2026-10-26"
OUT="${SCRATCH}/spin_gnn/sudoku_n96_L07_2026-10-28"
mkdir -p "$OUT"

echo "=== N=96 num_layers=7, ff=1024, n_train=80000 (one change from 49639312) ==="
python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 96 --degree 5 --n-colors 4 \
    --d-model 256 --nhead 4 --num-layers 7 --dim-feedforward 1024 \
    --n-train-instances 80000 --n-val-instances 300 --batch-size 32 \
    --epochs 150 --n-per-bin 100 \
    --out-dir "$OUT"

echo
echo "=== L*(T) for the new rung, on the same bins as L=4/6/8 ==="
# Same --rho/--n-pool/--min-per-bin as 49639312, off the same seeded generator
# stream, so these bins line up row for row with depth_by_layers_n96.csv and
# the columns compare bin by bin.
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${OUT}/d0256_h04_L07_ff01024_lr1e-03_pre" \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT}/depth_n96_L07.csv"

echo
echo "=== metric of record: L=7 against its two neighbours on one pool ==="
# L=8 FIRST so vs_reference and delta read against the current knee, which is
# the quantity (c) is about. L=6 is included so the new rung is bracketed by a
# paired comparison on both sides rather than against the knee alone.
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "${SCAN}/d0256_h04_L08_ff01024_lr1e-03_pre" \
                 "${OUT}/d0256_h04_L07_ff01024_lr1e-03_pre" \
                 "${SCAN}/d0256_h04_L06_ff01024_lr1e-03_pre" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500.csv"
