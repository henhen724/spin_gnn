#!/bin/bash
#SBATCH --job-name=sud2n96L80k
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=%x_%j.out
#
# THE N=96 LAYER SCAN, now that its gate is satisfied. This supersedes
# scripts/slurm/sudoku2_n96_layer_scan.sbatch, which was written on 10-24 before
# the gate existed and specifies n_train=10000 / 100 epochs / batch 16. We now
# know that recipe cannot work: at 20000 and 150 epochs the L=12 cell scores
# 0.8267 and clears 0.90 in NONE of the six rho=0.60 bins (49524030), so a scan
# at 10000 would have produced a column with no L* in it anywhere -- exactly the
# waste the gate was written to prevent. Do not submit the old file.
#
# WHAT OPENED THE GATE. 49565433 cell B (--n-train-instances 80000, everything
# else identical to 49524030) scores 0.9273 [0.9131, 0.9394] on the shared pool
# -- Wilson lower bound above 0.90 -- and clears 0.90 in ALL SIX rho=0.60 bins
# (T=4..9: 0.977/0.945/0.965/1.000/0.926/0.907) and all four rho=0.75 bins.
# That is pre-registration (d) of that job, the one that gates this one:
# L*(T) at N=96 is now measurable. Cell A (cosine lr at 20000) is not enough --
# 0.8740 aggregate, and it misses at T=8 (0.882) -- so 80000 is the recipe here.
#
# ONE CHANGE from cell B: num_layers. n_train=80000, 150 epochs, batch 32,
# d_model=256, nhead=4, ff=1024, lr=1e-3 CONSTANT, pre-norm, seed=0, N=96,
# degree=5, n_colors=4. Cosine is deliberately NOT used even though it helped
# at 20000 (+0.0473, p=3.5e-05): cell B, the cell that clears the gate, is a
# constant-lr cell, and the scan has to sit in the same recipe as its own
# reference. Stacking both levers is a separate experiment.
#
# L GRID {6, 4, 8}, in that order, with L=12 already measured as cell B.
# Ordered by information, not by size, because a timeout truncates the tail:
# L=6 is the likeliest knee (L* is 3 at N=24 and 4 at N=48), L=4 then decides
# whether the knee is at or below the N=48 value, and L=8 fills the gap to 12.
# After two cells we already know whether L*(96) is <=4, 6, or >6.
#
# THE CONFOUND, stated up front. L*(24)=3 and L*(48)=4 were both measured at
# n_train=20000; this arm is at 80000, because nothing at 20000 clears the
# threshold at N=96 at all. So "L* rises with N" cannot be read off these three
# numbers alone -- more data might lower L*. The companion job
# sudoku2_n48_data_control.sbatch re-measures L=3 and L=4 at N=48 at 80000 to
# close exactly that hole, and it is cheap (~4x smaller per instance). Read the
# two together; neither settles the trend alone.
#
# PRE-REGISTERED:
#  (a) L*(96) at rho=0.60 is >= 6: L=4 does NOT clear 0.90 in every rho=0.60
#      bin. Falsifier: L=4 clears everywhere, which would mean L* saturated
#      somewhere between N=48 and N=96, and that what the deeper cells were
#      buying at N=96 was a substitute for the data budget rather than depth.
#  (b) L*(96) is DEFINED inside {4, 6, 8} -- some cell below 12 clears every
#      rho=0.60 bin. Falsifier: only L=12 clears, putting L*(96) >= 12, a jump
#      from 4 to 12 over one doubling of N, which would be the strongest
#      size-dependence in the project and would need its own replication.
#  (c) L*(T) is FLAT in T inside every clearing column, as in all five series
#      measured so far (20 bins, 0 resolvably rising). Falsifier: a resolvable
#      rise at the largest N -- where a serial cost has the most room to show --
#      which would overturn findings 11/14 at the size that matters most.
#  (d) L=8 is at least as good as L=12 (cell B, 0.9273) on the shared pool,
#      replicating N=48 where L=8 (0.9533) beat L=12 (0.9393) by +0.0140. This
#      separates two things the project has been conflating: the THRESHOLD L*
#      and the OPTIMUM argmax_L. Falsifier: L=12 strictly best at N=96, meaning
#      the optimum does grow with N even if L* does not.
#
# COST. 49565433 ran 14:41:55 total: ~1 min smoke, ~2 h for the L=12 cell at
# 20000, ~8 h for the L=12 cell at 80000, and ~4:40 for two depth evals plus a
# 3-checkpoint n=1500 pool at 787-token generation. Training here is
# (6+4+8)/12 = 1.5x the 80000 cell, so ~12 h; a 3-checkpoint depth eval plus a
# 4-checkpoint pool is ~5.5 h at the measured per-checkpoint rate. ~17.5 h
# expected against a 36 h limit. 128G/4 cores is what the 80000-instance N=96
# build needed in cell B (sganguli caps memory at 32G per core).

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

CELLB="${SCRATCH}/spin_gnn/sudoku_n96_plateau_2026-10-26/d0256_h04_L12_ff01024_lr1e-03_pre"
OUT="${SCRATCH}/spin_gnn/sudoku_n96_layer_scan_2026-10-26"
mkdir -p "$OUT"

for L in 6 4 8; do
    echo
    echo "=== N=96 num_layers=${L}, n_train=80000 (one change from cell B) ==="
    python -m spin_gnn.sudoku_hparam_sweep \
        --n-nodes 96 --degree 5 --n-colors 4 \
        --d-model 256 --nhead 4 --num-layers "$L" --dim-feedforward 1024 \
        --n-train-instances 80000 --n-val-instances 300 --batch-size 32 \
        --epochs 150 --n-per-bin 100 \
        --out-dir "$OUT"
done

echo
echo "=== L*(T) at N=96: the measurement this whole arm exists for ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${OUT}"/d0256_h04_L0[468]_ff01024_lr1e-03_pre \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT}/depth_by_layers_n96.csv"

echo
echo "=== metric of record: the three new cells against cell B on one pool ==="
# Cell B goes first so vs_reference reads against the cell that opened the gate.
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "$CELLB" \
                 "${OUT}/d0256_h04_L04_ff01024_lr1e-03_pre" \
                 "${OUT}/d0256_h04_L06_ff01024_lr1e-03_pre" \
                 "${OUT}/d0256_h04_L08_ff01024_lr1e-03_pre" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500.csv"
