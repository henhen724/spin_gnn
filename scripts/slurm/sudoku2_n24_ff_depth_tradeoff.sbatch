#!/bin/bash
#SBATCH --job-name=sud2n24ffdep
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#
# Is the L=3 knee serial composition, or just parameters?
#
# WHERE THIS COMES FROM. 49433454 (finding 11) found L*(T) flat at 3 for every
# reasoning depth T=1..5, and -- the informative part -- that L=2 scores only
# 0.686 even at T=1, a SINGLE deduction round. So the third layer is not the
# first rung of a chaining ladder; it is a fixed cost paid before any chaining
# happens. That leaves two readings of what it buys, and they make opposite
# predictions here:
#
#   H1 capacity    The third layer is parameters. 1.65M (L=2, ff=1024) is not
#                  enough and 2.44M (L=3) is, and WHERE the parameters sit does
#                  not matter much. Widening the MLP at L=2 should then recover
#                  the knee without adding a layer.
#   H2 serial      Some minimum number of sequential attention->MLP blocks is
#                  needed whatever their width -- two rounds of global mixing
#                  cannot be folded into one wider one. Then no ff buys it.
#
# This is also the "MLP depth" axis of the original plan: 49433454 pinned
# ff=1024 so that only num_layers moved, which was right for isolating depth
# but leaves the depth/width trade-off untested.
#
# THE DESIGN: TWO BUDGETS, EACH SPENT TWO WAYS. Counted by instantiating
# SATReasoningTransformer rather than estimated, because predictions (a) and
# (b) are matched-parameter claims and an arithmetic slip would void both. The
# counts fall out into two pairs matched to 0.2%:
#
#   ~1.86M   L=2 ff=1024   1,859,618   measured: 0.467 agg, 0.686 at T=1
#            L=4 ff=256    1,863,202   NEW
#   ~2.65M   L=3 ff=1024   2,649,378   measured: 0.957 agg  <- the knee
#            L=2 ff=1792   2,647,586   NEW
#            L=6 ff=256    2,654,754   NEW
#
# So each budget is available in a wide-shallow and a thin-deep allocation, and
# the 2.65M budget -- the one that contains the knee -- is available in three.
# Under H1 the members of a pair score alike; under H2 the thin-deep member
# wins and the wide-shallow one stays stuck.
#
# (These absolute counts run ~208k above the ones in 49433454's sweep.csv,
# a constant offset in the embedding and output head from the vocab size the
# sweep actually built. It is identical across every cell here, so it cancels
# in the pairings, which are what the predictions rest on.)
#
# GRID (4 new cells; the three reference cells already exist in
# sudoku_layer_scan_2026-10-24_n24 and are NOT retrained):
#     (L=2, ff=1792)   2.65M   param-matched to the knee, spent on width
#     (L=2, ff=4096)   5.01M   1.89x the knee's budget, the strong form of H1
#     (L=4, ff=256)    1.86M   param-matched to L=2/ff=1024, spent on depth
#     (L=6, ff=256)    2.65M   param-matched to the knee, spent on depth
#
# PRE-REGISTERED PREDICTIONS.
#  (a) THE DISCRIMINATOR, at matched parameters: L=4/ff=256 reaches aggregate
#      solve_rate >= 0.90 while L=2/ff=1024 is at 0.467. Under H2 this holds --
#      depth is what the budget should be spent on. Under H1 the two land
#      within 0.10 of each other, since they have the same parameters.
#  (b) L=2/ff=4096 stays below 0.90 despite carrying 1.89x the knee's
#      parameters. Falsifier: it clears 0.90, in which case the knee is
#      capacity after all, finding 11's "fixed cost" reading is wrong about
#      WHAT is fixed, and the honest restatement is that two blocks suffice if
#      wide enough. The weaker L=2/ff=1792 cell decides the same question at
#      exactly the knee's budget rather than past it.
#  (c) Monotone in ff at fixed L=2 (1024 -> 1792 -> 4096 never decreasing by
#      more than 0.02). This is the sanity check: if a wider MLP makes things
#      WORSE at fixed depth, 150 epochs is not enough to fit the wide cells and
#      (b) is confounded with optimization rather than capacity. Report and
#      stop if so, do not read (b) as support for H2.
#  (d) L*(T) stays flat in T for every cell that clears the threshold at all,
#      i.e. the ff axis does not reintroduce a depth-dependent L*. Finding 11
#      was measured on one ff; this checks it is not an artifact of ff=1024.
#
# EVERYTHING ELSE PINNED at 49433454's settings so the reference cells are
# usable without retraining: d_model=256, nhead=4, lr=1e-3, pre-norm, 150
# epochs, n_train=20000, n_val=300, same seed, N=24, degree 5, n_colors 4.
# rho defaults to 0.3, below (5,4)'s rho_min 0.478, so it clips nothing here
# -- the same reason it was safe in 49433454.
#
# OUTPUT DIR is deliberately NOT sudoku_layer_scan_2026-10-24_n*: that glob
# feeds make_layer_scan_deck, whose by_layers() keys on num_layers alone and
# would silently collide two cells that share L with different ff.
#
# CELL ORDER is (4,256), (2,1792), (6,256), (2,4096): the matched pair that
# decides (a) runs first, so even a wall-clock overrun or a mid-flight failure
# leaves the discriminating comparison measured. (2,4096) is the most expensive
# and the least load-bearing -- it only strengthens a question (2,1792) already
# answers at matched budget -- so it goes last.
#
# Cost: 49433454 did 36 layer-units at ff=1024 in 2:52, so ~4.8 min per unit,
# and MLP cost scales with ff while attention does not. Estimating (4,256) at
# 2.4 units, (2,1792) at 3.5, (6,256) at 3.6 and (2,4096) at 8 gives ~1.4 h,
# plus four data builds and a depth eval: ~2.5 h. 6:00:00 is margin, since the
# ff scaling is extrapolated rather than measured.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_ff_depth_tradeoff_2026-10-25_n24"
mkdir -p "$OUT_DIR"

for CELL in "4 256" "2 1792" "6 256" "2 4096"; do
    set -- $CELL
    L=$1; FF=$2
    echo "=== N=24 num_layers=${L} dim_feedforward=${FF} ==="
    python -m spin_gnn.sudoku_hparam_sweep \
        --n-nodes 24 \
        --degree 5 \
        --n-colors 4 \
        --d-model 256 \
        --nhead 4 \
        --num-layers "$L" \
        --dim-feedforward "$FF" \
        --n-train-instances 20000 \
        --n-val-instances 300 \
        --batch-size 32 \
        --epochs 150 \
        --n-per-bin 100 \
        --out-dir "$OUT_DIR"
done

echo
echo "=== solve_rate by DD depth at fixed density, depth/width trade-off cells ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${OUT_DIR}"/d0256_h04_L*_ff*_lr1e-03_pre \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT_DIR}/depth_by_ff.csv"
