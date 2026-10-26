#!/bin/bash
#SBATCH --job-name=sud2n24shuf
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=%x_%j.out
#
# THE FIRST RUN IN THIS PROJECT THAT IS ACTUALLY MEASURING DEDUCTION.
#
# Job 49655447 established that every checkpoint here is substantially decoding
# an encoding artifact rather than solving graph colouring. planted_regular_graph
# appends edges in class-pair block order, so adj[v] arrives sorted by the
# neighbour's TRUE COLOUR for 100.0% of nodes at N=24/48/96, and encode_instance
# emits that order verbatim. Permuting adj[v] at eval time -- preserving the edge
# set, the colouring, the hints and the DD trace, all asserted equal -- does this:
#
#   N=24 record (0.9960)  identity 0.9975  shuffle 0.1450  byid 0.0825
#   N=96 cell B           identity 0.9400  shuffle 0.0375  byid 0.0325
#   N=48 L=8 ff=256       identity 0.9575  shuffle 0.0275  byid 0.0000
#
# all at p < 1e-100, against within_block (the in-distribution control that
# preserves the colour code while changing the order) tying identity on all
# three: -0.0025 p=1, -0.0225 p=0.175, +0.0000 p=1. The design check passes, so
# the collapse is attributable to the colour code and not to distribution shift.
#
# THE FIX, verified: SudokuReasoningConfig.shuffle_adj permutes adj[v] in
# build_instances from a dedicated rng stream (the 40M offset, following the
# existing 10/20/30M convention). Colour-sorted fraction 100.0% -> 5.7% at N=24
# and 9.1% at N=48 (chance for d=5 over q=4), the default path's generator stream
# stays bit-identical, and the edge set, colouring, hints and DD trace are
# unchanged instance by instance. It lives in build_instances rather than in
# planted_regular_graph because five other scripts call that function for the
# edge set alone and would have their streams perturbed for nothing.
#
# THIS CELL IS ONE CHANGE FROM THE RECORD MODEL: --shuffle-adj added to the
# config of sudoku_dd_..._sharedpool0.9960_job48912548, i.e. N=24, degree=5,
# n_colors=4, rho=0.3, rho_max=0.95, removal hints, d_model=256, nhead=4, L=12,
# ff=1024, pre-norm, lr=1e-3 constant, batch=32, 300 epochs, n_train=80000,
# n_val=300, seed=0. Tagged _shuf, a new tag segment, because this is a
# different instance distribution rather than a different hyperparameter and
# must never be mistaken for a comparable rerun. agg_eval's INSTANCE_KEYS now
# includes shuffle_adj, so check_comparable REFUSES to pool _shuf with any
# earlier cell -- deliberately: the paired comparison belongs in part B below,
# where the same graphs are scored under both orderings.
#
# WHY N=24 AND WHY ONLY ONE CELL. N=24 is the cheapest config in the project and
# the one holding the record, so it is where "can a transformer learn DD at all
# without the shortcut" is answered for the least GPU time (~5 h; the record job
# 48912548 ran 5:33:11 including eval). The answer sets the scope of everything
# else -- whether the L*(N) series, finding 13 and finding 15 need re-measuring
# or re-interpreting -- so it runs alone and gates the rest instead of my
# guessing the scope now.
#
# PRE-REGISTERED:
#  (a) Solve_rate on its own distribution lands BELOW 0.9960 at p<0.05 but ABOVE
#      0.50: a real cost, not a collapse. Reason: the sorted order gave an
#      attention-free way to tell which neighbour holds which colour, and
#      without it "do my neighbours show all q-1 other colours" needs
#      content-based attention over the adjacency list per colour, which is a
#      harder circuit but not an impossible one at L=12 and 80000 instances.
#      Falsifiers, both informative: >=0.99 means the order was pure redundancy
#      and the existing conclusions transfer once re-measured; <=0.50 means DD
#      at N=24 is largely out of reach for this architecture and the project's
#      headline numbers were the leak.
#      My record on mechanism predictions here is 1 for 2 -- finding 15's
#      mechanism was wrong, 49655447's pre-reg (a) was right -- so this is a
#      genuine prediction, not a formality.
#  (b) The model is ORDER-INSENSITIVE: within_block, shuffle and byid each tie
#      identity (p>0.05). Reason: trained on a permuted order, no ordering
#      convention is available to decode, so whatever it learned is
#      content-based. Falsifier: a resolvable spread across colour-free
#      orderings, meaning it found some other positional regularity and the fix
#      is incomplete.
#  (c) Re-instating the leak does NOT help it: bycolor ties identity. This is
#      the decisive control and the reason bycolor was added to ORDERINGS.
#      Falsifier: bycolor resolvably ABOVE identity, which would mean the
#      shortcut is attractive enough that withholding it during training is not
#      sufficient and the ENCODING needs to change (a set-based adjacency
#      representation), not just the generator.
#  (d) No memorization gap at 80000 (|gap|<0.03, p>0.05), replicating
#      49655447's N=96/80000 result (-0.0133, p=0.355) at N=24. Low stakes; the
#      data budget is unchanged and that result is the one I already have.
#
# COST. ~5 h training, ~0.5 h depth eval, ~0.5 h for part A at n=600 plus part B
# at n=400 x 5 orderings (49655447 did 3 checkpoints x 4 orderings plus 3 part-A
# pairs in 18:42 total, so the eval here is minutes, not hours). ~6 h expected
# against a 14 h limit.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT="${SCRATCH}/spin_gnn/sudoku_n24_leakfree_2026-10-26"
mkdir -p "$OUT"
CKPT="${OUT}/d0256_h04_L12_ff01024_lr1e-03_pre_shuf"

echo "=== N=24 record config + --shuffle-adj (one change) ==="
python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 24 --degree 5 --n-colors 4 \
    --shuffle-adj \
    --d-model 256 --nhead 4 --num-layers 12 --dim-feedforward 1024 \
    --n-train-instances 80000 --n-val-instances 300 --batch-size 32 \
    --epochs 300 --n-per-bin 100 \
    --out-dir "$OUT"

echo
echo "=== where does it fail? solve_rate by DD depth (pre-reg a) ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "$CKPT" \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT}/depth_n24_leakfree.csv"

echo
echo "=== the headline number, same pool protocol as the 0.9960 record ==="
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "$CKPT" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500.csv"

echo
echo "=== order sensitivity + memorization (pre-reg b, c, d) ==="
# Part B scores the SAME graphs under all five orderings, which is where the
# paired comparison against the colour-sorted order lives now that agg_eval
# correctly refuses to pool across shuffle_adj.
python -m spin_gnn.sudoku_memorization_eval \
    --checkpoint "$CKPT" \
    --n-memorization 600 \
    --n-ordering 400 \
    --out-csv "${OUT}/ordering_and_memorization.csv"
