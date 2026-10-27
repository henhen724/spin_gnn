#!/bin/bash
#SBATCH --job-name=sud2saeLF
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#
# DOES THE LEAK-FREE MODEL THINK ABOUT THE RELEVANT NODES AND EDGES AT THE
# RELEVANT TIMES?
#
# The cleanest controlled comparison this project can make. Two checkpoints,
# identical in every hyperparameter -- N=24, degree=5, q=4, rho=0.3,
# rho_max=0.95, removal hints, d_model=256, nhead=4, L=12, ff=1024, pre-norm,
# lr=1e-3 constant, batch=32, 300 epochs, n_train=80000, n_val=300, seed=0 --
# differing ONLY in --shuffle-adj:
#
#   A  the 0.9960 record (job 48912548), trained on the color-sorted adjacency.
#      Job 49655447 showed it is decoding that order: permuting adj[v] while
#      preserving the edge set, colouring, hints and DD trace drops it
#      0.9975 -> 0.1450 (p=4e-103), with within_block (colour code intact,
#      order changed) tying identity at p=1.
#   B  the leak-free retrain (job 49659502), 0.8540 [0.8352, 0.8710] on the
#      same n=1500 pool protocol, and provably order-INSENSITIVE: identity
#      0.8650, within_block -0.0075 (p=0.664), shuffle -0.0150 (p=0.327),
#      byid +0.0075 (p=0.69), and -- the decisive control -- bycolor, which
#      re-instates the leak, -0.0100 (p=0.541). Handing it the shortcut does
#      not help it, so whatever it is doing is content-based.
#
# B solves 85% of held-out instances without any positional shortcut. The
# question this job asks is WHAT it is doing instead, and whether that
# resembles DD. Each checkpoint is analyzed on ITS OWN distribution --
# sudoku_sae now threads shuffle_adj out of the checkpoint config, so A gets
# colour-sorted instances and B gets permuted ones. Feeding B colour-sorted
# instances would be measuring it off-distribution.
#
# WHAT IS NEW IN THE SCORING (all three were defects in the 2026-10-19 run):
#
#  1. EDGE PREDICATES. The old predicate set was entirely node-indexed:
#     frontier[v], known[v], color[v]=c, next_node[v]. DD does not deduce a
#     node from a node, it deduces it from the colours on its known incident
#     EDGES, so the set now also contains
#       justifies_next[u]  u is a KNOWN neighbour of the node being emitted,
#                          i.e. one of the edges forcing its colour (4.04 true
#                          per position)
#       nbr_of_next[u]     u is any neighbour of it (5.00 true per position),
#                          separating "attends to the graph table" from
#                          "attends to the part that constrains the answer"
#       next_node_distinct_colors=j
#                          the count DD actually thresholds. n_known_nbrs can
#                          be 4 with only 2 distinct colours, in which case the
#                          node is NOT deducible, so a feature matching this
#                          and not next_node_nbr_count is tracking the rule
#                          rather than the degree.
#     Verified against real instances in tests/test_sudoku_activations.py:
#     justifier_mask is exactly the known neighbourhood of next_node, its
#     colours are exactly the q-1 colours other than the forced one, and
#     n_known_colors[next_node] == q-1 at EVERY position, which is the DD
#     firing rule itself.
#
#  2. A NULL CONTROL. Every feature takes an argmax over 256 predicates, which
#     is worth a large F1 by itself: measured on pure noise features firing at
#     30%, best_f1 lands at 0.38-0.41 and the position-rolled null at
#     0.39-0.41 -- ALL of it is the search. best_f1 alone therefore cannot
#     distinguish a match from the argmax, and the 2026-10-19 report of
#     "frontier F1=0.254" was uninterpretable for exactly this reason.
#     beats_null = above all 19 rolls, a one-sided randomization test at level
#     0.05, and the summary carries expected_false_beats_null = n_alive/20 so
#     n_beats_null is never read without its chance baseline.
#
#  3. AN ALIVENESS FLOOR. `firing_rate > 0` called all 2048 features alive at
#     L4/8/12 in 2026-10-19, including features firing on one position in
#     26448. --alive-rate 1e-3 is a real floor and both counts are reported.
#
# SPARSITY TARGET, SET BY MEASUREMENT. Measured on 3468 real positions: 55.8
# DD predicates are simultaneously true per position (p10 46, p90 63), of
# which known and color contribute 18.3 each. The old --target-l0 32 is
# therefore BELOW the complexity of the ground truth -- a dictionary that
# sparse structurally cannot carry a faithful DD code, and would force low F1
# on the bookkeeping families whatever the model does. The main pass uses 64,
# which brackets the p90 of the true count. Because that is a judgement call
# that could drive the answer, a second pass repeats layers 8 and 12 at 32 so
# the conclusion can be checked against the sparsity target rather than
# resting on it.
#
# PRE-REGISTERED:
#  (a) LAYER 0, the sharpest one. The 2026-10-19 dictionary had 23 of its 24
#      interpretable layer-0 features in the next_node family -- at the
#      token+position embedding, before any attention, which can only be the
#      sorted-order confound (finding 16). Prediction: model B has NO
#      next_node-family features above null at layer 0, and model A still
#      does. There is no mechanism left for position to predict the next node
#      once adj[v] is permuted, so a null result here is a direct check that
#      the fix removed what it was supposed to remove.
#  (b) EDGES. Model B has justifies_next features above null in the middle
#      layers, model A has fewer or none. Reason: B must locate the known
#      neighbours by content, and that computation has to be represented
#      somewhere; A can read the forced colour off the gap in the sorted
#      order without ever representing which neighbours are known.
#      Falsifier, and a real possibility: B shows no edge structure above
#      null either, which would mean it reaches 0.854 by some distributed
#      route this predicate set cannot see -- a negative result about the
#      method, reportable as such and NOT as "it isn't doing DD".
#  (c) FRONTIER. best_frontier_f1 is higher for B than for A at every layer
#      above 0. The frontier is the DD predicate itself and the one thing a
#      genuine solver must compute.
#  (d) Neither model produces a DD-shaped dictionary outright: in both,
#      known/color bookkeeping families outnumber frontier+justifies_next
#      features. Low confidence and low stakes, stated so that a surprise
#      counts. 36.6 of the 55.8 true predicates per position ARE bookkeeping,
#      so a dictionary that merely mirrors the state distribution lands there.
#
#  My record on mechanism predictions in this project is 2 for 3 -- finding
#  15's mechanism was wrong, 49655447's pre-reg (a) and 49659502's (a)-(c)
#  were right -- so these are predictions, not decoration.
#
# THIS IS A RERUN. Job 49702811 ran this script and finished in 00:24:56 with
# ExitCode 0:0 against a ~4 h estimate, which is what gave it away: the l1
# search never searched. train_sae_to_target_l0 only ever RAISED l1_coeff and
# accepted on `l0 <= target + tolerance`, so starting from --l1-coeff 1.0 --
# already far too strong -- the first fit of all 18 sweeps landed at L0=2..16
# against target 64, satisfied the one-sided test, broke on iteration 1 and
# reported hit_target_l0=True. Every dictionary it produced had ~130 of 4096
# features alive. Against a ground truth with 55.8 simultaneously-true
# predicates, those cannot carry the code, so its zero F1 on nbr_of_next and
# justifies_next described the dictionary and not the model, and none of the
# pre-registrations below were actually tested. The search is now two-sided
# (brackets the target, then bisects in log space, returns the fit CLOSEST to
# the target rather than the sparsest tried, and flags a miss on either side);
# regression test test_target_l0_search_backs_off_when_the_start_is_already_
# too_strong. --l1-coeff now starts at the calibrated 0.1, on the dense side,
# so the search approaches the target from the direction the old code could
# handle as well as the new one.
#
# COST. 7 layers x 2 checkpoints at target 64, plus 2 layers x 2 at target 32,
# each an l1 search of up to 8 SAE fits at 1500 epochs. 49702811 did 18 fits
# plus all the scoring in 25 min, so a fit is ~1 min at this size; 18 layer
# passes x up to 8 fits is ~2.5 h of fitting plus ~0.5 h of scoring and nulls.
# Estimate ~3-4 h against a 12 h limit, and the limit holds even if every
# layer uses all 8 tries.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

LEAKFREE="${SCRATCH}/spin_gnn/sudoku_n24_leakfree_2026-10-26/d0256_h04_L12_ff01024_lr1e-03_pre_shuf"
LEAKED="${OAK}/hshunt/checkpoints/spin_gnn/sudoku_dd_d0256_L12_ntrain80000_ep300_sharedpool0.9960_job48912548"
BASE="${SCRATCH}/spin_gnn/sudoku_sae_2026-10-26_leakfree_l1fix"
mkdir -p "$BASE"

# ~17.3 reasoning positions per instance at N=24, so 3000 instances is ~52000
# positions -- enough that a feature at the 1e-3 aliveness floor still fires
# ~52 times and the rolled null has resolution.
COMMON=(--expansion 16 --l1-coeff 0.1 --l1-max-tries 8 --sae-epochs 1500
        --n-instances 3000 --max-positions-per-instance 40
        --alive-rate 1e-3 --n-null-shifts 19 --seed 0)

for arm in leakfree:"$LEAKFREE" leaked:"$LEAKED"; do
    name="${arm%%:*}"; ckpt="${arm#*:}"
    echo
    echo "=== ${name}: all layers at target L0=64 (the measured true-predicate count) ==="
    python -m spin_gnn.sudoku_sae --checkpoint "$ckpt" \
        --layer 0 2 4 6 8 10 12 --target-l0 64 \
        "${COMMON[@]}" --out-dir "${BASE}/${name}_l0_64"

    echo
    echo "=== ${name}: sparsity robustness, layers 8 and 12 at the old target L0=32 ==="
    python -m spin_gnn.sudoku_sae --checkpoint "$ckpt" \
        --layer 8 12 --target-l0 32 \
        "${COMMON[@]}" --out-dir "${BASE}/${name}_l0_32"
done

echo
echo "=== summaries ==="
for d in "${BASE}"/*/; do echo "--- $d"; cat "${d}sae_summary.csv"; done
