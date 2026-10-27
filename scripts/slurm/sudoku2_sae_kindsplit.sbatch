#!/bin/bash
#SBATCH --job-name=sud2saeKS
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#
# WERE THE LAYER-0 next_node FEATURES REAL, OR FREE?
#
# Job 49729375 answered the main question (finding 19: the leak-free model has
# 16 justifies_next features above null, best F1 0.486 at L8, against 2 and
# 0.162 for the leaked record) but left ONE verdict ambiguous. Its
# pre-registration (a) said the leak-free model would have NO next_node-family
# features at layer 0 while the leaked one still would. The counts came back
# 23 vs 24 -- essentially identical -- and I recorded it as FALSIFIED while
# noting it was probably UNTESTABLE as written. This job settles which.
#
# WHY IT MAY HAVE BEEN UNTESTABLE. The reasoning positions come in two kinds
# and the position index means something different in each:
#
#   pre_node   pos = i - 1   the model is about to emit the NEXT NODE, so
#                            tokens[pos] is the previous colour token. Nothing
#                            at the embedding knows which node comes next.
#   pre_color  pos = i       the model is about to emit that node's COLOUR, so
#                            tokens[pos] IS the node itself. next_node[v] is
#                            therefore matched FOR FREE by any token-identity
#                            feature, at layer 0, in BOTH arms, with no
#                            computation and no leak.
#
# Pooling the two kinds, as 49729375 did, mixes a free match into the count.
# If the layer-0 next_node features are the pre_color freebie in both arms,
# the 23-vs-24 count was never a test of anything.
#
# WHAT IS NEW: --split-by-kind re-scores each dictionary twice more, once on
# pre_node positions only and once on pre_color only, into
# sae_summary_by_kind.csv. The SAE fits, the predicate set, the aliveness
# floor and the rolled null are unchanged; `kind_is_node` was already in the
# predicate set, so the masks are the labels this pipeline already computes.
# Nothing in the training path is touched. Regression tests:
# test_position_mask_separates_a_free_predicate_from_a_computed_one plants the
# freebie and the computed version side by side and requires the masks to tell
# them apart (unmasked they carry the SAME predicate label, which is the bug);
# test_position_mask_rejects_a_mismatched_or_empty_mask covers the guards.
#
# THIS IS ALSO A REPRODUCTION CHECK. Same checkpoints, same seed, same layers,
# same target as 49729375's target-64 pass, and `position_mask=None` is the
# code path 49729375 ran, unchanged. sae_summary.csv must therefore come back
# IDENTICAL to 49729375's. If it does not, finding 19 is not reproducible and
# that supersedes everything below.
#
# PRE-REGISTERED (hypothesis first, as always):
#  (a) LEAK-FREE, LAYER 0, pre_node. ZERO next_node-family features above null,
#      and best_next_node_f1 at or below the median null F1. There is no
#      mechanism at the embedding for position to predict a node that has not
#      been chosen yet, once adj[v] is permuted.
#      Falsifier: one or more above null. That would mean something ordered is
#      still leaking through the permuted encoding, which would be a bug worth
#      more than this analysis.
#  (b) LEAK-FREE, LAYER 0, pre_color. MANY next_node features (order 10+),
#      at high F1. This is the freebie, and it must show up, or the mask is not
#      doing what I think it is doing. (a) and (b) together are the claim that
#      the layer-0 count was the token identity all along.
#  (c) THE VERDICT. If (a) and (b) hold for BOTH arms, 49729375's pre-reg (a)
#      was UNTESTABLE as written, not false, and I will record it that way. If
#      instead the leaked arm keeps next_node features at pre_node while the
#      leak-free arm does not, the original prediction was RIGHT and the pooled
#      count hid it -- the more interesting outcome, and the one I am NOT
#      predicting.
#  (d) EDGES SURVIVE THE SPLIT. At least one justifies_next feature clears null
#      at pre_node positions in the leak-free arm at L8 or L12, and the
#      leak-free arm's pre_node best_justifier_f1 exceeds the leaked arm's.
#      Falsifier, and the one that would hurt: all 16 of finding 19's edge
#      features sit at pre_color positions only. That would shrink the finding
#      from "it works out which neighbours constrain the node it is choosing"
#      to "it can look up the neighbours of a node it has already been told",
#      which is a much smaller claim and would have to be reported as such.
#  (e) Low confidence, stated so a surprise counts: best_frontier_f1 is HIGHER
#      at pre_node than at pre_color in the leak-free arm, since the frontier
#      is what selects the next node and at pre_color that node is already
#      fixed. Falsifier: equal or lower.
#
# CAVEATS THAT APPLY TO EVERY NUMBER IN sae_summary_by_kind.csv:
#  - Each mask keeps roughly half the positions, so a feature at the 1e-3
#    aliveness floor fires ~26 times instead of ~52 and the rolled null has
#    less resolution. Counts are comparable WITHIN a mask, not across masks of
#    different size, and the by-kind n_alive will be below the pooled n_alive
#    for that reason alone.
#  - The dictionaries are the pooled ones. A feature can only be scored on the
#    kinds of position it was trained to reconstruct; this does not ask what a
#    dictionary fitted to pre_node positions alone would look like.
#
# COST. 49729375 did 18 layer passes (7+2 layers x 2 arms) in 01:14:01, so a
# layer pass is ~4 min. This job is 14 passes plus 28 extra scorings on half
# the positions each: ~1 h of fitting plus ~0.5 h of scoring. Estimate ~1.5 h
# against a 4 h limit.

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
BASE="${SCRATCH}/spin_gnn/sudoku_sae_kindsplit_2026-10-27"
mkdir -p "$BASE"

# Identical to 49729375's COMMON, so sae_summary.csv is a reproduction check.
COMMON=(--expansion 16 --l1-coeff 0.1 --l1-max-tries 8 --sae-epochs 1500
        --n-instances 3000 --max-positions-per-instance 40
        --alive-rate 1e-3 --n-null-shifts 19 --seed 0)

for arm in leakfree:"$LEAKFREE" leaked:"$LEAKED"; do
    name="${arm%%:*}"; ckpt="${arm#*:}"
    echo
    echo "=== ${name}: all layers at target L0=64, split by position kind ==="
    python -m spin_gnn.sudoku_sae --checkpoint "$ckpt" \
        --layer 0 2 4 6 8 10 12 --target-l0 64 --split-by-kind \
        "${COMMON[@]}" --out-dir "${BASE}/${name}_l0_64"
done

echo
echo "=== pooled summaries (must match job 49729375's target-64 pass exactly) ==="
for d in "${BASE}"/*/; do echo "--- $d"; cat "${d}sae_summary.csv"; done

echo
echo "=== by-kind summaries (the new result) ==="
for d in "${BASE}"/*/; do echo "--- $d"; cat "${d}sae_summary_by_kind.csv"; done
