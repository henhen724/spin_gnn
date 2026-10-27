#!/bin/bash
#SBATCH --job-name=sud2n24evord
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=%x_%j.out
#
# DOES THE MODEL DEPEND ON THE ORDER THE DEDUCTIONS ARE WRITTEN IN?
#
# HYPOTHESIS, stated before the run. It depends on it a little and not much:
# solve_rate falls resolvably on the paired pool but stays above 0.75, i.e. the
# ascending-node-index order inside a round was a convenience the model used
# where it could, not the mechanism by which it produces a trace.
#
# WHY THIS RUN EXISTS. solve_with_dd_trace emits `sorted(...)` per round
# (sudoku_trace.py:83), so a round's deductions arrive in ascending node index
# for 100.0% of rounds, and shuffle_adj -- the finding-16 fix -- does not touch
# that, it permutes neighbour lists only. scripts/order_baseline_next_node.py
# then measured what that order is worth, with no model at all: the hand rule
# "smallest index above the previous one whose colour is still unknown" (the
# known set only, no frontier, no deduction) scores max-over-v F1 0.632 on
# `next_node` at pre_node positions, ABOVE the leak-free model's own best SAE
# feature at 0.545, and the frontier-aware rule is exact at 1.000. So "it knows
# which node comes next" is free in both arms and is retired as evidence. What
# that measurement CANNOT say is whether the model actually leans on the order,
# because no eval-time permutation can ask it: a model trained on sorted rounds
# and scored on permuted ones is simply out of distribution. Only a retrain can,
# which is what SudokuReasoningConfig.shuffle_events is for.
#
# ONE CHANGE FROM THE LEAK-FREE BASELINE 49659502 (finding 17, shared-pool
# 0.8540 [0.8352, 0.8710], final_val_nll 1.3087, val_accuracy 0.5086,
# n_params 9,548,834): --shuffle-events added. Everything else is held at that
# cell's values -- N=24, degree=5, n_colors=4, rho=0.3, rho_max=0.95, removal
# hints, --shuffle-adj, d_model=256, nhead=4, L=12, ff=1024, pre-norm, lr=1e-3
# constant, batch=32, 300 epochs, n_train=80000, n_val=300, seed=0.
#
# THE CONTROL IS SEMANTICS-PRESERVING, VERIFIED ON CPU BEFORE SUBMITTING
# (scripts/verify_shuffle_events.py, 300 instances): adj, the planted colouring
# and the hints are identical instance by instance, every round's membership is
# identical as a set, T is identical, the encoded sequence LENGTH is identical
# (so max_len and n_params cannot move), the False arm is still in canonical
# sorted order, and 73.6% of the 707 multi-event rounds actually come out
# reordered (61.5% of rounds have more than one event; a 2-event round is left
# sorted half the time by chance, hence not 100%). A DD round deduces every node
# whose neighbours already cover q-1 colours SIMULTANEOUSLY, so the order inside
# it is an arbitrary presentation choice -- which is why this is a strictly
# weaker intervention than shuffle_adj, which changed which tokens appear where.
#
# THE COMPARISON IS PAIRED, and that is worth being explicit about because it
# looks like it should not be. sudoku_agg_eval prompts with
# full_tokens[:reasoning_start] -- hints and graph only -- and scores free
# generation against the planted colouring through an order-insensitive
# check_solution. The one thing this flag changes, the order of the events
# inside a round of the TARGET trace, lands entirely in the discarded suffix.
# So the two cells get byte-identical prompts on the same 1500 instances and
# McNemar applies. shuffle_events is therefore deliberately NOT in
# INSTANCE_KEYS, unlike shuffle_adj which changes the prompt itself. (Every
# teacher-forced analysis -- SAE, probes, circuits, depth slots, attention -- is
# the other way round and must rebuild its sequences with the flag its
# checkpoint was trained with; all nine call sites now do.)
#
# PRE-REGISTERED:
#  (a) THE HEADLINE. solve_rate on the n=1500 pool falls resolvably against
#      0.8540 (McNemar p<0.05) but stays above 0.75. Two falsifiers, in opposite
#      directions, and both are more interesting than the prediction: NO
#      resolvable change means the ascending order was pure decoration, the
#      min_unknown shortcut was never being used, and the open problem closes
#      with the leak-free numbers intact; a fall below 0.50 means trace
#      generation was substantially DRIVEN by the index order, and every
#      leak-free score in STATUS.md is itself partly an order-decoding score.
#  (b) val_nll rises by about the irreducible floor and no more. Permuting a
#      round of k simultaneous deductions injects log(k!) nats no model can
#      predict; measured over evaluate()'s own denominator (every valid token,
#      graph section included) that is +0.0270 nats/token, i.e. 1.3087 ->
#      ~1.336, and the matching token-accuracy floor is -0.0163, i.e. 0.5086 ->
#      ~0.492. Falsifier: a rise materially above the floor, which would mean
#      the permutation cost predictive capacity elsewhere in the sequence (the
#      colours, the graph) and the flag is not the clean control claimed above.
#      NOTE both numbers are floors, not predictions of model quality -- token
#      metrics are NOT comparable across this flag, which is why (a) is the
#      headline and finding 4 is the reason to say so out loud.
#  (c) The depth cost survives. Finding 17 found solve_rate falling with DD
#      depth once the adjacency leak was closed (rho=0.60: 0.946 -> 0.827 over
#      T=2..5, r=-0.998; rho=0.75: 1.000 -> 0.890, r=-0.970). Prediction: r at
#      rho=0.60 stays below -0.8. Falsifier: r above -0.5, which would mean part
#      of finding 17's depth signal was deep instances having more rounds whose
#      sorted order is easier to continue, and the leak-free L*(T) series -- the
#      run this project has been building towards -- would need this flag on.
#  (d) The memorization gap does not shrink. Finding 17d measured +0.1000 at
#      this budget (0.9567 own-training vs 0.8567 held-out, p=2.6e-09).
#      Permuting removes a regularity that generalizes across instances, so at
#      fixed n_train there is more to memorize and less to generalize:
#      prediction >= +0.1000. Falsifier: it shrinks, meaning the within-round
#      order was itself part of what made memorization possible.
#
# THE CONFOUND, stated up front because (a) cannot be cleaned of it here. This
# cell's training distribution has strictly higher entropy at the same n_train,
# so any solve_rate cost is "lost a crutch" OR "got less usable supervision per
# instance", and this job cannot separate them. The separation needs a data arm
# (the same cell at a larger n_train), which is exactly what 49705064 is
# measuring for the sorted-order version -- so read (a) alongside it rather than
# alone, and if (a) lands anywhere near its falsifiers the data arm is the
# follow-up, not a reinterpretation.
#
# COST. 49659502 took ~5 h to train this cell and ~1 h for the same eval tail;
# this adds one checkpoint to the pool stage (2 x 1500 instead of 1 x 1500), so
# ~7 h expected against a 14 h limit. 64G/4 cores is what the 80000-instance
# N=24 build needs (sganguli caps memory at 32G per core).

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

BASE="${SCRATCH}/spin_gnn/sudoku_n24_leakfree_2026-10-26/d0256_h04_L12_ff01024_lr1e-03_pre_shuf"
OUT="${SCRATCH}/spin_gnn/sudoku_n24_eventorder_2026-10-27"
mkdir -p "$OUT"
CKPT="${OUT}/d0256_h04_L12_ff01024_lr1e-03_pre_shuf_ev"

echo "=== leak-free N=24 config + --shuffle-events (one change from 49659502) ==="
python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 24 --degree 5 --n-colors 4 \
    --shuffle-adj --shuffle-events \
    --d-model 256 --nhead 4 --num-layers 12 --dim-feedforward 1024 \
    --n-train-instances 80000 --n-val-instances 300 --batch-size 32 \
    --epochs 300 --n-per-bin 100 \
    --out-dir "$OUT"

echo
echo "=== does the depth cost survive without the order? (pre-reg c) ==="
# Same --n-pool 600 --min-per-bin 40 and the same generator stream as
# depth_n24_leakfree.csv, so the T bins come out identical and the two columns
# are comparable bin by bin.
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "$CKPT" \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT}/depth_n24_eventorder.csv"

echo
echo "=== the headline, paired against the 0.8540 baseline (pre-reg a) ==="
# The baseline goes first so vs_reference reads against finding 17.
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "$BASE" "$CKPT" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500.csv"

echo
echo "=== memorization gap + adjacency-order sensitivity (pre-reg d) ==="
# Part B re-scores the same graphs under all five adjacency orderings. Finding
# 17b found the leak-free model order-insensitive there (-0.0075 to +0.0075, all
# p>0.3); this asks whether a model trained without the EVENT order is any
# different on the ADJACENCY axis, which nothing has measured.
python -m spin_gnn.sudoku_memorization_eval \
    --checkpoint "$CKPT" \
    --n-memorization 600 \
    --n-ordering 400 \
    --out-csv "${OUT}/ordering_and_memorization.csv"
