#!/bin/bash
#SBATCH --job-name=ddctrlsweep
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#
# Is DD's critical point reachable at an N we can afford to train on?
#
# WHERE THIS COMES FROM. Two separate things were being conflated, and both
# are now stated separately:
#   * The RANDOM-HINT ensemble has no finite-density solvability transition.
#     Its 50% crossing slides toward rho=1 as N grows (0.930 at N=24 ->
#     0.987 at N=384) while the number of UNHINTED nodes at the crossing
#     stays O(1) (1.7, 2.0, 1.9, 2.7, 5.0). The threshold runs off the edge.
#   * The REMOVAL-PATH ensemble -- the one the transformers train on -- is
#     DD-solvable by construction, so it has no solvability transition
#     either. What it has is a floor on reachable density, rho_min ~ 0.4777,
#     N-independent, with deduction depth T growing as rho -> rho_min+.
# STATUS finding 1 reads as if DD itself transitions at N=24; it does not.
# dd_resolved_frac is identically 1.0 in that sweep. The 0.99 -> 0.44 curve
# is the TRANSFORMER degrading against a DD baseline pinned at 1.0.
#
# So the only candidate critical point in the training ensemble is the depth
# divergence at rho_min, and at N=24 its whole dynamic range is T <= 7.2
# (vs 14.9 at N=96, 28.1 at N=996, 46 at N=49992). Raising N is the obvious
# fix and costs ~8x per instance at N=48 and ~59x at N=96 (sequences are
# ~11*N tokens, attention is quadratic): the 5.5-GPU-hour best recipe becomes
# ~2 GPU-days at N=48 and ~2 GPU-weeks at N=96. This CPU-only sweep asks
# whether a different (degree, n_colors) buys the same divergence cheaply,
# and -- just as important -- whether rho_min is a critical point at all or
# merely where greedy removal happens to stop.
#
# PRE-REGISTERED PREDICTIONS (full statements in the module docstring).
#  (a) rho_min rises monotonically as the propagation ratio d/(q-1) falls;
#      specifically rho_min(d=3,q=4) > 0.477 > rho_min(d=10,q=4).
#  (b) At N=96, some cell with d/(q-1) <= 1.2 reaches max mean T >= 22, i.e.
#      >= 1.5x the 14.9 that (5,4) gives there.
#  (c) Falsifier for the cheap-corner idea: rho_min and max T flat in (d,q)
#      at fixed N, within +-20% of (5,4) everywhere. Then there is no cheap
#      corner and the program really does need large N.
#  (d) Finite-size scaling, the test of criticality itself: for (5,4),
#      T_at_rho55 should grow as a power of N while T_at_rho90 stays roughly
#      N-independent. If T at rho=0.90 grows just as fast, nothing is
#      critical -- bigger graphs simply contain longer chains.
#  (e) Control: unhinted_at_rho50 stays O(1) in EVERY random-hint cell, so
#      the "threshold runs off to rho=1" result is not special to (5,4).
#
# SUPERSEDES 49445119, which died after 41 min on the SAME exception as the
# run it replaced, in the other branch:
#   ValueError: need q | N and q(q-1) | d*N (N=24, d=4, q=5)
#   ... dd_removal_path_small_n.py line 38, planted_regular_graph
# My previous fix guarded only the random branch, on the strength of a
# docstring claim of mine that the removal branch does not use that
# constructor. It does, at dd_removal_path_small_n.py:38. The claim was wrong
# when I wrote it and I acted on it without checking. Three changes:
#   * random_is_constructible() now gates BOTH branches.
#   * check_grid() enumerates every (cell, size, branch) point the constructor
#     would reject and prints the list BEFORE any measurement starts. Both
#     failures were fully decidable from the arguments alone; this converts
#     that whole class into a printed line.
#   * --sizes-removal becomes 60 120 240 480 960 and --sizes-random 60 120 360.
#     Every multiple of 60 satisfies both divisibility conditions for every
#     (d, q) here, so all 21 cells are measurable on one common size list for
#     the first time. Verified by dry run: 168 of 168 points constructible,
#     none unaffordable, 5 removal sizes per cell.
# This is the third attempt at this sweep, so it is the last one: if it fails
# again, document and move on rather than resubmitting.
#
# The size change costs comparability with the N=24..996 numbers quoted in
# finding 1, which were measured at (5,4) on the old list. (5,4) is re-measured
# here on the new list, so the deck stays internally consistent and the two can
# be related through that cell rather than assumed equal.
#
# ALSO SUPERSEDES 49437796, which died after 50 min with
#   ValueError: need q | N and q(q-1) | d*N (N=24, d=4, q=5)
# raised by planted_regular_graph at the 13th of 21 cells and fatal under
# `set -e`. The constructor balances colour classes exactly and spreads each
# node's d edges evenly over the q-1 other classes, so it requires q | N; none
# of the random sizes 24/96/384 is divisible by 5, which makes EVERY q=5 cell
# unconstructible at those sizes. Three changes, no change to the science:
#   * random_is_constructible() skips such points with a printed reason,
#     separately from the random_is_affordable() cost heuristic -- those cells
#     are impossible at a given size, not merely expensive.
#   * --sizes-random is now 60 120 360 rather than 24 96 384. Chosen because
#     every one is divisible by 3, 4, 5 and 6 and satisfies q(q-1) | d*N for
#     every (d, q) on the grid, so all 21 cells become testable for prediction
#     (e) at 63 points, against 14 cells at 37 points before. The cost is the
#     shorter lever arm for the N-scaling, 6x rather than 16x, which is still
#     ample to separate "flat in N" from "proportional to N".
#   * write_all() now also runs before the random branch, not only at the end
#     of the cell. 49437796 still lost this cell's six completed removal rows
#     because they were in memory when the exception fired.
# The 12 cells 49437796 did complete are NOT overwritten: they stay in
# sudoku_dd_control_param_2026-10-24 and this run writes to ...-24b. They
# already decided predictions (a), (c) and (d); see
# notebooks/make_control_param_deck.py.
#
# ALSO SUPERSEDES 49431733, which I cancelled after 46 min. Two engineering faults,
# both now fixed; the science and the predictions above are unchanged.
#   * It wrote all three CSVs only after the last cell, so hitting the wall
#     would have thrown away every completed cell. Now written after each.
#   * My "~55 min for the random branch" estimate ignored how hard
#     planted_regular_graph retries when the degree approaches the number of
#     differently-coloured nodes available. Measured in that run: fill
#     d/((q-1)N/q) = 0.375 took 4 s, 0.500 took 49 s, and 0.625 (d=10 q=3
#     N=24) had not returned after 15 min. --max-fill 0.4 skips those cells,
#     7 of 63 random points, every one of them an N=24 cell whose 50%
#     crossing came back nan anyway because almost every node is hinted
#     there. The removal branch does not use that constructor and is
#     unaffected; 49431733 got through 36 of its 126 removal points in 46 min
#     only because the random branch was blocking between cells.
#
# Cost (measured on the login node with a warm JIT): a removal path is ~1 ms
# at N=24 and ~2.4 ms at N=192, so the whole removal branch across 21 cells
# and six sizes is ~10 min. The random branch, with the pathological cells
# skipped, should be the ~55 min originally estimated; 4:00:00 is now a
# margin rather than a bet, and partial results survive either way.
# No GPU is needed.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_dd_control_param_2026-10-25"
mkdir -p "$OUT_DIR"

python -m scripts.dd_control_parameter_sweep \
    --degrees 3 4 5 6 8 10 \
    --colors 3 4 5 6 \
    --sizes-removal 60 120 240 480 960 \
    --sizes-random 60 120 360 \
    --n-seeds-removal 400 \
    --n-seeds-removal-large 200 \
    --large-n 240 \
    --n-instances-random 300 \
    --seed-start 4242 \
    --out-dir "$OUT_DIR"
