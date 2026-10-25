#!/bin/bash
#SBATCH --job-name=ddctrlsweep
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
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
# Cost (measured on the login node with a warm JIT): a removal path is ~1 ms
# at N=24 and ~2.4 ms at N=192, so the whole removal branch across 20 cells
# and six sizes is ~10 min; the random branch is ~100x dearer per point
# (planted_regular_graph retries hard at small N) at ~55 min. ~65 min total,
# single core; 3:00:00 covers the weak-propagation cells, whose removal paths
# are longer than the (5,4) ones this was timed on. No GPU is needed.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_dd_control_param_2026-10-24"
mkdir -p "$OUT_DIR"

python -m scripts.dd_control_parameter_sweep \
    --degrees 3 4 5 6 8 10 \
    --colors 3 4 5 6 \
    --sizes-removal 24 48 96 192 384 996 \
    --sizes-random 24 96 384 \
    --n-seeds-removal 400 \
    --n-seeds-removal-large 200 \
    --large-n 192 \
    --n-instances-random 300 \
    --seed-start 4242 \
    --out-dir "$OUT_DIR"
