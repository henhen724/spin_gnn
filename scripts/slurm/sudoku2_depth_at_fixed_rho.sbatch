#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_depth
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH -o %x-%j.out

# Depth or density? Bin solve_rate by DD round count T at FIXED rho.
#
# Every solve_rate-vs-difficulty result in this project is confounded. In the
# removal-path ensemble a hint is removed only while DD can still recover the
# full coloring, so removing hints is *how* the deduction chain lengthens:
# rho and T move together by construction. The transition sweep (48581589)
# measured r(solve_rate, T) = -0.966 and r(solve_rate, rho) = +0.944 on the
# same instances, which is one fact reported twice, and it cannot say whether
# the model is limited by how much it must deduce or by how little it is told.
#
# The fix needs no new ensemble. At a single pinned rho, T still varies across
# instances because the graph and the removal order are random, so the SAME
# density can be split into depth bins. Checked on a login node before writing
# this (300 instances, no model):
#
#   rho=0.75: T=1 n=58, T=2 n=168, T=3 n=54, T=4 n=18, T=5 n=2
#             realized_rho = 0.7500 at EVERY T, exactly
#   rho=0.60: T=2 n=44 ... T=9 n=1
#             realized_rho 0.5843 (T=2) to 0.5833 (T>=3), spread 0.0010
#
# So depth spans roughly 1-6 with usable counts while density stays pinned to
# within 0.001 -- against a transition sweep that spans rho 0.3-0.95. That
# 0.001 is the whole validity argument, so eval_solve_by_depth reports
# realized_rho per bin, realized_rho_spread, and r(realized_rho, T), and the
# last of those must come out near zero or this measurement means nothing.
#
# Hypothesis: solve_rate falls with T at fixed rho. If it does, depth is doing
# real work and the -0.966 is not an artifact of density. If solve_rate is
# FLAT across T at fixed rho, then the apparent depth dependence was density
# all along, the model is limited by how many hints it can see rather than by
# how many deduction steps it must chain, and finding 3 (DD never appears)
# gains a mechanism: a model that never represents the frontier has no
# depth-dependent cost to pay, because it is not iterating.
#
# Pre-registering which way I expect: FLAT or nearly flat. Finding 3 says the
# model picks nodes by a sorted-emission convention and spends its capacity on
# colors, which is not an iterative procedure and should therefore not care
# how many DD rounds the instance would have needed. A clear negative slope
# would mean something depth-like survives that the probes did not see, and
# would be the more interesting of the two outcomes.
#
# Three densities so the depth gradient can be compared across them, and two
# checkpoints (the agg leader at 0.847 and the d0128 @900ep model all the
# interp work was done on, so the result connects to finding 3).
#
# rho=0.5 is deliberately close to the DD floor rho_min ~= 0.477; expect
# realized_rho to clamp above the request there, which the diagnostics will
# show rather than hide.
#
# Runtime: pool generation was timed at 6.1s/200 instances, so 1200 x 3 rhos
# x 2 checkpoints is ~12 min of single-threaded generation; the rest is
# autoregressive decoding of the kept bins. 3h is margin.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
# Slurm stdout is block-buffered, so progress lines otherwise stay invisible
# for the whole run and cannot be polled.
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_depth_at_fixed_rho_2026-10-21"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_depth_eval \
    --checkpoint \
        "${SCRATCH}/spin_gnn/sudoku_hparam_sweep/d0256_h04_L12_ff01024_lr1e-03_pre" \
        "${SCRATCH}/spin_gnn/sudoku_hparam_sweep_2026-10-19_ep900/d0128_h04_L12_ff01024_lr1e-03_pre" \
    --rho 0.50 0.60 0.75 \
    --n-pool 1200 \
    --min-per-bin 40 \
    --out-csv "${OUT_DIR}/depth_at_fixed_rho.csv"
