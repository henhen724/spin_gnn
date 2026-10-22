#!/bin/bash
#SBATCH --job-name=sud2aggpair
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1:30:00
#SBATCH --output=%x_%j.out
#
# Rank the top four checkpoints on ONE shared held-out instance set, paired.
#
# WHY THIS EXISTS. Two defects in reading agg_solve_rate out of sweep.csv, both
# of which only started to matter today:
#
#  1. The instance sets are not the same across rows. train_sudoku_model builds
#     its validation instances at seed_start = seed + n_train_instances, so the
#     n_train=20000 runs validated on a different 300 instances than the
#     n_train=5000 runs. The protocol matches and both are unbiased draws, so
#     0.847 -> 0.973 is a real improvement -- but I described it to hshunt and
#     in STATUS as "the identical eval", and that was wrong. Worse, the data4x
#     runs TRAIN on seeds seed+5000..seed+19999, which contains the
#     5000-instance runs' validation seeds, so the two sets are not even
#     independent in the convenient direction.
#  2. 300 instances cannot resolve what is left. SE at p=0.98 over 300 is
#     ~0.008, so 0.987 vs 0.973 is z ~ 1.2 -- a coin flip. I already had to
#     report 48869017 as a tie for exactly this reason.
#
# The fix is a single pool of 1500 instances from a seed region (5e10) that is
# disjoint from every training and eval seed used so far, scored by all four
# models, and compared with McNemar's exact test on the discordant pairs. Near
# the ceiling only the instances two models disagree on carry information:
# two models at 0.98 may disagree on 40 of 1500, and those 40 are the evidence.
#
# PRE-REGISTERED PREDICTION, in order of confidence.
#  (a) Both data4x models beat both 5000-instance models decisively
#      (p < 1e-6). This is the 0.126 gap; it cannot be noise.
#  (b) d384_data4x vs d256_data4x stays a tie: |delta| < 0.02 and p > 0.05.
#      If the paired test instead resolves it, that is the eval upgrade paying
#      for itself and the leaderboard gets a new top entry on 1500 instances.
#  (c) The two 5000-instance models (d0256 and d0384, both agg 0.847) are
#      within noise of each other on agg despite d384's huge k0/o1 advantage.
#      A split here would mean the 0.847 agg tie was itself a resolution
#      artifact at n=300.
# Falsifiable: if (a) fails, the headline result of 2026-10-21 is wrong and
# STATUS has to be rewritten, not patched.
#
# NOTE this measures agg only (rho spread [0.3, 0.95], matching the sweep's
# agg_solve_rate). k0/mean_o1 are a different eval (pinned rho per bin) and are
# NOT compared here.
#
# Runtime: 48863464 did roughly 3600 unbatched generations plus pool
# construction in 7:52, so ~8/s; 4 x 1500 = 6000 generations, two of them at
# d384, plus ~50s building the 1500-instance DD pool, gives ~30 min. 1:30
# leaves room for d384 being slower per token than that average without the
# 140x over-request of 48868696.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_agg_paired_2026-10-21"
mkdir -p "$OUT_DIR"

S="${SCRATCH}/spin_gnn"
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint \
        "${S}/sudoku_hparam_sweep_2026-10-21_d384_data4x/d0384_h04_L12_ff01024_lr1e-03_pre" \
        "${S}/sudoku_hparam_sweep_2026-10-21_d256_data4x/d0256_h04_L12_ff01024_lr1e-03_pre" \
        "${S}/sudoku_hparam_sweep_2026-10-21_d384/d0384_h04_L12_ff01024_lr1e-03_pre" \
        "${S}/sudoku_hparam_sweep/d0256_h04_L12_ff01024_lr1e-03_pre" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${OUT_DIR}/agg_paired.csv"
