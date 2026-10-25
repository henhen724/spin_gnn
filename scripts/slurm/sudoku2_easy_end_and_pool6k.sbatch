#!/bin/bash
#SBATCH --job-name=sud2easyend
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#
# Is the "easy-end collapse" a competence regression or an eval artifact?
#
# HYPOTHESIS. It is an artifact, and `solve_rate_k0` is not an easiness probe.
# `eval_solve_by_rho` pins bin k at rho = (N-k)/N, and removal_path snapshots
# at int(floor(rho*N)) hints, so with rho_max=0.95 and N=24 the most heavily
# hinted instance any of these models was trained on has 22 hints -- exactly
# the k=2 bin. k=1 (23 hints) and k=0 (24 hints) are both OUTSIDE the training
# support, and k=0 is doubly strange: with every node hinted the true trace
# has ZERO deduction rounds, a sequence shape no training instance has.
# Consistent with that, in the existing sweep every model at n_train >= 20000
# scores exactly 1.00 at k=2,3,4 and only k0/k1 move (d256: k0 0.24 -> 0.60 ->
# 0.07, k1 1.00 -> 0.99 -> 0.81), and k0 is erratic across models that are
# otherwise ordered (d384 x 5000 scores 0.99 where d256 x 5000 scores 0.24).
# So `mean_o1` averages three in-distribution bins with two OOD ones.
#
# PRE-REGISTERED PREDICTIONS (stage 1).
#  (a) For d0256 x 80000, solve_rate >= 0.98 at EVERY in-distribution hint
#      count (22, 21, 20, 18, 16). Falsifier: any of those below 0.95, which
#      would mean easy instances really did get worse and the OOD account is
#      wrong.
#  (b) The cliff sits at the support edge, not before it: solve_rate < 0.30 at
#      24 hints and < 0.95 at 23 hints for that model, reproducing the
#      100-instance 0.07 / 0.81 to within +-0.10 at 300 instances.
#  (c) Mechanism: >= 70% of its 24-hint failures are classified `extra_rounds`
#      -- the model invents deduction rounds on an instance whose true trace
#      has none. (A smoke test on the d128 series already shows T_emitted=1.00
#      against T_true=0.00 in that bin.) Falsifier: failures dominated by
#      `wrong_colors`, which would mean it cannot even copy the hints.
#  (d) The cliff LOCATION is distribution-set, not model-set: d0384 x 5000,
#      whose k0 is 0.99, also scores >= 0.98 at every in-distribution bin. So
#      in-distribution behaviour is uniform across these models while OOD
#      behaviour is not.
#
# WHY STAGE 2. The metric of record is running out of resolution: 0.9960 on
# 1500 instances is 6 failures, so a genuinely better model can at best win
# ~6 discordant pairs and a perfect one gives exact p = 2^-6 = 0.016. The next
# data-scaling rung (d0256 x 320000, ~24 GPU-hours) is therefore not worth
# submitting against a pool that cannot resolve its plausible outcome. Because
# build_instances seeds instance i at seed_start+i, n=6000 at the same
# seed_start 5e10 is a strict SUPERSET of the current pool, so the existing
# numbers stay valid and only gain precision.
#
# PRE-REGISTERED PREDICTIONS (stage 2).
#  (e) d0256 x 80000 still beats d0256 x 20000 with a positive paired delta at
#      n=6000 (the 1500-instance test gave p = 0.0015; this is a replication
#      on 4x the instances, 4500 of them new).
#  (f) d0128 x 20000 and d0384 x 20000 stay statistically indistinguishable
#      (|delta| < 0.01 and McNemar p > 0.05) at 4x the resolution. Falsifier:
#      they separate, which would weaken "at this scale data buys more than
#      width" -- that claim currently rests on a tie at n=1500.
#
# Runtime: 48999377 did ~1150 generations/min, so stage 1 is 3 ckpts x 7 bins
# x 300 = 6300 (~6 min) and stage 2 is 4 ckpts x 6000 = 24000 (~21 min), plus
# ~2.5 min building 12300 instances on CPU. ~30 min of work in one allocation;
# 2:00:00 covers a slow GPU.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

S="${SCRATCH}/spin_gnn"
OUT_DIR="${S}/sudoku_easy_end_2026-10-24"
mkdir -p "$OUT_DIR"

TAG="d0256_h04_L12_ff01024_lr1e-03_pre"
CKPT_D256_16X="${S}/sudoku_hparam_sweep_2026-10-21_d256_data16x/${TAG}"
CKPT_D256_4X="${S}/sudoku_hparam_sweep_2026-10-21_d256_data4x/${TAG}"
CKPT_D384_1X="${S}/sudoku_hparam_sweep_2026-10-21_d384/d0384_h04_L12_ff01024_lr1e-03_pre"
CKPT_D384_4X="${S}/sudoku_hparam_sweep_2026-10-21_d384_data4x/d0384_h04_L12_ff01024_lr1e-03_pre"
CKPT_D128_4X="${S}/sudoku_hparam_sweep_2026-10-21_d128_data4x/d0128_h04_L12_ff01024_lr1e-03_pre"

echo "=== 1/2 solve_rate by hint count, across the training support edge ==="
python -m spin_gnn.sudoku_easy_end_diag \
    --checkpoint "$CKPT_D256_16X" "$CKPT_D256_4X" "$CKPT_D384_1X" \
    --hint-counts 24 23 22 21 20 18 16 \
    --n-instances 300 \
    --seed-start 60000000000 \
    --out-csv "${OUT_DIR}/easy_end_by_hint_count.csv"

echo
echo "=== 2/2 shared pool at 6000 instances (superset of the 1500 pool) ==="
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "$CKPT_D256_4X" "$CKPT_D256_16X" "$CKPT_D128_4X" "$CKPT_D384_4X" \
    --n-instances 6000 \
    --out-csv "${OUT_DIR}/agg_paired_pool6000.csv"
