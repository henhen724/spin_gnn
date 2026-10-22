#!/bin/bash
#SBATCH -J sud2probeladder
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=0:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# Does the frontier representation stay flat ACROSS models, or only within one?
#
# Finding 3 (48861799) says frontier_mask AUROC reaches 0.863 by epoch 5, at
# solve_rate 0.007, and is still 0.863 at epoch 900 -- flat while solve_rate
# grows ~100x. That was measured on ONE training run, and that run was
# data-starved: every interp result so far comes from n_train=5000, the value
# pinned for all 31 runs before 2026-10-21. The models that actually solve the
# task (agg 0.973 and 0.986) have never been probed, so "the frontier is
# represented but never develops" is currently a claim about a regime, not
# about the architecture.
#
# THE LADDER. Five checkpoints, same instance distribution, same probe
# hyperparameters, spanning agg 0.816 -> 0.986 -- and, usefully, agg and k0
# move almost independently across them, so the two can be told apart:
#   d0256 5000  300ep  agg 0.846  k0 0.24
#   d0256 5000  900ep  agg 0.816  k0 0.87
#   d0384 5000  300ep  agg 0.846  k0 0.99
#   d0256 20000 300ep  agg 0.973  k0 0.60
#   d0384 20000 300ep  agg 0.986  k0 0.91   <- current best
#
# Probe hyperparameters are IDENTICAL to 48861799 (400/160 instances, 20
# positions, 600 probe epochs, seeds 0 1 2) rather than larger. Tighter error
# bars would be worth having, but comparability with the 0.863 number is worth
# more: the whole question is whether these land on it.
#
# PRE-REGISTERED PREDICTION. If finding 3 describes the representation, all
# five come in at 0.863 +/- 0.02 with no trend against agg, and the d384
# models do not exceed the d256 ones. I predict that. The informative
# alternative is frontier AUROC rising with agg -- anything above ~0.92 at
# agg 0.986 would mean the flatness was a property of the data-starved regime
# and finding 3 needs rewriting, not just extending. Third possibility worth
# naming in advance: frontier tracks k0 rather than agg, which would say the
# representation is about the rho=1.0 end (nothing to deduce) and not about
# deduction at all.
# Secondary: next_color was the one target that tracked competence WITHIN a
# run; if it does not also track agg ACROSS models, that correlation was about
# training time, not about competence.
#
# Baselines are label-only and checkpoint-independent, but rerun here anyway:
# they are the line every AUROC above is read against, and if the data
# pipeline has drifted since 48861799 the baselines are where it shows.
#
# Runtime: 48861799 probed 16 snapshots x 3 seeds in 16:54, so 5 checkpoints is
# ~6 min, plus d384 being wider than the d128 series and ~2 min of baselines.
# 45 min is the margin; this is NOT another 2h request for a 51s job.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

S="${SCRATCH}/spin_gnn"
OUT_DIR="${S}/sudoku_probes_competence_ladder_2026-10-21"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_probes \
    --checkpoint \
        "${S}/sudoku_hparam_sweep/d0256_h04_L12_ff01024_lr1e-03_pre" \
        "${S}/sudoku_hparam_sweep_2026-10-21_d256_ep900/d0256_h04_L12_ff01024_lr1e-03_pre" \
        "${S}/sudoku_hparam_sweep_2026-10-21_d384/d0384_h04_L12_ff01024_lr1e-03_pre" \
        "${S}/sudoku_hparam_sweep_2026-10-21_d256_data4x/d0256_h04_L12_ff01024_lr1e-03_pre" \
        "${S}/sudoku_hparam_sweep_2026-10-21_d384_data4x/d0384_h04_L12_ff01024_lr1e-03_pre" \
    --n-train-instances 400 \
    --n-test-instances 160 \
    --max-positions-per-instance 20 \
    --probe-epochs 600 \
    --seed 0 1 2 \
    --out-csv "${OUT_DIR}/probes_competence_ladder.csv"

echo
echo "=== label-only frontier baselines (same metrics, same settings as 48861799) ==="
python scripts/frontier_from_known_baseline.py \
    --n-train 250 --n-test 150 --probe-epochs 300 \
    --out-json "${OUT_DIR}/frontier_baseline_auc.json"
