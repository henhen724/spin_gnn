#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_hdt
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# THE headline measurement: does the transformer's solve rate collapse where
# DD's reasoning depth diverges?
#
# Hypothesis: solve_rate falls steeply as realized_rho approaches the
# removal-path floor rho_min ~ 0.46 (d=5, q=4, N=24), and the fall tracks
# dd_mean_T (serial deduction depth) rather than rho itself. DD solves every
# instance in this ensemble by construction (dd_resolved_frac == 1), so any
# drop is the model failing to execute a longer deduction chain -- not the
# instance becoming unsolvable.
#
# Null to beat: the drop is explained by sequence length alone, since deeper
# traces are longer. truncated_rate is reported per bin to make that visible;
# if it rises in lockstep with the solve_rate drop, the result is about the
# token budget and not about reasoning.
#
# Why this ensemble and not random hints: scripts/dd_random_hint_transition.py
# (Slurm 48579017) showed independent random hints are subcritical at EVERY
# density for d=5/q=4, with mean recovery identical from N=24 to N=384. There
# is no transition there to find. The removal-path ensemble -- the one these
# checkpoints trained on -- has the real one, as a divergence of deduction
# time at rho_min ~ 0.4777 (N-independent across a 50x range).
#
# Eval-only on existing checkpoints; no retraining. Instance pool seed
# 8e8 is disjoint from training, probes (5e8/9e8) and the series pool (7e8).

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

EP900="${SCRATCH}/spin_gnn/sudoku_hparam_sweep_2026-10-19_ep900"
OUT_DIR="${SCRATCH}/spin_gnn/sudoku_hint_density_transition_2026-10-19"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_hint_density_transition \
    --checkpoint \
        "${EP900}/d0128_h04_L12_ff01024_lr1e-03_pre" \
        "${EP900}/d0064_h04_L12_ff01024_lr1e-03_pre" \
    --n-per-bin 150 \
    --rho-min 0.44 --rho-max 1.0 --rho-step 0.02 \
    --out-csv "${OUT_DIR}/transition.csv"
