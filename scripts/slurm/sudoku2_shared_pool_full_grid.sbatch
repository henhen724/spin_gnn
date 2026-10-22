#!/bin/bash
#SBATCH --job-name=sud2pool7
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out
#
# Score d0256 x n_train=80000 (48912548) on the shared pool, and finish the
# width x data grid on one authoritative per-instance matrix.
#
# HYPOTHESIS. 48912548's own-val agg_solve_rate came back 0.9933, up from
# 0.9733 at n_train=20000, which if taken at face value would be a new best.
# Two reasons not to take it at face value, and one reason to doubt the
# direction entirely:
#   - own-val instances are seeded at seed + n_train_instances, so the
#     80000-instance run validates on 300 instances no other run has seen, and
#     at p ~ 0.99 that set's SE is ~0.005; +0.020 is ~2 sigma at best;
#   - the same run's easy-end metrics went the WRONG way, hard: solve_rate_k0
#     0.60 -> 0.07 and mean_o1 0.918 -> 0.776. That is the same anti-correlation
#     width produced back on 10-18 (agg up, k0 collapsing), now reproduced by
#     data scale, and the shared pool spans rho 0.3-0.95, so it contains the
#     easy instances where this model has got worse.
# I predict the shared pool still puts it at or above the current best:
# solve_rate in [0.980, 0.995] with a POSITIVE paired delta against 0.9847.
# The falsifier is a negative delta, which would mean the easy-end collapse
# outweighs the full-range gain and that 0.9933 was a val-set artifact.
# Secondary pre-registration: next_node peak accuracy >= 0.96, at or above
# d0256 x 20000's 0.964, since finding 8 says next_node steps up with data.
#
# WHY SEVEN CHECKPOINTS AND NOT TWO. The pool is deterministic (1500 instances,
# seed_start 5e10, the same build_instances call), so the six already-scored
# models would get identical numbers -- but their per-instance vectors are
# currently split across two .npz files with different reference columns, so
# any pair that was never a reference cannot be tested without a re-run. One
# pass over all seven writes a single 7x1500 matrix from which EVERY pairwise
# McNemar is computable offline, forever, and it is also what makes this job
# last past the 10-minute aggregation guideline rather than being a 3-minute
# GPU grab. The six re-scores double as a reproducibility check: any deviation
# from the recorded 0.9847/0.9720/0.9733/0.8760/0.8200/0.7707 means something
# drifted and the new number is not comparable either.
#
# Grid being completed (shared-pool solve_rate, 1500 instances):
#                 n_train=5000   n_train=20000   n_train=80000
#   d0128            0.7707          0.9733            --
#   d0256            0.8200          0.9847         THIS RUN
#   d0384            0.8760          0.9720            --
#
# Runtime: 48912134 scored 4 checkpoints in 5:31, so 7 is ~10 min, and the
# ladder probed 5 checkpoints x 3 seeds in 5:50, so one checkpoint x 3 seeds is
# ~1.5 min. 1:00:00 covers a slow GPU with margin.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

S="${SCRATCH}/spin_gnn"
OUT_DIR="${S}/sudoku_shared_pool_grid_2026-10-22"
mkdir -p "$OUT_DIR"

TAG="d0256_h04_L12_ff01024_lr1e-03_pre"
NEW_CKPT="${S}/sudoku_hparam_sweep_2026-10-21_d256_data16x/${TAG}"

echo "=== 1/2 shared-pool paired eval, 7 checkpoints x 1500 instances ==="
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint \
        "${S}/sudoku_hparam_sweep_2026-10-21_d256_data4x/${TAG}" \
        "$NEW_CKPT" \
        "${S}/sudoku_hparam_sweep_2026-10-21_d384_data4x/d0384_h04_L12_ff01024_lr1e-03_pre" \
        "${S}/sudoku_hparam_sweep_2026-10-21_d128_data4x/d0128_h04_L12_ff01024_lr1e-03_pre" \
        "${S}/sudoku_hparam_sweep_2026-10-21_d384/d0384_h04_L12_ff01024_lr1e-03_pre" \
        "${S}/sudoku_hparam_sweep/${TAG}" \
        "${S}/sudoku_checkpoint_series_2026-10-19/epoch00300" \
    --n-instances 1500 \
    --out-csv "${OUT_DIR}/agg_paired_full_grid.csv"

echo
echo "=== 2/2 probes on the 80000-instance model (finding 8's next test) ==="
python -m spin_gnn.sudoku_probes \
    --checkpoint "$NEW_CKPT" \
    --n-train-instances 400 \
    --n-test-instances 160 \
    --max-positions-per-instance 20 \
    --probe-epochs 600 \
    --seed 0 1 2 \
    --out-csv "${OUT_DIR}/probes_d256_data16x.csv"
