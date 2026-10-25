#!/bin/bash
#SBATCH --job-name=sud2easyend2
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#
# Replication of 49430217 stage 1 with the failure classifier fixed.
#
# WHY. 49430217 answered the headline question -- the easy-end collapse is an
# out-of-distribution artifact, 1.000 with 0/300 failures at every in-
# distribution hint count -- but it got the MECHANISM wrong in its own output.
# I pre-registered that >= 70% of the 24-hint failures would be `extra_rounds`
# (the model deducing on an instance with nothing to deduce); the CSV reported
# 280/282 `malformed`. The cause was the classifier, not the model:
# `classify_failure` tested `well_formed` first, and `decode` raises
# ParseError("empty BEGIN_ROUND") exactly when a round is opened and cannot be
# filled. Verified directly on a real N=24 rho=1.0 sequence: splicing one bare
# BEGIN_ROUND into the reasoning section gives
#     well_formed=False  error='empty BEGIN_ROUND at position 196'
# So the predicted signature was being charged to the wrong bucket. The fix
# splits `empty_round` out as its own label rather than reordering the tests,
# which would have discarded the distinction instead of recording it.
#
# What the parser check does NOT establish is that the model's 280 unparseable
# 24-hint failures ARE this error and not token soup. Only re-running with the
# fixed classifier can say, so this re-runs the identical configuration --
# same three checkpoints, same seven hint counts, same n_instances, same
# seed_start=6e10, greedy decoding -- and changes nothing but the labelling.
#
# PRE-REGISTERED PREDICTIONS.
#  (a) solve_rate reproduces 49430217 to the digit in all 21 bins. Decoding is
#      greedy and the instance seeds are fixed, so anything else means
#      nondeterminism I have not accounted for and the re-run, not the first
#      run, is what to distrust.
#  (b) >= 90% of d256_data16x's 282 failures at 24 hints land in `empty_round`,
#      i.e. almost all of the old `malformed` count was this one error.
#      Falsifier: a `malformed` remainder above 10%, which would mean the model
#      also emits genuinely unparseable output off-distribution and the
#      "opened a round it could not fill" story is only part of it.
#  (c) At 23 hints the failures stay `wrong_colors` (98% before, where the true
#      trace has exactly one round): the empty-round failure is specific to
#      the zero-round sequence shape, not to being off-distribution generally.
#  (d) d0384 keeps its 100% `extra_rounds` at 24 hints -- it fills the round it
#      opens, which is why it parses at all. This is the control that shows the
#      label split is not just relabelling every OOD failure.
#
# Cost: stage 1 of 49430217 was ~18 min for exactly this work. 2:00:00 is
# generous; the pool6000 re-score is NOT repeated, it is already written.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

SWEEP="${SCRATCH}/spin_gnn"
OUT_DIR="${SCRATCH}/spin_gnn/sudoku_easy_end_2026-10-24_relabelled"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_easy_end_diag \
    --checkpoint \
        "${SWEEP}/sudoku_hparam_sweep_2026-10-21_d256_data16x/d0256_h04_L12_ff01024_lr1e-03_pre" \
        "${SWEEP}/sudoku_hparam_sweep_2026-10-21_d256_data4x/d0256_h04_L12_ff01024_lr1e-03_pre" \
        "${SWEEP}/sudoku_hparam_sweep_2026-10-21_d384/d0384_h04_L12_ff01024_lr1e-03_pre" \
    --hint-counts 24 23 22 21 20 18 16 \
    --n-instances 300 \
    --seed-start 60000000000 \
    --out-csv "${OUT_DIR}/easy_end_by_hint_count_relabelled.csv"
