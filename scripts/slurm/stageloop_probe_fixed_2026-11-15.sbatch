#!/bin/bash
#
# STAGE LOOP, FORCED: what does a body that MUST read its own determinacy do?
# One array task per seed, `LOOP_SEED` = 0, 1, 2. Same script as finding 87,
# one new environment variable: `LOOP_FIX=0.10,0.30`.
#
# WHY. Finding 87 (job 52476255, HEAD 7e05fdd) came back a null of a specific
# kind. The feedback loop was present, differentiable at c = 0, free, and three
# independent trainings all left it SHUT: |c| = 0.0007 / 0.0006 / 0.0002, the
# feedback at 0.5% / 0.7% / 0.4% of the residual stream, and every other column
# on stage PAIR's published row inside the spread findings 84-86 measured. Adam
# moves a scalar by ~lr per step regardless of gradient size, so 3,000 steps
# could have carried c to 3.0; |c| at 1e-3 means the sign of dL/dc kept flipping.
# So 87 measured the OBJECTIVE'S APPETITE for the feedback, not the
# ARCHITECTURE'S CAPACITY with it. This arm measures the second one.
#
# HOW c IS SET, AND WHY IT IS NOT A GUESS. The delta is exactly linear in c, so
# one forward pass at c = 1 on one batch of this very body gives its size r1
# relative to h, and `c = target / r1` puts the feedback at exactly `target` of
# the residual stream at the warm start. `LOOP_FIX=0.10` therefore means "ten per
# cent of h, on this body, by measurement" -- not a number anyone picked. Then c
# is FROZEN (`requires_grad_(False)`), which returns the trainable count to
# exactly the registered 614,138; the script asserts that.
#
# THE ROWS, five on one case list: `Y` and stage PAIR's published `warm L1`
# reloaded (eval only, its published row READ from `pair_summary.json`),
# `fix0.10 L0` (lambda = 0, the no-margin control), `fix0.10 L1` and
# `fix0.30 L1`. Finding 87's learned-c arms are NOT re-run; this writes to
# `runs/stageloop_probe_2026-11-15[_s{N}]_fixed` and touches no prior artifact.
#
# THE PREDICTIONS ARE REGISTERED IN THE SCRIPT (the `LOOP_FIX` block above
# `FIX_TARGETS`) AND COMMITTED BEFORE THIS IS SUBMITTED, as 87's were: the body
# survives the mild setting (within 0.02 of this job's own Y validation token
# accuracy, and 0.30 within 0.05, where a bigger drop is reported as damage); the
# predicate still arrives (`fix0.10 L1` clears P1 and stays in P2's band on 3/3);
# the load-bearing one, now with the loop actually open (`fix0.10 L1`'s raw guard
# at least its own `pair L1` row on >= 2 of 3, at no more churn); and the reading
# I EXPECT TO WIN -- `fix0.10 L0`, which has no margin at all, still shows more
# flips than stage Y's 4 / 1 / 7, which would mean the feedback is a noise
# channel and 87's refusal was avoidance rather than indifference.
#
# WHAT WOULD OVERTURN THAT: a HIGHER guard at LOWER churn than `pair L1` on
# >= 2 of 3. That is what this whole line has been looking for and what 87's
# null did not deliver. Separately, train nll > 2.5 or val accuracy < 0.30 says
# the feedback destroyed the body and no AUROC in that row is readable.
#
# VALIDATION TOKEN ACCURACY IS PRINTED FIRST for every row, on the held-out
# stream (300,000,000, 400 instances) no arm trained on. A forced loop at 30% of
# h is the setting most likely to wreck a solved body, and that has to be visible
# before any AUROC is read as a statement about guards.
#
# NOTHING REGISTERED CHANGES AND NOTHING IS APPLIED. P4's denominator, the 1.630
# convergence rule, the form of L_pair, lambda, the margin and the 3,000-step
# length stay hshunt's call. NO VERDICT IS PRINTED.
#
# SMOKE FIRST, PER TASK, with the forced path on, so `calibrate_c` and the frozen
# parameter count are exercised at wrong-on-purpose sizes before a GPU hour.
#
# TIME. 87 did two trainings and four evals in 15:58 including both gates; this
# is three trainings and five evals, ~25 min. 60 min is margin and schedules fast.
#
#SBATCH --job-name=stageloopFixed
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

export LOOP_SEED="${SLURM_ARRAY_TASK_ID}"
export LOOP_FIX="0.10,0.30"
echo "host=$(hostname)  job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  LOOP_SEED=${LOOP_SEED}  LOOP_FIX=${LOOP_FIX}"
echo "start $(date '+%F %T')"
git rev-parse --short HEAD

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo
echo "################ INVARIANTS (cpu, seconds)  $(date '+%T') ################"
if ! python3 scripts/_loop_dry_check.py; then
  echo "DRY CHECK FAILED -- the feedback term is not inert at c=0 or not "
  echo "differentiable there. Nothing is trained."
  exit 1
fi

echo
echo "################ SMOKE (forced path, shapes only)  $(date '+%T') ################"
if ! LOOP_SMOKE=1 python3 scripts/stageloop_probe.py; then
  echo "SMOKE FAILED -- the real run is NOT started. Fix the code path first."
  exit 1
fi

echo
echo "################ THE RUN  seed ${LOOP_SEED}  c fixed at ${LOOP_FIX}  $(date '+%T') ################"
python3 scripts/stageloop_probe.py
echo "exit=$?"

echo
echo "end $(date '+%F %T')"
