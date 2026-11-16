#!/bin/bash
#
# THE 10% LOOP FINDING 88 NEVER ACTUALLY RAN. One array task per seed,
# `LOOP_SEED` = 0, 1, 2, `LOOP_FIX=0.10`, and one new variable:
# `LOOP_FIX_AT=first`.
#
# WHY. Finding 88 forced the determinacy loop to a "measured 10% of the residual
# stream" and the body was destroyed before a single gradient step (train nll
# 9.18 / 6.45 / 6.73 nats against log 58 = 4.060). Finding 89 (`52484098`, CPU,
# three forward passes per body, no training) measured what that calibration
# actually set, and the answer is that 88 never tested a 10% loop. `calibrate_c`
# read the ratio from `_fb_rel` after a forward at c = 1, which is the ratio at
# the LAST of the fifteen applications on a stream the fourteen earlier
# injections had already inflated: 0.0402 / 0.0785 / 0.0384. On the UNPERTURBED
# body the FIRST application's ratio is 4.66 / 3.94 / 4.90. Dividing the target
# by the small number gave c = 2.57 / 1.29 / 2.68, which puts the first injection
# at 1,197% / 508% / 1,315% of mean|h|. Five to thirteen times the stream it is
# added to is not a mild setting, and 88's step-0 nll is that, not a verdict on
# feedback.
#
# WHAT CHANGES, AND IT IS ONE KNOB. `LOOP_FIX_AT=first` reads r1 at the first
# application, where the hidden state is still the body's own, so
# `c = target / r1_0` means what the name says: c ~ 0.0215 / 0.0254 / 0.0204.
# Default stays `last`, so 88's path is byte-identical and its artifacts
# reproducible. The 0.30 row is dropped -- at a corrected calibration that is a
# 30%-per-application loop compounded fifteen times and 89 gives no reason to
# expect it survives, and one setting is what keeps this one knob. Everything
# else is stage PAIR's recipe unchanged: the bars, the 1,000 eval cases, lambda,
# the margin, the 3,000 steps, the held-out validation stream.
#
# THE ROWS, four on one case list: `Y` and stage PAIR's published `warm L1`
# reloaded (eval only, its row READ from `pair_summary.json`), `fix0.10 L0`
# (lambda = 0, the no-margin control) and `fix0.10 L1`. Output goes to
# `runs/stageloop_probe_2026-11-15[_s{N}]_fixed_first`, a NEW directory; no
# artifact of 87, 88 or 89 is touched.
#
# THE PREDICTIONS ARE REGISTERED IN `scripts/stageloop_probe.py` (the
# `LOOP_FIX_AT` block) AND COMMITTED BEFORE THIS IS SUBMITTED, as 84-88's were:
# the body SURVIVES (within 0.05 of this job's own Y validation token accuracy on
# 3/3, train nll below 2.0, against 88's 0.031 / 0.126 / 0.066 at nll
# 3.98 / 3.38 / 3.71); the predicate still arrives (P1 >= 0.65, P2 in band, which
# 88 could not even ask because every row of it was damage); the load-bearing one
# for the third and last time (`fix0.10 L1`'s raw guard at least its own
# `pair L1` row on >= 2 of 3, at no more churn); and THE READING I EXPECT TO WIN
# -- the body survives and the guard does NOT move, which would mean 87's |c| ~
# 1e-3 was indifference inside the safe range and avoidance outside it, and would
# close the stage-LOOP line for a sound reason instead of a broken calibration.
#
# WHAT WOULD OVERTURN IT: a survived body with a HIGHER guard at LOWER churn than
# `pair L1` on >= 2 of 3. That is what this whole line has been looking for.
# WHAT FALSIFIES THE RESCUE: still destroyed at a corrected 10% (val accuracy
# below 0.30 or train nll above 2.5), which says the loop is fatal at any usable
# size and 88 was right for the wrong reason -- reported that way, not as a
# confirmation.
#
# VALIDATION TOKEN ACCURACY IS PRINTED FIRST for every row, on the held-out
# stream (300,000,000, 400 instances) no arm trained on. It is the column that
# made 88 readable in one line and it never censors.
#
# NOTHING REGISTERED CHANGES AND NOTHING IS APPLIED. P4's denominator, the 1.630
# convergence rule, the form of L_pair, lambda, the margin and the 3,000-step
# length stay hshunt's call. NO VERDICT IS PRINTED.
#
# SMOKE FIRST, PER TASK, with the new calibration path on, so the hook and the
# frozen parameter count are exercised at wrong-on-purpose sizes before a GPU
# hour.
#
# TIME. 88 did three trainings and five evals in 24:44 including both gates; this
# is two and four, ~18 min. 45 is margin and still schedules fast.
#
#SBATCH --job-name=stageloopFirst
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

export LOOP_SEED="${SLURM_ARRAY_TASK_ID}"
export LOOP_FIX="0.10"
export LOOP_FIX_AT="first"
echo "host=$(hostname)  job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  LOOP_SEED=${LOOP_SEED}  LOOP_FIX=${LOOP_FIX}  LOOP_FIX_AT=${LOOP_FIX_AT}"
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
echo "################ SMOKE (first-application calibration)  $(date '+%T') ################"
if ! LOOP_SMOKE=1 python3 scripts/stageloop_probe.py; then
  echo "SMOKE FAILED -- the real run is NOT started. Fix the code path first."
  exit 1
fi

echo
echo "################ THE RUN  seed ${LOOP_SEED}  c fixed at ${LOOP_FIX} of h at the FIRST application  $(date '+%T') ################"
python3 scripts/stageloop_probe.py
echo "exit=$?"

echo
echo "end $(date '+%F %T')"
