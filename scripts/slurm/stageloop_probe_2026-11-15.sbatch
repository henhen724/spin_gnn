#!/bin/bash
#
# STAGE LOOP: can a body have the predicate AND the guard WITHOUT the churn?
# One array task per seed, `LOOP_SEED` = 0, 1, 2.
#
# WHY NOW. Findings 82-86 closed the timing question as far as a grid can: the
# margin buys a real predicate on 3/3, it arrives at step 75/25/25 of 3,000, and
# a checkpoint with the predicate and no guard exists on every body. What stays
# open is the churn -- the guard stage PAIR does buy comes with 142/175/177 flips
# of 1,000 against stage Y's 4, and guard and flips rank together at +0.88. No
# further snapshot can separate them, because in stage Y and stage PAIR alike the
# gate is read off the FINAL hidden state and written to the emission: g can
# change what the network says, never what it computes (findings 68-74, 78).
#
# THE ONE KNOB. `determinacy_feedback=True` recomputes the gate after every
# application of the shared block and folds it back into the residual stream:
#   g_t = <h_t*s, E_v>;  a_t = log sigma(g_t) + log 2;
#   h_t <- h_t + c*(a_t @ E_nodes)/n_nodes   for t = 0..T-2, never after the last.
# Recipe, data stream, warm-start body, 1,000 eval cases and every bar: exactly
# `stagepair_pairprobe.py`'s. The only difference from the published `warm L1`
# is the flag.
#
# ONE NEW SCALAR, STARTING AT ZERO. 614,139 against 614,138. c = 0 makes the arm
# start AT the stage Y body (`scripts/_loop_dry_check.py` asserts bit-identical
# logits at c = 0 with a non-zero det_scale, and a non-zero dL/dc so the loop CAN
# open), so |c| at the end measures whether the objective wanted the feedback.
#
# FOUR ROWS, SAME JOB, SAME CASES: Y and stage PAIR's published warm L1 reloaded
# (eval only, and its published row is READ from `pair_summary.json`, never
# retyped), plus `loop L0` (lambda=0) and `loop L1` (lambda=1) trained here.
# Without loop L0 any movement could be the loop's presence rather than the
# margin's use of it -- the pairprobe's own design argument.
#
# THE PREDICTIONS ARE REGISTERED IN THE SCRIPT (`PREDICTION`) AND IN ITS
# DOCSTRING, AND ARE COMMITTED BEFORE THIS IS SUBMITTED: |c| ends in [0.05, 2.0];
# loop L1 still clears P1 and stays in P2's band; its raw guard is at least its
# own pair L1 row on >= 2 of 3 seeds (load-bearing); its flips are at most pair
# L1's on >= 2 of 3; loop L0's guard stays under 0.10; and validation token
# accuracy on both loop arms is within 0.02 of this job's own Y row.
#
# WHAT FALSIFIES IT: a loop L1 guard BELOW its pair L1 row on 2 of 3, which would
# say the reach was never the constraint. Separately, train nll > 2.5 or val
# accuracy < 0.30 would say the feedback destroys the body -- then the arm is
# broken and no AUROC in the table is readable.
#
# VALIDATION TOKEN ACCURACY IS PRINTED FIRST, on a held-out stream (300,000,000,
# 400 instances) no arm trained on. A feedback loop through sixteen iterations
# can wreck a solved body and that has to be visible before any AUROC.
#
# NOTHING REGISTERED CHANGES AND NOTHING IS APPLIED. P4's denominator, the 1.630
# convergence rule, the form of L_pair, lambda, the margin and the 3,000-step
# length stay hshunt's call. NO VERDICT IS PRINTED.
#
# NO DATA IS OVERWRITTEN. New directory, `runs/stageloop_probe_2026-11-15[_s{N}]`;
# every stage PAIR and trajectory artifact is read-only input here.
#
# SMOKE FIRST, PER TASK. `LOOP_SMOKE=1` runs the whole path at wrong-on-purpose
# sizes in ~2 min into its own `_smoke` directory and gates the real run.
#
# TIME. The pairprobe did two arms and three evals in 13:52; this adds a fourth
# row, a validation pass and ~15% per forward for the feedback. 60 min is margin.
#
#SBATCH --job-name=stageloopProbe
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
echo "host=$(hostname)  job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  LOOP_SEED=${LOOP_SEED}"
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
echo "################ SMOKE (shapes only)  $(date '+%T') ################"
if ! LOOP_SMOKE=1 python3 scripts/stageloop_probe.py; then
  echo "SMOKE FAILED -- the real run is NOT started. Fix the code path first."
  exit 1
fi

echo
echo "################ THE RUN  seed ${LOOP_SEED}  $(date '+%T') ################"
python3 scripts/stageloop_probe.py
echo "exit=$?"

echo
echo "end $(date '+%F %T')"
