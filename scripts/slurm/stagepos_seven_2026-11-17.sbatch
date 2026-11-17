#!/bin/bash
#
# STAGE POS, JOB 1 OF 2: WHERE ALONG THE TRACE THE MODEL GOES WRONG.
# NOTHING IS TRAINED. Every model here is a checkpoint that already exists.
#
# THE QUESTION. hshunt's background fit over the 15-directory minarch scan has
# solve rate a power law in validation token accuracy at each fixed density,
# S ~ acc^k, R^2 ~ 0.92 over rho 0.50 to 0.75, with k = 2.0, 4.6, 19.9, 38, 69,
# 92 at rho = 0.95, 0.90, 0.80, 0.70, 0.60, 0.50. Against the number of blank
# cells n = 48(1 - realized_rho) that exponent scales as n^(1.84 +- 0.03).
# Independent per-step failure at a constant rate would make log(1/S) LINEAR in
# n. A per-step rate rising linearly with position makes it QUADRATIC. 1.84 is
# close to 2, so the hypothesis is that the per-step error rate grows with
# position in the trace -- a quantity nobody here has measured.
#
# WHAT RUNS. Measurement A (teacher-forced next-token accuracy at every
# reasoning position, indexed by token offset, by deduction index j, and by
# round, split by token type) and measurement B (greedy rollout as in
# solve_rate, then per-pair node validity, colour correctness, first-error
# index and the hazard P(first error at j | none before j)) on the SEVEN cells
# hshunt named: full L12/L6/L3/L2 d256, no_mlp L4 d256 as the mid-accuracy
# no_mlp cell, J2 no_mlp L16 d192, S1 tied_no_mlp T16 d320. Seed 0 throughout.
# B is the direct test; A says whether per-token error under a PERFECT prefix
# already rises with position, which is what separates intrinsic difficulty
# from the model corrupting its own prefix.
#
# THE EVAL IS THE RHO CURVES' OWN: the 14-point grid 0.95 .. 0.33, 400 per bin,
# rho_transition_eval's default seed start 60,000,000,000, fall and shuffle_adj
# read per checkpoint by cfg_from_checkpoint. All 80 runs share one instance
# distribution (checked in stagepos_cells), so this job BUILDS THE INSTANCES
# ONCE into runs/stagepos_2026-11-17/instances/ and job 2's array reuses them.
# That is the only reason job 2 depends on this one.
#
# full L=2 d256 s0 IS THE EIGHTH ROW. It has solve rate 0.000 at rho = 0.50 and
# so is NOT one of the 79 successful runs; hshunt named it by hand. It is
# measured, flagged in79=False in every CSV, and excluded from anything fitted
# against the 79. It is worth the run: it is the only cell that collapses ABOVE
# rho = 0.50, which is where a hazard curve should be steepest.
#
# THE PREDICTIONS ARE REGISTERED IN runs/stagepos_2026-11-17/REGISTERED.md AND
# IN scripts/stagepos_score.py's docstring, COMMITTED BEFORE THIS IS SUBMITTED:
# P1 hazard rises with j (falsified by a flat or falling hazard on the
# majority); P2 sum_j h(j) scales as n^[1.5, 2.3] (falsified outside
# [1.3, 2.5]); P3 teacher-forced accuracy also falls with position -- AND I
# EXPECT P3 TO MISS, because A flat with B rising would say the position
# dependence is entirely error ACCUMULATION, which is the more interesting
# reading and will be reported as such rather than as a failed prediction;
# P4 the round/position confound, reported either way; P5 the j/n confound.
#
# DRY CHECK FIRST, because four things here can be wrong without crashing: the
# position schema off by one token (every curve shifted), the 2n+T+1 identity,
# batched greedy diverging from generate() at a non-tie, and a walker that
# cannot score the ground-truth trace as perfect. All four are checked on CPU
# in seconds before the GPU is touched.
#
# TIME. The CPU smoke did 8 instances per bin in ~12 s on the deepest bin; on a
# GPU at batch 64 a bin is seconds and a run is a few minutes. 3 hours is far
# more margin than that needs and still schedules fast on sganguli.
#
# NOTHING IS APPLIED AND NO VERDICT IS PRINTED.
#
#SBATCH --job-name=stageposSeven
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

export POS_OUT="/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17"
export POS_RUNS="seven"
export POS_MEASURE="AB"
export POS_BATCH="64"

echo "host=$(hostname)  job=${SLURM_JOB_ID}  out=${POS_OUT}"
echo "start $(date '+%F %T')"
git rev-parse --short HEAD

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo
echo "################ POPULATION  $(date '+%T') ################"
if ! python3 scripts/stagepos_cells.py; then
  echo "POPULATION CHECK FAILED -- the successful-run count is not 79, or a"
  echo "checkpoint is missing, or the runs disagree on the instance"
  echo "distribution. Nothing is measured."
  exit 1
fi

echo
echo "################ INVARIANTS (cpu, seconds)  $(date '+%T') ################"
if ! python3 scripts/_stagepos_dry_check.py; then
  echo "DRY CHECK FAILED -- the position schema, the 2n+T+1 identity, the"
  echo "batched rollout or the pair walker is wrong. No GPU time is spent."
  exit 1
fi

echo
echo "################ SMOKE (2 runs, 2 densities, 8 instances)  $(date '+%T') ################"
if ! POS_SMOKE=1 python3 scripts/stagepos_score.py; then
  echo "SMOKE FAILED -- the real run is NOT started."
  exit 1
fi

echo
echo "################ THE SEVEN, A and B  $(date '+%T') ################"
python3 scripts/stagepos_score.py
echo "exit=$?"

echo
echo "end $(date '+%F %T')"
ls -la "${POS_OUT}"
