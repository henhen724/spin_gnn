#!/bin/bash
#
# WHEN does the predicate arrive? Findings 84 and 85 could only BOUND it. Same
# script, same recipe, `TRAJ_GRID=fine` and `TRAJ_SEED` = 0, 1, 2 as an array.
#
# WHY. All three bodies clear the 0.65 predicate bar at step 125 of 3,000 -- and
# 125 is the FIRST TRAINED SNAPSHOT on the coarse grid, so every "arrives at 125"
# in 84 and 85 is an upper bound, not a measurement. Seed 2 is at 98% of its final
# AUROC there (0.8520 of 0.8650), which is also the body 85 used to falsify 84's
# quiet checkpoint: its raw guard at 125 reads 0.1056, just past P4's 0.10 bar,
# with 127 of its final 166 flips paid. If the real crossing is at step ~25, that
# falsification is a sampling artifact. This grid adds 10, 25, 50, 75, 100 and
# keeps 0, 125, 250 and 3,000, so both reproduction anchors and the coarse run's
# own step-125 row stay comparable.
#
# THE PREDICTIONS ARE REGISTERED IN THE SCRIPT, IN `PREDICTION_FINE` AND IN THE
# DOCSTRING, AND WERE COMMITTED BEFORE THIS WAS SUBMITTED. From a saturating fit
# to each seed's own step-0/125/3,000 AUROCs (tau 104 / 82 / 36): crossings in
# [50, 100] / [10, 25] / [10, 50]; the raw guard at that crossing BELOW 0.10 on
# all three, seed 2 in [0.02, 0.09]; flips there <= 20 / 30 / 90; the placebo in
# (0.40, 0.60) at 27 of 27 snapshots; and step 125 landing within 0.01 of the
# coarse corrupt AUROC and 0.03 of its guard. The load-bearing one is seed 2's
# guard; the falsifier is seed 2 reading >= 0.10 at EVERY snapshot past the
# predicate bar, which would make 85's 2-of-3 real rather than unresolved.
#
# NOTHING REGISTERED CHANGES AND NOTHING IS APPLIED. Dating an event is not
# stopping early: training length stays part of the registered stage, like the P4
# denominator, the 1.630 rule and the form of `L_pair`. No verdict is printed.
#
# NO DATA IS OVERWRITTEN. `TRAJ_GRID=fine` writes to
# `runs/stagepair_trajectory_2026-11-15[_s{N}]_fine/`; the coarse artifacts that
# findings 84 and 85 were read off are untouched, seed 0's included.
#
# SMOKE FIRST, PER TASK. `TRAJ_SMOKE=1` runs the whole path at wrong-on-purpose
# sizes in ~2 min into its own `_fine_smoke` directory and gates the real run.
#
# TIME. The coarse runs did a seed in 12:29-12:50 including the smoke, with one
# fewer snapshot to evaluate; 60 min is the margin.
#
#SBATCH --job-name=stagepairTrajFine
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

export TRAJ_SEED="${SLURM_ARRAY_TASK_ID}"
export TRAJ_GRID=fine
echo "host=$(hostname)  job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  TRAJ_SEED=${TRAJ_SEED}  TRAJ_GRID=${TRAJ_GRID}"
echo "start $(date '+%F %T')"
git rev-parse --short HEAD

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo
echo "################ SMOKE (shapes only)  $(date '+%T') ################"
if ! TRAJ_SMOKE=1 python3 scripts/stagepair_trajectory.py; then
  echo "SMOKE FAILED -- the real run is NOT started. Fix the code path first."
  exit 1
fi

echo
echo "################ THE RUN  seed ${TRAJ_SEED}  fine grid  $(date '+%T') ################"
python3 scripts/stagepair_trajectory.py
echo "exit=$?"

echo
echo "end $(date '+%F %T')"
