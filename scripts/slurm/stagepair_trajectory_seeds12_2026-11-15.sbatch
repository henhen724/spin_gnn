#!/bin/bash
#
# Does finding 84's shape hold on the other two bodies? Same script, same recipe,
# `TRAJ_SEED` = 1 and 2 as an array.
#
# WHY. 84 measured, on seed 0, that the predicate is in place at step 125 of
# 3,000 (corrupt AUROC 0.7480, 29 flips of the final 160) while the raw guard
# there is 0.0140 -- under P4's 0.10 bar. So the predicate is separable from the
# churn in time and the guard is not. That is a claim about stage PAIR read off
# ONE body, which is exactly the mistake finding 83 made and 82 had to correct.
# The two remaining bodies end with MORE churn (175 and 177 flips against 142),
# and seed 1's stage Y starts closer to the bar than any other (matched-pair
# corrupt 0.5870 against 0.4560 / 0.4640), so if any seed shows the guard and the
# predicate arriving together, it is seed 1.
#
# THE PREDICTIONS ARE REGISTERED IN THE SCRIPT, IN `PREDICTION[SEED]` AND IN THE
# DOCSTRING, AND WERE COMMITTED BEFORE THIS WAS SUBMITTED. Both seeds cross 0.65
# at step 125 or 250; flips at the crossing <= 61 (35% of that seed's published
# final); the raw guard at the crossing is BELOW 0.10 on both; the placebo stays
# in (0.40, 0.60) at all ten snapshots; Spearman(guard, flips) >= +0.7. The
# load-bearing falsifier is a guard already at or above 0.10 where the predicate
# arrives -- that would make 84 seed 0's property, not the stage's.
#
# NOTHING REGISTERED CHANGES AND NOTHING IS APPLIED. Even on 3/3 this does not
# license stopping the arm early: training length is part of the registered
# stage, like the P4 denominator, the 1.630 rule and the form of `L_pair`. The
# script prints no verdict.
#
# SMOKE FIRST, PER TASK. `TRAJ_SMOKE=1` runs the whole path at wrong-on-purpose
# sizes in ~2 min into its own `_s{N}_smoke` directory and gates the real run.
# The code path is the one 52461470 already ran, so this is cheap insurance on
# the two things that changed: the per-seed checkpoint path and the per-seed
# anchor/prediction tables.
#
# TIME. 52461470 did one seed in 12:50 including its smoke. 60 min is the margin.
#
#SBATCH --job-name=stagepairTraj12
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=1-2
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

export TRAJ_SEED="${SLURM_ARRAY_TASK_ID}"
echo "host=$(hostname)  job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  TRAJ_SEED=${TRAJ_SEED}"
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
echo "################ THE RUN  seed ${TRAJ_SEED}  $(date '+%T') ################"
python3 scripts/stagepair_trajectory.py
echo "exit=$?"

echo
echo "end $(date '+%F %T')"
