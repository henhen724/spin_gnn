#!/bin/bash
#
# Does the predicate appear BEFORE the churn? One lambda=1 arm, seed 0, the
# identical recipe 52452867 ran, snapshotted at ten points and evaluated at every
# one against the same registered 1,000-case list.
#
# WHY THIS AND NOT SOMETHING ELSE. Stage PAIR has four open items. Three of them
# -- whether P4 is the raw conditional or the placebo-corrected excess, whether
# the 1.630 convergence rule is right when it also refuses one of stage Y's own
# bodies, and which change to `L_pair` to make -- all alter a REGISTERED quantity
# and are hshunt's to decide. The fourth, whether the churn (142 / 175 / 177
# emission flips per 1,000 against stage Y's 4) is necessary or merely
# concurrent, needs no decision from anyone: it re-runs an arm that already ran,
# with nothing changed, and looks at it more often.
#
# THE HYPOTHESIS IS REGISTERED IN THE SCRIPT'S DOCSTRING, BEFORE THE NUMBERS.
# Corrupt AUROC first clears 0.65 between step 250 and 750; flips there are
# 20-70, at most half the final 142; the placebo never leaves (0.40, 0.60); the
# guard rises monotonically with the flips. The falsifier is flips already above
# 100 at the crossing -- the churn paid up front, with no slack to find.
#
# NOTHING REGISTERED CHANGES AND NOTHING IS APPLIED. Even if the slack is there,
# stopping the arm early changes the stage's training length, which is hshunt's
# call like the other three. The script prints no verdict.
#
# SMOKE FIRST. TRAJ_SMOKE=1 runs the whole code path at wrong-on-purpose sizes in
# ~2 minutes, writes to its own `_smoke` directory, and gates the real run. A
# KeyError found there costs two minutes instead of a GPU hour -- 52456376's
# predecessor lost a run to exactly that shape of bug.
#
# TIME. 52452867 trained TWO arms and evaluated three models in 13:52. This
# trains ONE arm and evaluates ten snapshots, with the expensive CPU case-list
# build done once, so ~15 min; 60 is the margin.
#
#SBATCH --job-name=stagepairTraj
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

echo "host=$(hostname)  job=$SLURM_JOB_ID"
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
echo "################ THE RUN  $(date '+%T') ################"
python3 scripts/stagepair_trajectory.py
echo "exit=$?"

echo
echo "end $(date '+%F %T')"
