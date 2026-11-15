#!/bin/bash
#
# The switch-point ladder: how well must the body already solve before stage
# PAIR's margin is satisfiable? One lambda=0 trunk of 15,000 steps, forked at
# 2,500 / 7,500 / 15,000 into a copy that gets 1,500 steps at lambda=1, carrying
# the optimizer moments as a lambda ramp inside one run would.
#
# NOT A STAGE AND NOT A RESULT. No bar is read, no checkpoint is kept, and it
# registers no schedule -- it measures the number a schedule would otherwise be
# a guess about. The design decision on the registered term stays hshunt's.
#
# WHY IT IS THE NEXT JOB. 52441848 and 52441962 bracketed the answer and left
# the middle empty: the margin is unreachable from a fresh init (main 4.31, pins
# at 3.95 forever) and comfortably reachable from stage Y (main 1.61, hinge
# 1.10 -> 0.43 while main holds at 1.67). Every "warm the weight in" schedule is
# a guess about where in between the crossover sits, and 40 minutes on one GPU
# measures it instead. If even 15,000 lambda=0 steps is too early, the honest
# answer is that the margin needs a trained model and not a ramp, which is a
# different design and worth knowing before 108 GPU-hours go into the wrong one.
#
# TIME. Trunk 15,000 at about 76 ms is 19 min; three forks of 1,500 at about
# 151 ms is 11; generating and DD-solving 60,000 instances is about 6. 1:20 is
# requested against that ~36.
#
#SBATCH --job-name=stagepairSwitch
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:20:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
git rev-parse --short HEAD

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# No `set -e`: an early fork's table is the point even if a later one dies.
python3 scripts/stagepair_switchpoint.py
echo "switchpoint exit=$?"

echo
echo "end $(date '+%F %T')"
