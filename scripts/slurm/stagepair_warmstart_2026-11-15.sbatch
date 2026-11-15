#!/bin/bash
#
# The warm-start probe STATUS asked for: can the pair margin be driven down by a
# body that ALREADY solves? Two arms, lambda = 0 and lambda = 1, both from the
# same stage-Y seed-0 checkpoint, both on 30,000 instances the stage-Y run never
# saw (data seed 777).
#
# NOT A STAGE AND NOT A RESULT. No bar is read, no checkpoint is written, and
# whichever way it comes out the three candidate design changes to the
# registered term stay hshunt's call. It is run alongside 52441848 rather than
# after it because the two ask different questions: that one asks WHY the
# from-scratch run fails, this one asks whether its target is reachable at all.
# No outcome of either makes the other redundant.
#
# WHY IT CANNOT BE SKIPPED. If the hinge will not leave 1.0 even from a model at
# val accuracy 0.4830 with a live channel, then the contrast is not
# representable by this cell and warming the weight in, lowering lambda, or
# restricting the margin's reach are all answers to the wrong question. That is
# a one-GPU-hour way to find out before spending 36 GPU-hours x 3 on a schedule.
#
# TIME. 3,000 steps at 151 ms (two forwards) plus 3,000 at 76 ms is about 12
# minutes, plus roughly 3 for generating and DD-solving 30,000 instances, plus
# the per-125-step instrumentation. 0:50 is requested against that ~18.
#
#SBATCH --job-name=stagepairWarm
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:50:00
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

# No `set -e`: if the lambda=1 arm dies the control's table is still the point.
python3 scripts/stagepair_warmstart.py
echo "warmstart exit=$?"

echo
echo "end $(date '+%F %T')"
