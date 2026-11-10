#!/bin/bash
#SBATCH --job-name=stagemYardstick
#SBATCH --partition=sganguli
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagemYardstick_%j.out
#
# Re-print stage F's score so stage M's bar can be FROZEN FROM MEASURED NUMBERS
# rather than from a number remembered out of a summary.
#
# Stage F's registration froze s*_96 = 0.119 from three TWO-SEED N=96 rungs and
# said, in writing, that the re-estimate "is what a LATER stage should freeze its
# own bar from". Stage M is that later stage. This job exists to transcribe the
# re-estimate and its per-rung parts exactly -- 0.117 is the rounded figure that
# reached STATUS.md, and a bar is not frozen from a rounding.
#
# Read-only: it writes nothing into the repo and nothing into the run dirs. It is
# a job and not a login-node command because $HOME/$SCRATCH I/O on sh02-ln02
# wedged four times today in cl_sync_io_wait, and because 2000 bootstraps over
# ten cells is not login-node work.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
    py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
    py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export MPLCONFIGDIR="${SCRATCH}/mplcache"
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
echo "commit $(git rev-parse --short HEAD)"

python3 -u -m scripts.stagef_score
echo "stagef_score rc=$?"

echo "end $(date '+%F %T')"
