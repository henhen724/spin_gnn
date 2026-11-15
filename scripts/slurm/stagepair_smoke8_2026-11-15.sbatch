#!/bin/bash
#
# Run stagepair_smoke.py for its new assertion 8, which measures the margin's
# gradient on the SHARED BODY at a live channel (|s| = 0.429, the magnitude
# 52430713 measured a real run reaching) rather than at det_scale = 0, where
# dg/dh = det_scale * E_v is identically zero and the margin cannot reach the
# body at all. Check 5's 0.729 was true and vacuous for that reason.
#
# ASSERTION 8 IS EXPECTED TO FAIL on the term as registered -- that is the
# point. 52423780 ran 45,000 steps at uniform output with the body's gradient
# at ~40x the control's, so a check that passes here would be another vacuous
# one. This job is for the NUMBER, which is what tells hshunt how far off the
# scale is and therefore which of the fixes is in range.
#
#SBATCH --job-name=stagepairSmoke8
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# No `set -e` on the python call: a nonzero exit is the expected outcome and
# the eight lines above the exit are the whole reason for the job.
python3 scripts/stagepair_smoke.py
echo "smoke exit=$?"

echo
echo "end $(date '+%F %T')"
