#!/bin/bash
#SBATCH --job-name=stagetScore
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagetScore_%j.out
#
# Score stage T with the scorer that was registered before the first cell ran
# (scripts/staget_score.py, committed by job 51904633). All 18 grids are on disk:
# 3 seeds x T in {4,8,12,16,20,24}, 14 bins x 400 instances each, fixed weights.
#
# In a job rather than on the login node because the scorer bootstraps every
# curve and the readings are the thing this stage exists to produce -- a reading
# that dies to a login-node limit halfway through is worse than one that waits.
#
# No git, no renders: this job only reads. Whatever it prints is the result.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo
echo "=== how the eval cells exited ========================================="
sacct -j 51904636,51905401 -X -o JobID%16,State,ExitCode,Elapsed 2>/dev/null

echo
echo "=== grids on disk ====================================================="
ls -1 /scratch/users/hshunt/spin_gnn/sudoku_n48_tiedT_T_2026-11-11/rho_curve_*.csv | wc -l

echo
echo "=== the registered scorer ============================================="
python3 -u -m scripts.staget_score
echo "scorer rc=$?"
echo "end $(date '+%F %T')"
