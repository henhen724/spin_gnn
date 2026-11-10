#!/bin/bash
#SBATCH --job-name=stagenScore
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagenScore_%j.out
#
# Score stage N the moment its array closes.
#
#   submitted with --dependency=afterany:51771061
#
# WHY A DEPENDENCY AND NOT A POLL: the array runs ~5 h. Scheduler etiquette caps
# status checks at one a minute and forbids unattended polling loops; a
# dependency costs nothing and fires whether or not a session is awake.
#
# WHY afterany AND NOT afterok: the scorer reports per seed and is written to
# say "2 of 3 fresh cells landed" rather than to crash. afterok would mean one
# dead cell leaves the other two unscored.
#
# WHAT THIS JOB DELIBERATELY DOES NOT DO: no deck, no notebook, no commit, no
# upload, no STATUS.md edit. Every deck page is read back as a PNG before it
# ships and an unattended job cannot make that call. The scorer's output is the
# input to all of those decisions, not a substitute for them.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
    py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
    py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export MPLCONFIGDIR="${SCRATCH}/mplcache"
export PYTHONUNBUFFERED=1

ARRAY=51771061

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
echo "commit $(git rev-parse --short HEAD)"

echo "=== STEP 1/2  how the array exited ======================================"
sacct -j "$ARRAY" -X --format=JobID,JobName%18,State,ExitCode,Elapsed,MaxRSS
echo "step1 rc=$?"

echo "=== STEP 2/2  the registered score ======================================"
python3 -u -m scripts.stagen_score
echo "step2 rc=$?"

echo "end $(date '+%F %T')"
