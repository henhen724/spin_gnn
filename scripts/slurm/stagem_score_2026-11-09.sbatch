#!/bin/bash
#SBATCH --job-name=stagemScore
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagemScore_%j.out
#
# Score stage M the moment the array closes.
#
#   submitted with --dependency=afterany:51745334
#
# WHY A DEPENDENCY AND NOT A POLL. The array runs ~15 h. Scheduler etiquette caps
# status checks at one a minute and forbids unattended polling loops; a
# dependency costs nothing and fires whether or not a session is awake.
#
# WHY afterany AND NOT afterok. The scorer reports per corner: a cell that failed
# or was killed shows up as "nothing on disk yet" or as fewer than three seeds,
# and M2 is written to say so rather than to crash. afterok would mean one dead
# cell leaves the other five unscored.
#
# WHAT THIS JOB DELIBERATELY DOES NOT DO.
#   * No deck and no notebook. Stage F's builders (notebooks/make_stagef_deck.py,
#     notebooks/build_stagef_nb.py) are stage-F-shaped -- two arms, one bar -- and
#     stage M is a 2x2 with three bars. Their stage-M equivalents have to be
#     WRITTEN, and writing them against a number this job has not printed yet is
#     how a deck ends up contradicting its own verdict. They come after the score.
#   * No commit, no upload, nothing pushed. Every deck page is read back as a PNG
#     before it ships, and an unattended job cannot make that call.
#   * No STATUS.md edit. The scorer's output is the input to that decision.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
    py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
    py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export MPLCONFIGDIR="${SCRATCH}/mplcache"
export PYTHONUNBUFFERED=1

ARRAY=51745334

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
echo "commit $(git rev-parse --short HEAD)"

echo "=== STEP 1/2  how the array exited ======================================"
sacct -j "$ARRAY" -X --format=JobID,JobName%18,State,ExitCode,Elapsed,MaxRSS
echo "step1 rc=$?"

echo "=== STEP 2/2  the registered score ======================================"
python3 -u -m scripts.stagem_score
echo "step2 rc=$?"

echo "end $(date '+%F %T')"
