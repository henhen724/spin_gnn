#!/bin/bash
#SBATCH --job-name=stagefWriteup
#SBATCH --partition=sganguli
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --dependency=afterany:51656823
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagefWriteup_%j.out
#
# Stage F write-up: score the N=96 minimum-architecture array and build its deck
# and notebook, as soon as 51656823 stops running.
#
# WHY A DEPENDENCY AND NOT A POLL: 51656823 trains to ~17:05 and then evaluates,
# so it closes somewhere around 18:00-18:45. Rule 10 prefers a dependency to
# polling and this is exactly that case. `afterany` on purpose, not `afterok`:
# the registered scorer was written before the first F cell had an eval on disk
# and reports `absent` / `not_scored` per cell, so a partial array is still worth
# scoring -- a cell that died is a result about the cell, and finding out costs
# one job either way.
#
# WHY A JOB AT ALL (rule 1): CPU-only small-file I/O across every N=96 run
# directory with a 2000-sample bootstrap, the same shape that wedged on a login
# node twice today (scripts.stagek_score at 13:1x and a git commit at 14:02, both
# parked at 00:00:00 of CPU in cl_sync_io_wait on sh02-ln02).
#
# WHAT THIS JOB DELIBERATELY DOES NOT DO: it does not upload the deck and it
# does not commit. Every page of a deck is read back as a PNG before it ships,
# and that read-back is a judgement an unattended job cannot make. This job
# leaves the scored verdicts in its .out file and the rendered pages on disk so
# the next session reads them and then ships.
#
# NOTHING HERE IS NEW CODE: scripts/stagef_score.py (6c66e3a),
# notebooks/make_stagef_deck.py (2b78a9c) and notebooks/build_stagef_nb.py
# (5537032) were all registered before the first F cell landed. The scorer runs
# as its own step so its verdicts are in the log even if the deck raises.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 \
    py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312 \
    py-matplotlib/3.8.3_py312 2>/dev/null

export NUMBA_CACHE_DIR=$SCRATCH/numba_cache
export MPLCONFIGDIR=$SCRATCH/mplcache

COMMIT=$(git rev-parse --short HEAD)
NB=notebooks/sudoku_n96_stagef_2026-11-09_results.ipynb

echo "host=$(hostname)  job=$SLURM_JOB_ID  commit=$COMMIT"
echo "start $(date '+%F %T')"

echo "=== STEP 0/4  what 51656823 did ========================================"
sacct -j 51656823 -o JobID,State,Elapsed,MaxRSS,ExitCode | head -20

echo "=== STEP 1/4  score ===================================================="
python3 -u -m scripts.stagef_score
echo "step1 rc=$?"

echo "=== STEP 2/4  deck ====================================================="
python3 -u -m notebooks.make_stagef_deck --commit "$COMMIT"
echo "step2 rc=$?"

echo "=== STEP 3/4  notebook source =========================================="
python3 -u notebooks/build_stagef_nb.py
echo "step3 rc=$?"

echo "=== STEP 4/4  execute notebook ========================================="
python3 -u -m notebooks.exec_nb "$NB"
echo "step4 rc=$?"

echo "end $(date '+%F %T')"
