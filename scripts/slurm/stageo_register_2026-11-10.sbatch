#!/bin/bash
#SBATCH --job-name=stageoRegister
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stageoRegister_%j.out
#
# Register stage O BEFORE its first cell lands.
#
# The array sudoku2_n48_sizeshape_O_2026-11-10.sbatch is submitted
#   --dependency=afterok:<this job>
# so "registered first" is a mechanism rather than a promise: if the scorer does
# not run clean, this job fails, and not one GPU cell starts.
#
# EVERY STEP FAILS CLOSED.
#   1. Run scripts/stageo_score.py dry. It must exit 0 against the data that
#      exists today (SCAN_O absent, the four bracketing rungs on disk). It exits
#      non-zero if any bracket rung is missing or has drifted from the literals
#      in its own docstring -- the comparison this stage makes is measured
#      against that bracket, so an unverifiable bracket must stop the array.
#   2. Refuse to commit if .git/index.lock exists. $HOME I/O on sh02-ln02 wedges
#      intermittently in cl_sync_io_wait; waiting is what works, and the lock
#      belongs to whichever process holds it.
#   3. Commit. Push failure is explicitly NOT fatal -- the commit is the
#      registration, the push is the backup.
#
# From a compute node, like every git step on this project, for the same reason.
# The scorer is also sustained small-file I/O across every N=48 run directory
# with a 2000-draw bootstrap per cell, which is a job under Sherlock rule 1
# regardless of where the git step runs.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
    py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
    py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export MPLCONFIGDIR="${SCRATCH}/mplcache"
export PYTHONUNBUFFERED=1

MSG=/scratch/users/hshunt/commit_msg_1110_stageo.txt

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/3  the scorer must run clean ================================="
python3 -u -m scripts.stageo_score
RC=$?
echo "scorer rc=$RC"
if [ "$RC" -ne 0 ]; then
    echo "FAILING CLOSED: the registered scorer does not run, or the bracket it"
    echo "measures against is not on disk. No commit, and the array's afterok"
    echo "dependency will never be satisfied, so no cell starts."
    exit "$RC"
fi

echo "=== STEP 2/3  commit ===================================================="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged, nothing"
    echo "committed, and the lock is left alone."
    ls -l .git/index.lock
    exit 1
fi

git add scripts/stageo_score.py \
    scripts/slurm/sudoku2_n48_sizeshape_O_2026-11-10.sbatch \
    scripts/slurm/stageo_register_2026-11-10.sbatch
echo "add rc=$?"
if git diff --cached --quiet; then
    echo "nothing staged -- already committed; no commit made"
else
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -F "$MSG" -q
    RC=$?
    echo "commit rc=$RC"
    [ "$RC" -eq 0 ] || { echo "FAILING CLOSED: commit failed"; exit "$RC"; }
fi

echo "=== STEP 3/3  push (failure is NOT fatal) ==============================="
module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$? (non-fatal: the commit is the registration)"
git log --oneline -2

echo "end $(date '+%F %T')"
