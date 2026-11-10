#!/bin/bash
#SBATCH --job-name=stagenRegister
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagenRegister_%j.out
#
# Register stage N BEFORE its first cell lands.
#
# The array sudoku2_n48_kbrep_N_2026-11-10.sbatch is submitted
#   --dependency=afterok:<this job>
# so "registered first" is a mechanism rather than a promise: if the scorer does
# not run clean, this job fails, and not one GPU cell starts.
#
# EVERY STEP FAILS CLOSED.
#   1. Run scripts/stagen_score.py dry. It must exit 0 against the data that
#      exists today (stage K's three seeds, stage N's dir absent). A scorer that
#      crashes is a registration that was never read.
#   2. Refuse to commit if .git/index.lock exists. $HOME I/O on sh02-ln02 wedges
#      intermittently in cl_sync_io_wait; waiting is what works, and the lock
#      belongs to whichever process holds it.
#   3. Commit. Push failure is explicitly NOT fatal -- the commit is the
#      registration, the push is the backup.
#
# From a compute node, like every git step on this project, for the same reason.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
    py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
    py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export MPLCONFIGDIR="${SCRATCH}/mplcache"
export PYTHONUNBUFFERED=1

MSG=/scratch/users/hshunt/commit_msg_1110_stagen.txt

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/3  the scorer must run clean ================================="
python3 -u -m scripts.stagen_score
RC=$?
echo "scorer rc=$RC"
if [ "$RC" -ne 0 ]; then
    echo "FAILING CLOSED: the registered scorer does not run. No commit, and the"
    echo "array's afterok dependency will never be satisfied, so no cell starts."
    exit "$RC"
fi

echo "=== STEP 2/3  commit ===================================================="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged, nothing"
    echo "committed, and the lock is left alone."
    ls -l .git/index.lock
    exit 1
fi

git add scripts/stagen_score.py \
    scripts/slurm/sudoku2_n48_kbrep_N_2026-11-10.sbatch \
    scripts/slurm/stagen_register_2026-11-10.sbatch
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
