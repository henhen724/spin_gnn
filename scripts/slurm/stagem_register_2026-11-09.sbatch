#!/bin/bash
#SBATCH --job-name=stagemRegister
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagemRegister_%j.out
#
# Commit stage M's pre-registration BEFORE its first cell runs, and fail closed
# if the scorer does not run.
#
# The array (sudoku2_n96_cuthalf_M_2026-11-09.sbatch) is submitted with
# --dependency=afterok on THIS job, which is what makes "registered before the
# first cell lands" a mechanism rather than a claim: if the scorer errors or the
# commit fails, this job exits non-zero and not one GPU cell starts.
#
# STEP 1 runs scripts/stagem_score.py with only corners A and D on disk. That is
# the real dry run: every path through collect/summarize/show is exercised, the
# SCAN_DIRS assertion fires or does not, and the registered bars are printed, but
# M0-M3 report "nothing on disk yet" for the two corners the array will fill. A
# scorer that cannot run before the cells exist is a scorer that gets edited
# after seeing the numbers, which is the thing registration is for.
#
# From a compute node, like every git step today: $HOME/$SCRATCH I/O on sh02-ln02
# wedged four times on 11-09 in cl_sync_io_wait, including a git commit that held
# a 0-byte .git/index.lock for 33 minutes and then completed by itself, exit 0.
# The commit step refuses to run if a lock exists -- that lock belongs to
# whichever process holds it, and waiting is what works.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
    py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
    py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export MPLCONFIGDIR="${SCRATCH}/mplcache"
export PYTHONUNBUFFERED=1

MSG=/scratch/users/hshunt/commit_msg_1109_stagem.txt

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/3  the scorer runs (dry: only corners A and D exist) ========="
python3 -u -m scripts.stagem_score
rc=$?
echo "stagem_score rc=$rc"
if [ "$rc" -ne 0 ]; then
    echo "FAILING CLOSED: the scorer did not run, so nothing is committed and the"
    echo "dependent array will not start."
    exit "$rc"
fi

echo "=== STEP 2/3  commit the registration ==================================="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged, nothing"
    echo "committed, and the lock is left alone."
    ls -l .git/index.lock
    exit 1
fi
git add scripts/stagem_score.py \
    scripts/slurm/sudoku2_n96_cuthalf_M_2026-11-09.sbatch \
    scripts/slurm/stagem_register_2026-11-09.sbatch \
    scripts/slurm/stagem_yardstick_2026-11-09.sbatch \
    scripts/slurm/stagem_yardstick2_2026-11-09.sbatch
echo "add rc=$?"
if git diff --cached --quiet; then
    echo "nothing staged -- already committed; no commit made"
else
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -F "$MSG" -q
    rc=$?
    echo "commit rc=$rc"
    [ "$rc" -eq 0 ] || { echo "FAILING CLOSED: commit failed."; exit "$rc"; }
fi

echo "=== STEP 3/3  push ======================================================"
# A push failure must NOT block the science: the registration is already in the
# local history with its timestamp, which is what the dependency is protecting.
module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$?  (non-fatal by design)"
git log --oneline -1
git status --short | head -10

echo "end $(date '+%F %T')"
exit 0
