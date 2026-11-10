#!/bin/bash
#SBATCH --job-name=stagenWriteup
#SBATCH --partition=sganguli
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagenWriteup_%j.out
#
# Stage N write-up, run as a job rather than on a login node, for the reason
# stagek_writeup recorded: this is sustained small-file I/O across every N=48
# run directory the programme has produced (nine sufficiency dirs plus E/G/H/I/J
# and K/N) with a 2000-draw bootstrap per cell, plus two subprocesses that
# re-read the same tree. On sh02-ln02 that wedged in cl_sync_io_wait once
# already; under Sherlock rule 1 it belongs in a job regardless.
#
# ORDER MATTERS HERE. The builders are committed FIRST, so that steps 2-5 can
# stamp the pages and the notebook with the sha of the code that drew them. A
# deck that prints "commit unknown" cannot be checked against anything.
#
# The generated artifacts -- slides.pdf and the executed notebook -- are NOT
# committed here. They are read back first (every PNG, and the notebook's
# output) and committed by a second job, the way stage K's record jobs did it.
#
# Still refuses to touch .git/index.lock if one exists: the lock belongs to
# whichever process holds it, and a second git writing an index behind a live
# one is how an index gets corrupted. Waiting is what works.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 \
    py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312 \
    py-matplotlib/3.8.3_py312 2>/dev/null

export NUMBA_CACHE_DIR=$SCRATCH/numba_cache
export MPLCONFIGDIR=$SCRATCH/mplcache

MSG=/scratch/users/hshunt/commit_msg_1110_stagen_deckfix.txt
NB=notebooks/sudoku_n48_stagen_2026-11-10_results.ipynb

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/5  commit the builders ======================================"
if [ -e .git/index.lock ]; then
    echo "SKIPPED: .git/index.lock exists. Nothing staged, nothing committed,"
    echo "and the lock is left alone. The deck below will be stamped with the"
    echo "PREVIOUS commit, which is wrong, so this job's output is then void."
    ls -l .git/index.lock
else
    git add notebooks/make_stagen_deck.py notebooks/build_stagen_nb.py \
        scripts/slurm/stagen_writeup_2026-11-10.sbatch
    echo "add rc=$?"
    if git diff --cached --quiet; then
        echo "nothing staged -- already committed; no commit made"
    else
        git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
            commit -F "$MSG" -q
        echo "commit rc=$?"
    fi
    git log --oneline -2
fi

COMMIT=$(git rev-parse --short HEAD)
echo "stamping deck and notebook with commit=$COMMIT"

echo "=== STEP 2/5  the registered scorer, again on this node ================="
python3 -u -m scripts.stagen_score
echo "step2 rc=$?"

echo "=== STEP 3/5  deck ====================================================="
python3 -u -m notebooks.make_stagen_deck --commit "$COMMIT"
echo "step3 rc=$?"

echo "=== STEP 4/5  notebook source =========================================="
python3 -u notebooks/build_stagen_nb.py
echo "step4 rc=$?"

echo "=== STEP 5/5  execute notebook ========================================="
python3 -u -m notebooks.exec_nb "$NB"
echo "step5 rc=$?"

echo "--- artifacts ---"
ls -l notebooks/stagen_2026-11-10/ | head -12
ls -l "$NB"
echo "--- uncommitted after this job ---"
git status --short | head -20

echo "end $(date '+%F %T')"
