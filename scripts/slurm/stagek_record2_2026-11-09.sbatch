#!/bin/bash
#SBATCH --job-name=stagekRecord2
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagekRecord2_%j.out
#
# Commit what 88b3150 could not contain, and push stage K.
#
# 88b3150 is the commit whose `git commit` wedged at 14:02; it completed on its
# own at ~14:35, exit 0, with everything it had staged. Two files changed after
# its `git add` and so are not in it: LOG.md's stage K line (appended by job
# 51704309 at 14:25) and the STATUS.md patch. scripts/slurm/stagef_writeup is
# new. Everything else stage K produced is already in 88b3150.
#
# FROM A COMPUTE NODE because `git status` on sh02-ln02 wedged again at 14:4x in
# cl_sync_io_wait, after 88b3150 had already cleared -- the fault is intermittent,
# it is reported to the user, and it is not worked around here.
#
# Still refuses to touch .git/index.lock if one reappears: that lock belongs to
# whichever process holds it, and a second git writing an index behind a live one
# is how an index gets corrupted. 88b3150 is the evidence that waiting works.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

MSG=/scratch/users/hshunt/commit_msg_1109_ktail.txt

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/3  what 88b3150 already holds ==============================="
git show --stat --oneline 88b3150 | head -12

echo "=== STEP 2/3  commit the tail =========================================="
if [ -e .git/index.lock ]; then
    echo "SKIPPED: .git/index.lock exists again. Nothing staged, nothing"
    echo "committed, and the lock is left alone."
    ls -l .git/index.lock
else
    git add STATUS.md LOG.md scripts/slurm/stagef_writeup_2026-11-09.sbatch \
        scripts/slurm/stagek_record2_2026-11-09.sbatch
    echo "add rc=$?"
    if git diff --cached --quiet; then
        echo "nothing staged -- 88b3150 already holds all of it; no commit made"
    else
        git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
            commit -F "$MSG" -q
        echo "commit rc=$?"
    fi
    git log --oneline -2
fi

echo "=== STEP 3/3  push ====================================================="
module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$?"
git log --oneline -1
echo "--- still uncommitted after this job ---"
git status --short | head -20

echo "end $(date '+%F %T')"
