#!/bin/bash
#SBATCH --job-name=stagekRecord3
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagekRecord3_%j.out
#
# Commit stage K's two actual artifacts -- the rendered deck and the executed
# notebook -- which neither 88b3150 nor f22c8b4 contains.
#
# WHY A THIRD GIT JOB: 88b3150's `git add` ran inside the chain that wedged at
# 14:02 and had staged only the deck SOURCE and the two write-up scripts, not the
# PDF or the .ipynb; f22c8b4 carried the files that changed after it. 51708483
# printed `git status --short` at the end precisely so this would be visible
# instead of assumed, and it was: three stage K paths still untracked.
#
# FROM A COMPUTE NODE for the same reason as 51708483: `git status` on sh02-ln02
# wedged again at 14:4x in cl_sync_io_wait. Reported to hshunt, not worked around.
#
# STATUS.md is patched HERE rather than beforehand so its note can name this
# job. If the patch fails its uniqueness assertion, STATUS.md is left exactly as
# committed in f22c8b4 and the `git add` below is simply a no-op for it, so a
# failed patch cannot produce a half-written handoff file.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module load devel math viz python/3.12.1 2>/dev/null

MSG=/scratch/users/hshunt/commit_msg_1109_kartifacts.txt

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/4  STATUS.md note ==========================================="
python3 -u /scratch/users/hshunt/patch_status_1109_gitnote.py "$SLURM_JOB_ID"
echo "step1 rc=$?"

echo "=== STEP 2/4  commit ==================================================="
if [ -e .git/index.lock ]; then
    echo "SKIPPED: .git/index.lock exists again. Nothing staged, nothing"
    echo "committed, and the lock is left alone."
    ls -l .git/index.lock
else
    git add notebooks/sudoku_n48_stagek_2026-11-09_slides.pdf \
        notebooks/sudoku_n48_stagek_2026-11-09_results.ipynb \
        scripts/slurm/stagek_record_2026-11-09.sbatch \
        scripts/slurm/stagek_record3_2026-11-09.sbatch \
        STATUS.md
    echo "add rc=$?"
    if git diff --cached --quiet; then
        echo "nothing staged -- already committed; no commit made"
    else
        git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
            commit -F "$MSG" -q
        echo "commit rc=$?"
    fi
fi
echo "step2 rc=$?"

echo "=== STEP 3/4  push ====================================================="
module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$?"
git log --oneline -3

echo "=== STEP 4/4  what is left untracked ==================================="
# Expected to remain: _smoke_dirablate.json (a stray smoke-test artifact, not
# mine to classify) and notebooks/sudoku_n96_stagef_2026-11-09_results.ipynb
# (stage F's notebook, regenerated and executed by 51708442 and committed with
# F's own write-up). Anything else here is a miss and should be read, not
# assumed.
git status --short | head -20
echo "STATUS.md is $(wc -l < STATUS.md) lines"

echo "end $(date '+%F %T')"
