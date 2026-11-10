#!/bin/bash
#SBATCH --job-name=stagefStatusFix
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagefStatusFix_%j.out
#
# Reflow STATUS.md from 99 lines to 31. Same content, one line per bullet.
#
# 75b382e installed a hard-wrapped version of the new handoff: readable, but 99
# lines against a 60-line cap, which is the one thing the standing brief says
# about this file. Wrapping at ~100 columns was the mistake -- the cap counts
# lines, so a bullet is one line. Nothing in the text changes except where the
# newlines fall, and FINDINGS.md (the verbatim archive) is untouched.
#
# From a compute node, same reason as every git step today: $HOME I/O on
# sh02-ln02 wedges intermittently in cl_sync_io_wait.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module load devel math viz python/3.12.1 2>/dev/null

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/3  reflow ===================================================="
python3 -u - << 'PY'
import io

P = "/home/users/hshunt/spin_gnn/STATUS.md"
NEW = "/scratch/users/hshunt/status_new2_1109f.md"

old = io.open(P, encoding="utf-8").read()
new = io.open(NEW, encoding="utf-8").read()
assert new.count("\n") < 60, "the replacement is not under 60 lines either"
assert old.startswith("# STATUS — spin_gnn, branch `sudoku-dd` (2026-11-09, after stage F)"), \
    "STATUS.md is not the file this job was written for"
if old.count("\n") < 60:
    print("already reflowed to %d lines; nothing to do" % old.count("\n"))
else:
    io.open(P, "w", encoding="utf-8").write(new)
    print("reflowed %d -> %d lines" % (old.count("\n"), new.count("\n")))
PY
echo "step1 rc=$?"

echo "=== STEP 2/3  commit ===================================================="
if [ -e .git/index.lock ]; then
    echo "SKIPPED: .git/index.lock exists. Nothing staged, nothing committed."
    ls -l .git/index.lock
else
    git add STATUS.md scripts/slurm/stagef_statusfix_2026-11-09.sbatch
    echo "add rc=$?"
    if git diff --cached --quiet; then
        echo "nothing staged; no commit made"
    else
        git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
            commit -q -m "STATUS.md: reflow the new handoff to 31 lines

75b382e installed it hard-wrapped at ~100 columns, which came to 99 lines
against the standing brief's 60-line cap -- the one requirement that file has.
The cap counts lines, so a bullet is one line. Identical text, different
newlines: no number, claim or ordering changes, and FINDINGS.md (the verbatim
archive of the old 107 KB file) is untouched.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
        echo "commit rc=$?"
    fi
fi
echo "step2 rc=$?"

echo "=== STEP 3/3  push ======================================================"
module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$?"
git log --oneline -1
echo "--- STATUS.md is $(wc -l < STATUS.md) lines, $(wc -c < STATUS.md) bytes ---"
git status --short | head -10

echo "end $(date '+%F %T')"
