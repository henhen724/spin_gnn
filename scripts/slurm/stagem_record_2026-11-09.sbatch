#!/bin/bash
#SBATCH --job-name=stagemRecord
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagemRecord_%j.out
#
# Install the stage-M handoff STATUS.md, append LOG.md's one line for this
# session, and commit both with the scorer's sbatch.
#
# From a compute node, like every git step today: $HOME I/O on sh02-ln02 wedges
# intermittently in cl_sync_io_wait. Every step is idempotent and asserts before
# it writes -- the install refuses unless STATUS.md is the file it was written
# against, the LOG append checks for its own text first, and the commit refuses
# to run if a .git/index.lock exists (waiting is what works; the lock belongs to
# whichever process holds it).

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module load devel math viz python/3.12.1 2>/dev/null

MSG=/scratch/users/hshunt/commit_msg_1109_stagem_record.txt
LOGLINE=/scratch/users/hshunt/stagem_logline_2026-11-09.txt

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/4  STATUS.md ================================================="
python3 -u - << 'PY'
import io

P = "/home/users/hshunt/spin_gnn/STATUS.md"
NEW = "/scratch/users/hshunt/status_new_1109m.md"
TITLE_OLD = "# STATUS — spin_gnn, branch `sudoku-dd` (2026-11-09, after stage F)"
TITLE_NEW = "# STATUS — spin_gnn, branch `sudoku-dd` (2026-11-09, stage M running)"

old = io.open(P, encoding="utf-8").read()
new = io.open(NEW, encoding="utf-8").read()
assert new.startswith(TITLE_NEW), "the replacement is not the file this job expects"
assert new.count("\n") < 60, "the replacement is not under 60 lines: %d" % new.count("\n")
if old.startswith(TITLE_NEW):
    print("STATUS.md is already the stage-M handoff; nothing to do")
else:
    assert old.startswith(TITLE_OLD), "STATUS.md is not the file this job was written for"
    io.open(P, "w", encoding="utf-8").write(new)
    print("installed: %d -> %d lines" % (old.count("\n"), new.count("\n")))
PY
echo "step1 rc=$?"

echo "=== STEP 2/4  LOG.md ===================================================="
python3 -u - "$LOGLINE" << 'PY'
import io
import sys

L = "/home/users/hshunt/spin_gnn/LOG.md"
line = io.open(sys.argv[1], encoding="utf-8").read()
assert line.startswith("- 2026-11-09:") and line.endswith("\n"), "bad log line"
body = io.open(L, encoding="utf-8").read()
if "Stage M registered and launched" in body:
    print("already present; not appending twice")
else:
    with io.open(L, "a", encoding="utf-8") as f:
        f.write(line)
    print("appended 1 line")
body = io.open(L, encoding="utf-8").read()
print("LOG.md now %d bytes, %d lines, %d dated 2026-11-09"
      % (len(body.encode()), body.count("\n"), body.count("\n- 2026-11-09:")))
PY
echo "step2 rc=$?"

echo "=== STEP 3/4  commit ===================================================="
if [ -e .git/index.lock ]; then
    echo "SKIPPED: .git/index.lock exists. Nothing staged, nothing committed,"
    echo "and the lock is left alone."
    ls -l .git/index.lock
else
    git add STATUS.md LOG.md \
        scripts/slurm/stagem_score_2026-11-09.sbatch \
        scripts/slurm/stagem_record_2026-11-09.sbatch
    echo "add rc=$?"
    if git diff --cached --quiet; then
        echo "nothing staged -- already committed; no commit made"
    else
        git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
            commit -F "$MSG" -q
        echo "commit rc=$?"
    fi
fi
echo "step3 rc=$?"

echo "=== STEP 4/4  push ======================================================"
module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$?"
git log --oneline -2
echo "--- STATUS.md is $(wc -l < STATUS.md) lines, $(wc -c < STATUS.md) bytes ---"
echo "--- still untracked ---"
git status --short | head -10

echo "end $(date '+%F %T')"
