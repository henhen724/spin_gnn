#!/bin/bash
#SBATCH --job-name=stagenRecord
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagenRecord_%j.out
#
# Append LOG.md's one line for 2026-11-10 and commit it with the updated
# STATUS.md and stage N's scorer sbatch.
#
# From a compute node, like every git step on this project: $HOME I/O on
# sh02-ln02 wedges intermittently in cl_sync_io_wait. Every step asserts before
# it writes -- the append checks for its own text first, and the commit refuses
# to run if a .git/index.lock exists. Waiting is what works; the lock belongs to
# whichever process holds it and is never deleted.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module load devel math viz python/3.12.1 2>/dev/null

MSG=/scratch/users/hshunt/commit_msg_1110_stagen_record.txt
LOGLINE=/scratch/users/hshunt/stagen_logline_2026-11-10.txt

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/3  LOG.md ===================================================="
python3 -u - "$LOGLINE" << 'PY'
import io
import sys

L = "/home/users/hshunt/spin_gnn/LOG.md"
line = io.open(sys.argv[1], encoding="utf-8").read()
assert line.startswith("- 2026-11-10:") and line.endswith("\n"), "bad log line"
body = io.open(L, encoding="utf-8").read()
if "Stage M mid-run health check" in body:
    print("already present; not appending twice")
else:
    with io.open(L, "a", encoding="utf-8") as f:
        f.write(line)
    print("appended 1 line")
body = io.open(L, encoding="utf-8").read()
print("LOG.md now %d bytes, %d lines, %d dated 2026-11-10"
      % (len(body.encode()), body.count("\n"), body.count("\n- 2026-11-10:")))
PY
echo "step1 rc=$?"

echo "=== STEP 2/3  STATUS.md must still be under the cap ====================="
python3 -u - << 'PY'
import io
P = "/home/users/hshunt/spin_gnn/STATUS.md"
body = io.open(P, encoding="utf-8").read()
n = body.count("\n")
assert body.startswith("# STATUS — spin_gnn, branch `sudoku-dd` (2026-11-10, "
                       "stages M and N running)"), "STATUS.md is not today's file"
assert n < 60, "STATUS.md is over the 60-line cap: %d" % n
print("STATUS.md %d lines, %d bytes -- under the cap" % (n, len(body.encode())))
PY
echo "step2 rc=$?"

echo "=== STEP 3/3  commit and push ==========================================="
if [ -e .git/index.lock ]; then
    echo "SKIPPED: .git/index.lock exists. Nothing staged, nothing committed,"
    echo "and the lock is left alone."
    ls -l .git/index.lock
else
    git add STATUS.md LOG.md scripts/slurm/stagen_score_2026-11-10.sbatch \
        scripts/slurm/stagen_record_2026-11-10.sbatch
    echo "add rc=$?"
    if git diff --cached --quiet; then
        echo "nothing staged -- already committed; no commit made"
    else
        git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
            commit -F "$MSG" -q
        echo "commit rc=$?"
    fi
fi

module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$? (non-fatal)"
git log --oneline -3
echo "--- STATUS.md is $(wc -l < STATUS.md) lines ---"
git status --short | head -10

echo "end $(date '+%F %T')"
