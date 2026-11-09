#!/bin/bash
#SBATCH --job-name=stagekRecord
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagekRecord_%j.out
#
# Append stage K's LOG.md line and land the stage K commit, from a compute node.
#
# WHY A JOB FOR TWO FILE OPERATIONS: I/O to $HOME/spin_gnn from sh02-ln02 is
# intermittently wedging this afternoon. Three processes today have parked at
# 00:00:00 of CPU in cl_sync_io_wait -- scripts.stagek_score at 13:1x, a
# `git commit` at 14:02 (still stuck, holding a 0-byte .git/index.lock), and a
# 1.5 KB append to LOG.md at 14:2x which `timeout` killed at 120 s having written
# nothing (LOG.md verified unchanged: 129,922 bytes, 203 lines, mtime 07:41).
# Other operations on the same files return instantly, including git log,
# git cat-file, git hash-object -w and a write-read-delete probe in $HOME, and
# both write-up jobs read and wrote these same trees from a compute node in
# ~65 s. This is reported to the user, not diagnosed here.
#
# The commit step REFUSES to run if .git/index.lock still exists. That lock
# belongs to the wedged process; it is not this job's to remove, and a second
# git writing an index behind a live one is how an index gets corrupted.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

LOGLINE=/scratch/users/hshunt/stagek_logline_2026-11-09.txt
MSG=/scratch/users/hshunt/commit_msg_1109_kdeck3.txt

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/3  LOG.md ===================================================="
python3 - "$LOGLINE" << 'PY'
import io
import sys

L = "/home/users/hshunt/spin_gnn/LOG.md"
line = io.open(sys.argv[1], encoding="utf-8").read()
assert line.startswith("- 2026-11-09:") and line.endswith("\n"), "bad log line"
body = io.open(L, encoding="utf-8").read()
if "stage K scored" in body:
    print("already present; not appending twice")
else:
    with io.open(L, "a", encoding="utf-8") as f:
        f.write(line)
    print("appended 1 line")
body = io.open(L, encoding="utf-8").read()
print("LOG.md now %d bytes, %d lines, %d dated 2026-11-09"
      % (len(body.encode()), body.count("\n"), body.count("\n- 2026-11-09:")))
PY
echo "step1 rc=$?"

echo "=== STEP 2/3  commit ===================================================="
if [ -e .git/index.lock ]; then
    echo "SKIPPED: .git/index.lock still exists -- the wedged commit has not"
    echo "cleared. Nothing staged, nothing committed. Do not delete that lock."
    ls -l .git/index.lock
else
    git add notebooks/make_stagek_deck.py \
        scripts/slurm/stagek_writeup_2026-11-09.sbatch \
        scripts/slurm/stagek_writeup2_2026-11-09.sbatch \
        scripts/slurm/stagek_record_2026-11-09.sbatch \
        notebooks/sudoku_n48_stagek_2026-11-09_results.ipynb \
        notebooks/sudoku_n48_stagek_2026-11-09_slides.pdf \
        STATUS.md LOG.md
    echo "add rc=$?"
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -F "$MSG" -q
    echo "commit rc=$?"
    git log --oneline -1
fi
echo "step2 rc=$?"

echo "=== STEP 3/3  push ======================================================"
if git log --oneline -1 | grep -q "Stage K closed"; then
    module load system git/2.45.1 2>/dev/null
    git push origin sudoku-dd
    echo "push rc=$?"
else
    echo "SKIPPED: the stage K commit is not HEAD, so there is nothing new to push."
fi
echo "step3 rc=$?"

echo "end $(date '+%F %T')"
