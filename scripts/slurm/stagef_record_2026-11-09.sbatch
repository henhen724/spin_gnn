#!/bin/bash
#SBATCH --job-name=stagefRecord
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagefRecord_%j.out
#
# Land stage F: rewrite STATUS.md, archive the findings, append LOG.md's line,
# copy the deck into the repo, commit and push.
#
# FROM A COMPUTE NODE because $HOME I/O on sh02-ln02 wedged four times today in
# cl_sync_io_wait (a git commit held a 0-byte .git/index.lock for 33 minutes and
# then completed by itself, exit 0; a 1.5 KB LOG.md append was killed at 120 s
# having written nothing). Reported to hshunt, not worked around.
#
# Every step is idempotent and asserts before it writes: the split refuses to
# overwrite an existing FINDINGS.md or to run twice, the LOG append checks for
# its own text first, and the commit refuses to run if a .git/index.lock exists
# (that lock belongs to whichever process holds it -- waiting is what works).

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module load devel math viz python/3.12.1 2>/dev/null

MSG=/scratch/users/hshunt/commit_msg_1109_stagef.txt
LOGLINE=/scratch/users/hshunt/stagef_logline_2026-11-09.txt
DECK=/scratch/users/hshunt/spin_gnn/decks/stagef_2026-11-09/stagef_slides.pdf
DEST=notebooks/sudoku_n96_stagef_2026-11-09_slides.pdf

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/5  STATUS.md -> FINDINGS.md + new handoff ===================="
python3 -u /scratch/users/hshunt/split_status_1109f.py
echo "step1 rc=$?"

echo "=== STEP 2/5  LOG.md ===================================================="
python3 -u - "$LOGLINE" << 'PY'
import io
import sys

L = "/home/users/hshunt/spin_gnn/LOG.md"
line = io.open(sys.argv[1], encoding="utf-8").read()
assert line.startswith("- 2026-11-09:") and line.endswith("\n"), "bad log line"
body = io.open(L, encoding="utf-8").read()
if "stage F scored" in body:
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

echo "=== STEP 3/5  deck into the repo ========================================"
cp "$DECK" "$DEST"
echo "cp rc=$?  $(ls -l "$DEST" | awk '{print $5, $9}')"

echo "=== STEP 4/5  commit ===================================================="
if [ -e .git/index.lock ]; then
    echo "SKIPPED: .git/index.lock exists. Nothing staged, nothing committed,"
    echo "and the lock is left alone."
    ls -l .git/index.lock
else
    git add STATUS.md FINDINGS.md LOG.md "$DEST" \
        notebooks/sudoku_n96_stagef_2026-11-09_results.ipynb \
        scripts/slurm/stagef_writeup_2026-11-09.sbatch \
        scripts/slurm/stagef_record_2026-11-09.sbatch
    echo "add rc=$?"
    if git diff --cached --quiet; then
        echo "nothing staged -- already committed; no commit made"
    else
        git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
            commit -F "$MSG" -q
        echo "commit rc=$?"
    fi
fi
echo "step4 rc=$?"

echo "=== STEP 5/5  push ======================================================"
module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$?"
git log --oneline -2
echo "--- STATUS.md is $(wc -l < STATUS.md) lines, $(wc -c < STATUS.md) bytes ---"
echo "--- still untracked ---"
git status --short | head -10

echo "end $(date '+%F %T')"
