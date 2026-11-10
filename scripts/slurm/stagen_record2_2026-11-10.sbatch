#!/bin/bash
#SBATCH --job-name=stagenRecord2
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagenRecord2_%j.out
#
# Ship stage N: commit the generated artifacts -- the deck and the executed
# notebook -- with the rewritten STATUS.md and the second LOG.md line for
# 2026-11-10, and push.
#
# The artifacts were deliberately NOT committed by the write-up job that made
# them (51799922). They are read back first -- all eight PNGs at the sha the
# footer prints, and the notebook's 309 lines of output against the registered
# scorer -- and only then committed, here. That order is why the deck and the
# notebook can be trusted to agree with the score.
#
# From a compute node, like every git step on this project: $HOME I/O on
# sh02-ln02 wedges intermittently in cl_sync_io_wait. Every step asserts before
# it writes -- the append checks for its own text first, STATUS.md is checked
# against today's header and the 60-line cap, the staged set is checked to
# contain the two artifacts (a `git add` on an ignored path is silent), and the
# commit refuses to run if a .git/index.lock exists. Waiting is what works; the
# lock belongs to whichever process holds it and is never deleted.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module load devel math viz python/3.12.1 2>/dev/null

MSG=/scratch/users/hshunt/commit_msg_1110_stagen_ship.txt
LOGLINE=/scratch/users/hshunt/logline_1110_stagen_stageo.txt
PDF=notebooks/stagen_2026-11-10/slides.pdf
NB=notebooks/sudoku_n48_stagen_2026-11-10_results.ipynb

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/4  the artifacts must be on disk ============================="
ls -l "$PDF" "$NB" || { echo "FAILING CLOSED: an artifact is missing"; exit 1; }

echo "=== STEP 2/4  LOG.md ===================================================="
python3 -u - "$LOGLINE" << 'PY'
import io
import sys

L = "/home/users/hshunt/spin_gnn/LOG.md"
line = io.open(sys.argv[1], encoding="utf-8").read()
assert line.startswith("- 2026-11-10:") and line.endswith("\n"), "bad log line"
TOKEN = "Stage N scored, decked and notebooked (finding 56)"
assert TOKEN in line, "the dedupe token is not in the line it guards"
body = io.open(L, encoding="utf-8").read()
if TOKEN in body:
    print("already present; not appending twice")
else:
    with io.open(L, "a", encoding="utf-8") as f:
        f.write(line)
    print("appended 1 line")
body = io.open(L, encoding="utf-8").read()
n_today = body.count("\n- 2026-11-10:")
print("LOG.md now %d bytes, %d lines, %d dated 2026-11-10"
      % (len(body.encode()), body.count("\n"), n_today))
assert n_today <= 3, "more than 3 dated lines for today: %d" % n_today
PY
echo "step2 rc=$?"

echo "=== STEP 3/4  STATUS.md must be today's file and under the cap =========="
python3 -u - << 'PY'
import io
P = "/home/users/hshunt/spin_gnn/STATUS.md"
body = io.open(P, encoding="utf-8").read()
n = body.count("\n")
assert body.startswith("# STATUS — spin_gnn, branch `sudoku-dd` (2026-11-10, "
                       "stage M running, stage O queued)"), \
    "STATUS.md is not the file this job was written for"
assert n < 60, "STATUS.md is over the 60-line cap: %d" % n
for needed in ("51801489", "51801495", "51801620", "56 (11-10)"):
    assert needed in body, "STATUS.md does not mention %s" % needed
print("STATUS.md %d lines, %d bytes -- under the cap, stage O recorded"
      % (n, len(body.encode())))
PY
echo "step3 rc=$?"

echo "=== STEP 4/4  commit and push ==========================================="
if [ -e .git/index.lock ]; then
    echo "SKIPPED: .git/index.lock exists. Nothing staged, nothing committed,"
    echo "and the lock is left alone."
    ls -l .git/index.lock
else
    git add STATUS.md LOG.md "$PDF" "$NB" \
        scripts/slurm/stageo_score_2026-11-10.sbatch \
        scripts/slurm/stagen_record2_2026-11-10.sbatch
    echo "add rc=$?"
    echo "--- staged ---"
    git diff --cached --name-only
    python3 -u - "$PDF" "$NB" << 'PY'
import subprocess
import sys

staged = subprocess.run(["git", "diff", "--cached", "--name-only"],
                        capture_output=True, text=True).stdout.split()
for p in sys.argv[1:]:
    assert p in staged, ("not staged -- is it gitignored? a `git add` on an "
                         "ignored path says nothing: %s" % p)
print("both artifacts are staged")
PY
    if [ "$?" -ne 0 ]; then
        echo "FAILING CLOSED: an artifact did not stage. Nothing committed."
        exit 1
    fi
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
echo "push rc=$? (non-fatal: the commit is the record)"
git log --oneline -3
echo "--- STATUS.md is $(wc -l < STATUS.md) lines ---"
git status --short | head -10

echo "end $(date '+%F %T')"
