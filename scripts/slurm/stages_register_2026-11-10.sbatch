#!/bin/bash
#SBATCH --job-name=stagesRegister
#SBATCH --partition=sganguli
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagesRegister_%j.out
#
# REGISTER STAGE S BEFORE ITS FIRST CELL CAN START.
#
# The array scripts/slurm/sudoku2_n48_tiedwidth_S_2026-11-10.sbatch is submitted
# --dependency=afterok on THIS job, so every step below fails closed:
#
#   1. the width dry check must pass. d320 and d448 must count 613,818 and
#      1,088,698 exactly, d192 must still count 270,010 so the trend's own
#      baseline has not moved, and H1's 879,418 floor must STRADDLE the two
#      widths -- that straddle is the only reason the stage has a discriminating
#      cell at all. A wrong count there and no GPU is worth spending.
#   2. the registered scorer must run and exit 0. Its exit code is guard 4 --
#      stage P's d192 cell AND the untied twin J2 are re-read from disk and
#      checked against their frozen literals. The width trend is a difference
#      against P1; if P1 has drifted, this job fails, the array's afterok is
#      never satisfied, and no cell starts against an unverifiable baseline.
#   3. the commit must happen. The docstring of scripts/stages_score.py IS the
#      registration, including the reason width was chosen over the schedule
#      change P2 asked for, so it is in git before any number exists to tempt it.
#
# From a compute node, like every git step on this project: $HOME I/O on
# sh02-ln02 wedges intermittently in cl_sync_io_wait. The commit refuses to run
# if a .git/index.lock exists and never deletes it -- the lock belongs to
# whichever process holds it, and waiting is what works.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

MSG=/scratch/users/hshunt/commit_msg_1110_stages.txt

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/3  the tied WIDTH dry check =================================="
python3 -u -m scripts._tiedwidth_dry_check
DRY=$?
echo "dry check rc=$DRY"
if [ "$DRY" -ne 0 ]; then
    echo "FAILING CLOSED: a width is not what the registration says it is."
    echo "No cell may start. Nothing committed."
    exit 1
fi

echo
echo "=== STEP 2/3  run the registered scorer DRY (no new cells on disk) ====="
echo "Everything below should read PENDING for S1 and S2, and the CONTROL block"
echo "should find stage P's tied d192 cell and the untied twin J2 on disk and"
echo "matching their frozen literals."
python3 -u -m scripts.stages_score
SCORER=$?
echo "scorer rc=$SCORER"
if [ "$SCORER" -ne 0 ]; then
    echo "FAILING CLOSED: stage P's baseline or the untied twin is missing or has"
    echo "drifted, so the width trend has nothing to be a difference against."
    echo "The array's afterok will not fire."
    exit 1
fi

echo
echo "=== STEP 3/3  commit the registration ==================================="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged, nothing"
    echo "committed, and the lock is left alone."
    ls -l .git/index.lock
    exit 1
fi

git add scripts/stages_score.py scripts/_tiedwidth_dry_check.py \
    scripts/slurm/sudoku2_n48_tiedwidth_S_2026-11-10.sbatch \
    scripts/slurm/stages_register_2026-11-10.sbatch \
    scripts/slurm/stagep_score_2026-11-10.sbatch
echo "add rc=$?"
echo "--- staged ---"
git diff --cached --name-only

python3 -u - << 'PY'
import subprocess
import sys

staged = subprocess.run(["git", "diff", "--cached", "--name-only"],
                        capture_output=True, text=True).stdout.split()
need = ["scripts/stages_score.py",
        "scripts/_tiedwidth_dry_check.py",
        "scripts/slurm/sudoku2_n48_tiedwidth_S_2026-11-10.sbatch"]
for p in need:
    if p not in staged:
        print("NOT STAGED -- is it gitignored? a `git add` on an ignored path "
              "says nothing: %s" % p)
        sys.exit(1)
print("the registration, the dry check and the array are all staged")
PY
if [ "$?" -ne 0 ]; then
    echo "FAILING CLOSED: the registration did not stage. Nothing committed."
    exit 1
fi

if git diff --cached --quiet; then
    echo "nothing staged -- already committed; no commit made"
else
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -F "$MSG" -q
    echo "commit rc=$?"
fi

module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$? (non-fatal: the commit is the record)"
git log --oneline -3

echo
echo "=== registered ==="
date +'%F %T'
echo "the array may now start; score with: python3 -u -m scripts.stages_score"
