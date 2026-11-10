#!/bin/bash
#SBATCH --job-name=stagepRegister
#SBATCH --partition=sganguli
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagepRegister_%j.out
#
# REGISTER STAGE P BEFORE ITS FIRST CELL CAN START.
#
# The array scripts/slurm/sudoku2_n48_tied_P_2026-11-10.sbatch is submitted
# --dependency=afterok on THIS job, so every step below fails closed:
#
#   1. the dry check for the new arch_mode must pass 7/7. A tied cell that does
#      not actually share its block, or that regresses the untied count the whole
#      ladder rests on, must never reach a GPU.
#   2. the registered scorer must run and exit 0. Its exit code is guard 4 --
#      the untied twin J2 is re-read from disk and checked against its frozen
#      literals, and if it has drifted this job fails, the array's afterok is
#      never satisfied, and no cell starts against an unverifiable control.
#   3. the commit must happen. The docstring of scripts/stagep_score.py IS the
#      registration, so it is in git before any number exists to tempt it.
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

MSG=/scratch/users/hshunt/commit_msg_1110_stagep.txt

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/3  the tied arch_mode dry check (7/7 or nothing starts) ======"
python3 -u -m scripts._tied_dry_check
DRY=$?
echo "dry check rc=$DRY"
if [ "$DRY" -ne 0 ]; then
    echo "FAILING CLOSED: the tied mode does not do what the registration says."
    echo "No cell may start. Nothing committed."
    exit 1
fi

echo
echo "=== STEP 2/3  run the registered scorer DRY (no cells on disk yet) ====="
echo "Everything below should read PENDING for the P1 cell and the CONTROL"
echo "block should find the untied twin J2 and H2 on disk and matching."
python3 -u -m scripts.stagep_score
SCORER=$?
echo "scorer rc=$SCORER"
if [ "$SCORER" -ne 0 ]; then
    echo "FAILING CLOSED: the untied twin is missing or has drifted, so P0 has"
    echo "nothing to be a difference against. The array's afterok will not fire."
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

git add spin_gnn/sat_transformer2.py spin_gnn/sudoku_hparam_sweep.py \
    scripts/_tied_dry_check.py scripts/stagep_score.py \
    scripts/slurm/sudoku2_n48_tied_P_2026-11-10.sbatch \
    scripts/slurm/stagep_register_2026-11-10.sbatch
echo "add rc=$?"
echo "--- staged ---"
git diff --cached --name-only

python3 -u - << 'PY'
import subprocess
import sys

staged = subprocess.run(["git", "diff", "--cached", "--name-only"],
                        capture_output=True, text=True).stdout.split()
need = ["scripts/stagep_score.py",
        "scripts/slurm/sudoku2_n48_tied_P_2026-11-10.sbatch",
        "spin_gnn/sat_transformer2.py"]
for p in need:
    if p not in staged:
        print("NOT STAGED -- is it gitignored? a `git add` on an ignored path "
              "says nothing: %s" % p)
        sys.exit(1)
print("the registration, the array and the model change are all staged")
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
echo "the array may now start; score with: python3 -u -m scripts.stagep_score"
