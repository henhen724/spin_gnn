#!/bin/bash
#SBATCH --job-name=stageuRegister
#SBATCH --partition=sganguli
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stageuRegister_%j.out
#
# REGISTER STAGE U BEFORE ITS FIRST CELL CAN START.
#
# The array scripts/slurm/sudoku2_n48_cycled_U_2026-11-11.sbatch is submitted
# --dependency=afterok on THIS job, so every step below fails closed:
#
#   1. the cycled dry check must pass. k=1 must still count 613,818 -- stage S's
#      cell IS the control and a moved control is a ladder with no baseline --
#      k=2 and k=4 must count 1,025,978 and 1,850,298 exactly, both must be above
#      H1's 879,418 censoring floor, the application order must be the cycle
#      0,1,..,k-1 and not k applications or a 0,0,1,1 repeat, and the feedforward
#      must write zero in EVERY block, not just the first.
#   2. THE FULL TEST SUITE must pass. This is the step stage S did not need:
#      stage S changed no model code, and stage U changes
#      spin_gnn/sat_transformer2.py (the forward loop itself), config.py, the
#      trainer's save/load, the sweep CLI, minarch_analysis.rung_key and
#      rho_transition_eval's row. tests/test_cycled_blocks.py has never been
#      executed, and a k=1 cell that is no longer byte-identical to stage S's
#      would invalidate every number this stage differences against.
#   3. the registered scorer must run and exit 0. Its exit code is guard 4 --
#      stage S's k=1 cell is re-read from disk and checked against the frozen
#      literals (613,818 trainable, rho_50 0.757, escapes 3/3). If it has
#      drifted, this job fails, the array's afterok never fires, and no GPU is
#      spent on a ladder whose bottom rung is unverifiable.
#   4. the commit must happen. The docstring of scripts/stageu_score.py IS the
#      registration -- the hypothesis, what stage T contributed, which reading
#      outranks which, and what runs next in each case -- so it is in git before
#      any number exists to tempt it.
#
# From a compute node, like every git step on this project: $HOME I/O on
# sh02-ln02 wedges intermittently in cl_sync_io_wait. The commit refuses to run
# if a .git/index.lock exists and never deletes it.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

MSG=/scratch/users/hshunt/commit_msg_1111_stageu.txt

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/4  the cycled-blocks dry check ==============================="
python3 -u -m scripts._cycled_dry_check
DRY=$?
echo "dry check rc=$DRY"
if [ "$DRY" -ne 0 ]; then
    echo "FAILING CLOSED: the k ladder is not what the registration says it is."
    echo "No cell may start. Nothing committed."
    exit 1
fi

echo
echo "=== STEP 2/4  the FULL test suite (the model forward changed) ==========="
echo "27 files / 237 cases was the last clean run (51919540), with"
echo "test_exact_recovery and test_transformer_shapes SKIPPED by name for"
echo "torch_geometric. tests/test_cycled_blocks.py is new and has never run."
python3 -u scripts/run_tests.py
TESTS=$?
echo "test suite rc=$TESTS"
if [ "$TESTS" -ne 0 ]; then
    echo "FAILING CLOSED: the suite does not pass with the k plumbing in place."
    echo "A GPU-hour spent on a model whose k=1 path has moved is a wasted one."
    exit 1
fi

echo
echo "=== STEP 3/4  run the registered scorer DRY (no new cells on disk) ======"
echo "Everything below should read PENDING for U1 and U2, and the CONTROL block"
echo "should find stage S's k=1 d320 cell on disk and matching its literals."
python3 -u -m scripts.stageu_score
SCORER=$?
echo "scorer rc=$SCORER"
if [ "$SCORER" -ne 0 ]; then
    echo "FAILING CLOSED: stage S's k=1 control is missing or has drifted, so the"
    echo "ladder has nothing to be a difference against. The array's afterok will"
    echo "not fire."
    exit 1
fi

echo
echo "=== STEP 4/4  commit the registration ==================================="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged, nothing"
    echo "committed, and the lock is left alone."
    ls -l .git/index.lock
    exit 1
fi

git add scripts/stageu_score.py scripts/_cycled_dry_check.py \
    scripts/slurm/sudoku2_n48_cycled_U_2026-11-11.sbatch \
    scripts/slurm/stageu_register_2026-11-11.sbatch \
    scripts/slurm/staget_score_2026-11-11.sbatch \
    tests/test_cycled_blocks.py \
    spin_gnn/sat_transformer2.py spin_gnn/config.py \
    spin_gnn/train_sudoku_transformer.py spin_gnn/sudoku_hparam_sweep.py \
    scripts/minarch_analysis.py scripts/rho_transition_eval.py
echo "add rc=$?"
echo "--- staged ---"
git diff --cached --name-only

python3 -u - << 'PY'
import subprocess
import sys

staged = subprocess.run(["git", "diff", "--cached", "--name-only"],
                        capture_output=True, text=True).stdout.split()
need = ["scripts/stageu_score.py",
        "scripts/_cycled_dry_check.py",
        "tests/test_cycled_blocks.py",
        "spin_gnn/sat_transformer2.py",
        "scripts/slurm/sudoku2_n48_cycled_U_2026-11-11.sbatch"]
for p in need:
    if p not in staged:
        print("NOT STAGED -- is it gitignored? a `git add` on an ignored path "
              "says nothing: %s" % p)
        sys.exit(1)
print("the registration, the dry check, the tests, the model and the array are "
      "all staged")
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
echo "the array may now start; score with: python3 -u -m scripts.stageu_score"
