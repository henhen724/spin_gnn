#!/bin/bash
#SBATCH --job-name=shimCommit
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/shimCommit_%j.out
#
# Second commit of the 11-10 overnight session: the ten test files that had
# never been collected here, and the shim features they need. Same shape as
# depthfix_commit_2026-11-10.sbatch -- a job because $HOME I/O wedges, fail
# closed on .git/index.lock, and verify the index before committing.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists; nothing staged or committed."
    ls -l .git/index.lock
    exit 1
fi

PATHS="
tests/_pytest_shim.py
tests/test_ksat2.py
tests/test_ksat2_tokens.py
tests/test_sat_transformer_shapes.py
tests/test_sudoku_agg_eval.py
tests/test_sudoku_depth_slots.py
tests/test_sudoku_probes.py
tests/test_sudoku_tokens.py
tests/test_sudoku_depth_eval.py
tests/test_sudoku_trace.py
tests/test_train_sudoku_transformer.py
tests/test_sudoku_sae.py
scripts/slurm/run_tests.sbatch
STATUS.md
LOG.md
scripts/slurm/depthfix_commit_2026-11-10b.sbatch
"

# shellcheck disable=SC2086
git add $PATHS
echo "add rc=$?"
git diff --cached --name-only

if git diff --cached --quiet; then
    echo "nothing staged -- already committed"
    git log --oneline -1
    echo "end $(date '+%F %T')"
    exit 0
fi

missing=0
staged=$(git diff --cached --name-only)
for p in $PATHS; do
    if ! printf '%s\n' "$staged" | grep -qx -- "$p"; then
        echo "MISSING FROM INDEX: $p"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "FAILING CLOSED: not committing a partial set."
    exit 1
fi

git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
    commit -q -F - <<'MSG'
test: the suite is 237 cases, not 142 -- ten files had never run here

_pytest_shim.py argued that `approx`, `raises` and fixtures should be absent so
that a test reaching for one fails loudly rather than quietly not running.
Running the whole suite in one job showed the cost of that: ten of the 27 files
still began with a bare `import pytest`, so on this machine -- where every
result in FINDINGS.md is produced -- they had never been collected at all. A
missing `approx` does not make a test fail loudly; it makes the file holding it
disappear, which is the failure mode the shim was written to end.

So the shim now provides what those files actually use: `approx` (pytest's
tolerance rule and its 1e-6/1e-12 defaults, scalars and sequences, and
deliberately not numpy-aware), `raises` with `match` and `.value`, `fail`, and
a `tmp_path` fixture bound at CALL time so an interrupted run has not already
made a directory for every case after it. Fixture directories are not removed
afterwards, on purpose. Loudness moves to where it works: the driver counts an
import error as a failure, and anything still unprovided raises AttributeError
inside the case, with a traceback.

The ten files get the convention test_sudoku_dd.py already used -- path
bootstrap, real pytest when there is one, shim when there is not, and a
__main__ block so each is runnable on its own -- applied by a scripted edit
(/scratch/users/hshunt/convert_pytest_imports.py) whose diffs were read file by
file before writing. test_sudoku_sae's in-function `import pytest` goes the
same way.

Result: 27 files, 237 cases, 0 failures (job 51872982), against 142 cases and
12 import failures this morning. The two remaining import failures are
test_exact_recovery and test_transformer_shapes, which reach spin_gnn/model.py
and therefore torch_geometric; `ml spider torch-geometric` finds nothing on
Sherlock, so they cannot run here at all and that is recorded rather than
worked around. run_tests.sbatch also loads pandas and scipy now, which the
newly collectable eval tests need.
MSG
echo "commit rc=$?"
git log --oneline -1

module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$? (non-fatal: the commit is the record)"
echo "end $(date '+%F %T')"
