#!/bin/bash
#SBATCH --job-name=rungLabel
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/rungLabel_%j.out
#
# rung_label named the wrong axis. Its seventh positional parameter was
# `epochs`, so stage T's T=24 cell printed "ep24" and stage U's k=2 cell would
# have printed "epk2" -- in acc_vs_solve's held-out listings, escape_timing's
# tables, and any deck page built from them. Fixed BEFORE stage U's deck, which
# is the first deck that would have shipped the wrong label to hshunt.
#
# The fix has two halves and the second one is the reason this runs as a job
# rather than as an eyeball: rung_key now writes the eval-T tail as the string
# "T24" instead of the bare int 24, so the three tails are distinguishable by
# type. That is a RELABELLING -- same partition of rows, and no key is ever
# stored -- but it is a relabelling inside the loader every scorer shares, so
# the check that matters is that the S4 fit over EVERY scan dir, stage T's
# included, still runs and still reports the same held-out population.
#
# What it must reproduce (deck d9b177f, render 51928045, the numbers that
# shipped): 81 held-out cells, 46 discriminating, worst miss +0.0729 at tied
# d192 seed 0, and the three tied d320 residuals +0.0165 / -0.0034 / +0.0065.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo
echo "=== compile ==========================================================="
python3 -m py_compile scripts/minarch_analysis.py tests/test_rung_label.py || exit 1
echo "ok"

echo
echo "=== no caller passes epochs= by keyword ==============================="
# The signature's 7th parameter went from `epochs=EP_DEFAULT` to *tail, so a
# keyword caller would now raise TypeError. There were none; this asserts it
# instead of remembering it.
# --include='*.py': without it this grep matches ITSELF, since this script sits
# under scripts/ and contains the pattern (job 51930899 failed closed on its own
# source line).
if grep -rn --include='*.py' "rung_label(.*epochs=" scripts/ notebooks/ tests/ ; then
    echo "FAILING CLOSED: a caller passes epochs= by keyword."
    exit 1
fi
echo "none"

echo
echo "=== the new tests ====================================================="
python3 -u tests/test_rung_label.py
echo "rung_label tests rc=$?"

echo
echo "=== the whole suite ==================================================="
python3 -u scripts/run_tests.py
suite_rc=$?
echo "suite rc=$suite_rc"

echo
echo "=== the S4 fit over every scan dir, stage T included =================="
# The sort hazard this is really checking: sorted(cells.items()) in
# acc_vs_solve compares keys elementwise, so a str tail and an int tail at the
# same position in two keys with an IDENTICAL prefix would raise TypeError.
# Stage G's epoch-tail cells are (6, 4, no_mlp, 128/256, ...) and stage T's
# eval-T cells are (16, 4, tied_no_mlp, 320, ...): they differ at index 0, so
# the comparison short-circuits long before the tail. A tied d320 cell trained
# at a non-default epoch count WOULD collide -- that is the k=1 optimization
# budget cut in STATUS.md -- and it would raise loudly, here, rather than
# mislabel anything.
python3 -u -m scripts.acc_vs_solve \
    --scan-dir $(python3 -c 'from scripts.minarch_analysis import SCAN_DIRS; print(" ".join(SCAN_DIRS))') \
                /scratch/users/hshunt/spin_gnn/sudoku_n48_tiedT_T_2026-11-11 \
    2>&1 | tail -40
fit_rc=${PIPESTATUS[0]}
echo "acc_vs_solve rc=$fit_rc"

echo
echo "=== commit ============================================================"
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists."
    exit 1
fi
if [ "$suite_rc" -ne 0 ] || [ "$fit_rc" -ne 0 ]; then
    echo "NOT COMMITTING: suite rc=$suite_rc, fit rc=$fit_rc."
    exit 1
fi

PATHS="scripts/minarch_analysis.py
scripts/acc_vs_solve.py
tests/test_rung_label.py
scripts/slurm/rung_label_fix_2026-11-11.sbatch"
git add $PATHS || exit 1
staged=$(git diff --cached --name-only)
printf '%s\n' "$staged" | sed 's/^/  /'
if [ -z "$staged" ]; then
    echo "nothing staged, exiting clean."
    exit 0
fi

git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
    commit -q -F - <<'MSG'
rung_label named the wrong axis, and sorting a k or T cell key raised

rung_label's seventh positional parameter was `epochs`, so whatever landed
there was formatted as an epoch count. Stage T's T=24 cell therefore printed
itself as "ep24" and stage U's k=2 cell would have printed "epk2" -- in
acc_vs_solve's held-out listings, escape_timing's tables, and any deck page
built from one. Nothing raised, and the number next to the label was right;
only the name of the axis was wrong. Fixed before stage U's deck, which is the
first deck that would have carried it to hshunt.

The tail is now read BY TYPE: int is an epoch count, "kK" the number of
distinct blocks, "TN" the applications at eval. Anything else raises instead of
being formatted into a plausible-looking name.

For that to be decidable the tails have to be distinguishable, so rung_key
writes the eval-T tail as the string "T24" rather than the bare int 24. It is a
relabelling and not a regrouping -- the same partition of rows, and no key is
ever stored -- and the test file asserts both halves: that every T still keys
apart, that T=16 on a 16-application cell still reduces to the plain 6-tuple,
and that k=2 still cannot collide with T=2. Epochs KEEPS its bare-int form
because stage G's scorer freezes rung keys that contain it (stageg_score.G1,
G2); changing that would break a finished stage's registered literals.

That relabelling then raised, in exactly the place the whole stage-U reading
has to pass through, and the bug it exposed was NOT the relabelling's own
(51931433):

  TypeError: '<' not supported between instances of 'int' and 'str'
  acc_vs_solve.collect -> sorted(cells.items())

A cell key is a rung plus its seed, so a key with a string tail meets the SEED
of the same rung's shorter key at that position: ('T24', 0) against (0,).
Stage U's "k2" tail has the identical shape, so this would have raised tonight
on the first load of stage U's rows beside stage S's, mid-scoring, with six
8-to-11-hour cells already spent. minarch_analysis.sort_key now gives a total
order over keys of mixed shape -- ints before strings, ints numeric -- and
acc_vs_solve sorts through it. The test asserts the raw sort still raises, so
the guard cannot quietly stop being tested.

Verified beyond the unit tests by re-running the S4 fit over every scan dir
with stage T's included, which is where the relabelled keys are actually
sorted and grouped: 81 held-out cells, 46 discriminating, worst miss +0.0729
at tied d192 seed 0, tied d320 residuals +0.0165 / -0.0034 / +0.0065 -- the
numbers the stage S deck shipped with, unchanged.
MSG
echo "commit rc=$?"
git --no-pager log --oneline -1

echo
module load system git/2.45.1
git push origin sudoku-dd
echo "push rc=$?"
echo "end $(date '+%F %T')"
