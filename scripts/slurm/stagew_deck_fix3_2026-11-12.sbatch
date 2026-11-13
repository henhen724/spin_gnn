#!/bin/bash
#SBATCH --job-name=stagewDeck4
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagewDeck4_%j.out
#
# Fourth render of decks/spin-gnn/budget, from the SECOND text read-back. One
# defect, and it is the kind only a read of the whole deck finds: three tables
# read the same two training histories and two of them labelled a column by the
# log's epoch index (49, 99) while the third labelled the identical values by
# the budget (50, 100). Train NLL 1.5738 therefore appeared under "epoch 49" on
# one page and "epoch 50" on another -- each true in its own convention, and a
# contradiction to anyone holding the two pages side by side.
#
#  * make_stagew_web.MID_EP / END_EP now own every epoch label a table prints,
#    so one page cannot be edited into a different convention from the others.
#    The log-index convention wins because the figures' x axis already uses it
#    (curves.png marks "epoch 58") and $n_e$ on the LEVELS page already means
#    the NLL after epoch e.
#  * LEVELS states the convention outright: a 50-epoch run's last epoch is
#    epoch 49, a 100-epoch run's is epoch 99, so a value under both labels is
#    one value. deck_explain.check() asserts the page states the same two
#    numbers that MID_EP/END_EP print.
#  * The reproducibility table said "difference" with no sign convention; its
#    two difference columns now say "arm minus control", which is what the
#    numbers are.
#
# No result changes, no new arm. Still NO publish step: the figures and the
# four pages touched here get their second PNG read-back after this.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache

echo
echo "=== 1. compile and lint =========================================="
python3 -m py_compile notebooks/make_stagew_web.py notebooks/deck_explain.py \
                      notebooks/stagew_figs.py scripts/step_epoch_census.py || {
    echo "py_compile FAILED"; exit 1; }
python3 scripts/deck_lint.py notebooks/make_stagew_web.py || {
    echo "deck_lint FAILED"; exit 1; }
echo "lint: PASS"

echo
echo "=== 2. explanation pages ========================================="
python3 -u -m notebooks.deck_explain || { echo "deck_explain FAILED"; exit 1; }

echo
echo "=== 3. commit ===================================================="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged or committed."
    exit 1
fi
git add notebooks/make_stagew_web.py notebooks/deck_explain.py \
        scripts/slurm/stagew_deck_fix3_2026-11-12.sbatch
git diff --cached --name-only | sed 's/^/  staged /'
if ! git diff --cached --quiet; then
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
Stage W's deck: one value, two epoch labels, three tables

The second text read-back of the rendered deck found train NLL 1.5738 printed
under "train NLL at epoch 50" on the W-c page and under "train NLL at epoch 49,
100-epoch arm" on the reproducibility page. Same history, same entry, two
conventions: the budget (a 50-epoch run) and the log's 0-based epoch index
(epoch 49). Each label is correct on its own page and the pair is a
contradiction for a reader who holds them together, which is the reader this
deck is written for.

The log-index convention wins, because two things already use it: the figures'
x axis, where curves.png annotates "epoch 58", and $n_e$ on the LEVELS page,
defined as the train NLL after epoch e. MID_EP and END_EP now own every epoch
label any table prints, so the convention cannot be split again by editing one
page, and the one place that legitimately mixes them -- nll_at() takes a budget
while history.json is indexed by epoch -- asserts the two agree before either
number reaches a page.

LEVELS states the convention for the reader rather than leaving it inferable:
a 50-epoch run's last epoch is epoch 49, a 100-epoch run's is epoch 99, and a
value appearing under both labels is one value. deck_explain.check() asserts
that page states the same two numbers the builder prints, computed from the
scorer's own EPOCHS and ADDED_EPOCHS.

The reproducibility table also said "difference" with no sign convention, on a
page whose whole point is which direction the two runs differ in. Both
difference columns now say "arm minus control".

No numbers change. 28 pages, 14 explanation, 7 figures. W-f stays registered
and unsubmitted.
MSG
    echo "commit rc=$?"
fi
SHA=$(git rev-parse --short HEAD)
echo "builder commit: $SHA"

echo
echo "=== 4. render ===================================================="
python3 -u -m notebooks.make_stagew_web --commit "$SHA"
RC=$?
echo "render rc=$RC"

echo
echo "=== 5. KaTeX and delimiter warnings =============================="
# build.sh is scoped to this one deck on purpose: another session has an
# uncommitted deck in the same repo and nothing here should rebuild it.
bash /home/users/hshunt/slides/tools/build.sh spin-gnn/budget
echo "build rc=$?"

echo
echo "=== 6. push ======================================================"
module load system git/2.45.1
git push origin sudoku-dd
echo "push rc=$?"

echo "end $(date '+%F %T')"
exit $RC
