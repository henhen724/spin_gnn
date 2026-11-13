#!/bin/bash
#SBATCH --job-name=stagewDeck5
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagewDeck5_%j.out
#
# Fifth render of decks/spin-gnn/budget, from the SECOND PNG read-back of the
# four figures that had only had one (curves, acc, tokenization, trace). Two
# defects, both in how a figure reads rather than in what it reports:
#
#  * curves.png printed seed 1's "100 ep" label across the 1.600 floor line.
#    The level lines span the axes, so a label at a curve's right-hand end lands
#    on one whenever the curve ends near that level -- which on a figure about
#    which level a cell reached is precisely when it happens. clearing_dy()
#    doubles the offset in the SAME direction when the label would touch a
#    level, so the two arms keep opposite sides and only the colliding facet
#    moves. Labels are now placed after tight_layout(), because the points-to-
#    data answer it depends on changes when the axes is resized.
#  * acc.png and level.png drew BOTH arms in the arm's blue, two pages after
#    curves.png and solve.png taught the reader that grey is the control. The x
#    axis still said which column was which, so nothing was wrong -- it just had
#    to be unlearned. _seed_dots() now colours the control MUTED.
#
# tokenization.png and trace.png passed their second read unchanged, including
# their arithmetic: 29 hints + 19 deductions = 48 nodes, 59 + 289 + 45 = 393
# tokens, and the worked deduction's emission is the trace's first.
#
# No numbers change, no new arm. This is the last render before publishing.

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
git add notebooks/make_stagew_web.py notebooks/stagew_figs.py \
        scripts/slurm/stagew_deck_fix4_2026-11-12.sbatch
git diff --cached --name-only | sed 's/^/  staged /'
if ! git diff --cached --quiet; then
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
Stage W's deck: a label on the floor line, and grey that stopped meaning control

Second PNG read-back of the four figures that had only had one.

curves.png put seed 1's "100 ep" label exactly on the 1.600 floor line. The
level lines span the whole axes, so a label at a curve's right-hand end sits on
one whenever that curve ends near the level -- and this figure exists to show
which level each cell reached, so the collision is structural, not bad luck.
clearing_dy() asks the laid-out axes where an offset in points lands in data
units and doubles it when the label would touch a level, keeping the direction:
the two arms' labels are on opposite sides of their curves and flipping one
would stack them. Placement moved after tight_layout(), since resizing the axes
changes the answer. Only the facet that collides moves; seed 0 and seed 2 are
byte-identical to before.

acc.png and level.png drew both arms' seeds in SERIES[0]. Two pages earlier
curves.png and solve.png had just established MUTED = the 50-epoch control,
SERIES[0] = the arm under test. Nothing on those pages was wrong -- the x axis
names the arms -- but a reader had to drop a convention and pick it up again
three pages later. The control's dots are MUTED now, on both pages.

tokenization.png and trace.png passed unchanged. Their arithmetic was checked
rather than assumed: 29 hints plus 19 deductions is 48 nodes, the section
lengths 59 + 289 + 45 are the stated 393 tokens, the reasoning section's 45 is
1 + 5 rounds + 38 node/colour tokens + 1 terminator, and the worked deduction
on the right of trace.png emits exactly the pair the trace emits first.

No numbers change. 28 pages, 14 explanation, 7 figures.
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
bash /home/users/hshunt/slides/tools/build.sh spin-gnn/budget
echo "build rc=$?"

echo
echo "=== 6. push ======================================================"
module load system git/2.45.1
git push origin sudoku-dd
echo "push rc=$?"

echo "end $(date '+%F %T')"
exit $RC
