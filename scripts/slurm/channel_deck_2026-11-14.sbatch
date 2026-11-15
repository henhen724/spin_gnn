#!/bin/bash
#SBATCH --job-name=channelDeck
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/channelDeck_%j.out
#
# Render of decks/spin-gnn/channel, the stage-Y deck. Run once per revision of
# the builder: the first render was b288a76, this one fixes two unreadable
# y-axes and three titles that did not survive being checked against their own
# figure. Re-running is idempotent -- it overwrites deck.md and the PNGs.
#
# Stage Y finished clean: array 52348732 (3 cells, 50 epochs, 9h15-9h18 each)
# and probe 52375912 (37 s), both COMPLETED 0:0. The registered verdict in
# runs/2026-11-14b_stagey_guard/stagey_verdict.json is
#
#     "Y4 failed, Y1-Y3 not interpreted"
#
# and all four bars fail on their own numbers regardless: Y4 0.4825/0.4825/
# 0.4572 against 0.4800, Y1 0.772/0.758/censored against 0.754/0.757/0.765,
# Y2 1.0000/0.9956/1.0000 against 0.90, Y3 unreadable at n=1 fired case.
#
# WHY THIS IS A DECK AND NOT A LINE IN STATUS.md. It is the second independent
# coupling of the determinacy signal to the output, and it refutes the two
# explanations stage X's null result left open -- that the readout was carrying
# two objectives at once, and that a block-level gate could only suppress. The
# channel fixes both (separate score, per-node re-ranking, 320 new parameters)
# and the guard still fires once in 681 cases.
#
# The deck is NESTED under spin-gnn, so it inherits that project's existing
# password; publishing it costs a link, not a password. Publishing is NOT done
# here -- claude-slides-publish.sh is run by hand after the PNGs are read back,
# which is the step this job exists to make possible.
#
# No cell is retrained and no published number moves. Nothing here writes to
# any directory outside $HOME/spin_gnn and $HOME/slides/decks/spin-gnn/channel.

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
python3 -m py_compile notebooks/make_channel_web.py notebooks/deck_explain.py \
                      scripts/stagey_score.py || { echo "py_compile FAILED"; exit 1; }
python3 scripts/deck_lint.py notebooks/make_channel_web.py || {
    echo "deck_lint FAILED"; exit 1; }
echo "lint: PASS"

echo
echo "=== 2. the scorer's own self-test, and the shared explanation set ="
python3 scripts/stagey_score.py --self-test || { echo "self-test FAILED"; exit 1; }
python3 -u -m notebooks.deck_explain || { echo "deck_explain FAILED"; exit 1; }

echo
echo "=== 3. commit ===================================================="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged or committed."
    exit 1
fi
git add notebooks/make_channel_web.py \
        scripts/slurm/channel_deck_2026-11-14.sbatch
git diff --cached --name-only | sed 's/^/  staged /'
if ! git diff --cached --quiet; then
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
Stage Y's deck, second render: readable axes and three titles that were wrong

Re-render of decks/spin-gnn/channel after reading the first render's PNGs
back. No number in the deck changes; what changes is whether a reader can
see the numbers that were already there, plus three page titles that did not
survive being checked against their own figure.

Axes. acc_trajectory and train_nll both plot a difference of 0.02 and 0.07
sitting on top of an initial collapse of 0.2 and 1.7, so the full range put
every claim on those pages inside a sliver a few pixels tall -- seed 0's
epoch-41 step was invisible. A _zoom() helper cuts the shared y-axis to the
epochs after the collapse (2 for accuracy, MIN_EP for NLL, the step
detector's own boundary) and WRITES ON THE FIGURE that earlier epochs run off
it, because a clipped curve with no note is a figure that lies about its own
range. train_nll's in-axes "the 1.600 floor line" label, which was drawn
across the curves, is gone; the page note already says what the dashed line
is.

Titles that were false against their own figure:
- "The AUROC is won in one epoch" -- true of seed 0 (0.9059 at epoch 1) and
  not of seeds 1 and 2 (0.8467, 0.7684). The seeds are within 0.01 of each
  other and 0.019 of final from epoch 5, so: five epochs.
- "Seeds 0 and 1 sit on the control curve almost exactly" -- seed 1 does
  (0.758 vs 0.757); seed 0 sits visibly right of it (0.772 vs 0.754). Now
  names seed 1 and seed 2, and the note places seed 0.
- draw_auroc's docstring claimed the seeds lie within 0.003 after the first
  epoch. They differ by 0.1375 there. Corrected, and the docstring now also
  records that the curve is the training run's own per-epoch AUROC while the
  grey band is the probe's control -- different sets, same predicate -- which
  the page note now states too, since the control has no determinacy head and
  a per-epoch control curve does not exist to plot.

Three numbers that were typed are now computed from the histories: seed 0's
pre-step gap and how long it took to close, and stage X's accuracy tax range.
A typed number in a note is the one thing check_claims() cannot catch.

The first render, with the deck's full rationale in its message, is b288a76.
MSG
    echo "commit rc=$?"
fi
SHA=$(git rev-parse --short HEAD)
echo "builder commit: $SHA"

echo
echo "=== 4. render ===================================================="
python3 -u -m notebooks.make_channel_web --commit "$SHA"
RC=$?
echo "render rc=$RC"

echo
echo "=== 5. KaTeX and delimiter warnings =============================="
bash /home/users/hshunt/slides/tools/build.sh spin-gnn/channel
echo "build rc=$?"

echo
echo "=== 6. push ======================================================"
module load system git/2.45.1
git push origin sudoku-dd
echo "push rc=$?"

echo "end $(date '+%F %T')"
exit $RC
