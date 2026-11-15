#!/bin/bash
#SBATCH --job-name=channelDeck
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/channelDeck_%j.out
#
# First render of decks/spin-gnn/channel, the stage-Y deck.
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
Stage Y's deck: the determinacy channel, and the guard that still does not fire

decks/spin-gnn/channel, built from stage Y's own artifacts. Every number is
read out of history.json, the three 14-point rho grids, skip_behavior.csv,
determinacy_both_arms.csv and stagey_verdict.json; every bar is imported from
scripts/stagey_score.py, which registered them before the first cell ran.
check_claims() asserts the accuracies, guard rates and rho_50 values the pages
print against the verdict file the measurement job wrote independently, so a
deck whose prose has drifted fails to build rather than publishing to a URL.

The headline is the registered verdict, "Y4 failed, Y1-Y3 not interpreted",
and the deck does not reach for the two-seed subset that looks better. All
four bars fail on their own numbers anyway.

The scientific content is the second refutation. Stage X installed the
determinacy representation on the existing readout (AUROC 0.947) and changed
no behaviour; two explanations survived that -- the readout was answering two
questions at once, and a monotone block gate can only suppress. The channel
removes both: the determinacy loss reads a separate score g = <h*s, E_v>, and
log sigma(g) is added per node so the emission can be re-ordered rather than
only lowered. AUROC 0.9493/0.9478/0.9480 and a guard that fires once in 681
cases. The gap is not the parameter conflict and not the monotonicity.

One thing the channel does buy, and the deck says so: stage X's uniform
accuracy tax is gone on both converged seeds (+0.0003 and -0.0005 against
-0.0042/-0.0063/-0.0055), and stage X's collapse of the solved/unsolved call
does not reappear. The cost is convergence time -- seed 0 leaves its plateau
only at epoch 41, seed 2 never does, and seed 2 is why Y4 fails.

Pages that exist because a reader has seen no other deck from this project:
the determinacy predicate, the guard, the conditional denominator, the gauge
argument, stage X and its null result, the static-ranking vs counterfactual-
responsiveness distinction, finding 74's re-ranking shape, the channel's
equation with every symbol defined, the zero-init and the bug it replaced,
and the four knobs deliberately not turned.

SUFFICIENCY is dropped from the shared explanation set -- no residual against
the isotonic ladder is reported here. CENSOR is kept and is load-bearing:
seed 2's rho_50 is genuinely censored.

No cell retrained. No published number moves.
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
