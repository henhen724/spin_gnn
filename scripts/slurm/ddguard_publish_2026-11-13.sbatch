#!/bin/bash
#SBATCH --job-name=ddgPublish
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/ddgPublish_%j.out
#
# FINAL RENDER of decks/spin-gnn/guard, after four draft renders to $SCRATCH and
# two read-backs of every figure. The drafts caught six defects that all exited 0:
#   - a claim page asserting three rates "coincide" while plotting one
#     unconditional rate against two conditional ones (different denominators,
#     so they could not have coincided and the title was false);
#   - two tables mixing those same denominators in adjacent columns;
#   - a ratio column reading 6 on noise-over-noise and 31 at J2 s2 against 32 at
#     J2 s0, making a 0.331 effect look like a 10.020 one;
#   - lines drawn across the tied/untied divider, a trend along an axis with none;
#   - two pairs of right-hand end labels struck through each other;
#   - coincident series painting over each other, so the page whose whole claim
#     is "these are equal" showed one visible dot.
#
# The builder is COMMITTED FIRST and its hash passed back in as --commit, so the
# deck's provenance line names the exact code that drew it. It is a separate Bash
# call from the publish for the same reason every gated command is: the publish
# pushes, and a chain would hide which half failed.
#
# This job does NOT git add anything in ~/slides. That repo is shared with other
# sessions -- there is an uncommitted decks/arpes-detector-sim/deck.md belonging
# to another session right now -- and claude-slides-publish.sh stages only
# decks/<slug> and lib/passwords.js. The job renders and validates; publishing is
# the separate step.

set -uo pipefail
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
date +'start %F %T'

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
module load system git/2.45.1
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

DECK=/home/users/hshunt/slides/decks/spin-gnn/guard

echo
echo "=== 1. lint ======================================================="
python3 scripts/deck_lint.py notebooks/make_ddguard_web.py || {
    echo "LINT FAILED -- nothing rendered, nothing committed"; exit 1; }

echo
echo "=== 2. commit the builder ========================================="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged or committed."
    exit 1
fi
git add notebooks/make_ddguard_web.py \
        scripts/slurm/ddguard_deck_2026-11-13.sbatch \
        scripts/slurm/ddguard_publish_2026-11-13.sbatch
git diff --cached --name-only | sed 's/^/  staged /'
if ! git diff --cached --quiet; then
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
Deck for the fire-and-guard probe: the shared block has DD's rule, not its guard

Builder for decks/spin-gnn/guard. Renders from the per-case CSVs job 52211808
wrote; check_claims asserts every rate it prints against ddguard_summary.json,
which the registered scorer wrote inside that job, so the deck cannot disagree
with its own appendix.

RESULT (350 paired round-0 cases per cell, 6 checkpoints, nothing trained):
  H1 rule   CONFIRMED  S1 1.0000/1.0000/1.0000, J2 1.0000/0.9971/1.0000
  H3 placebo CONFIRMED  every cell >= 0.9343, floor was 0.85
  H2 guard  CONFIRMED  S1 1.0000/1.0000/1.0000 (spread 0.0000)
                        J2 0.0000/0.0000/0.8444, median diff +1.0000
            McNemar 3 of 3 seed pairs significant with b > c, bar was 2:
            Holm p 5.8e-70 / 2.7e-71 / 6.0e-08, and c = 0 in all three.

So one shared block applies DD's elimination rule as well as its untied twin and
does not run DD's scheduler: it picks a node without checking determinacy. Mean
drop in v's node logit under the determinacy-destroying edit is -0.003/+0.001/
+0.000 for the tied arm against +10.020/+16.130/+0.331 untied, so the invariance
is real and not an argmax threshold effect. Finding 28's decomposition transfers
to the tied/untied pair.

The guard statistic has ZERO seed spread inside the tied arm where rho_50 spans
0.754-0.765, which is what STATUS next-step 1 candidate (b) was asking for.

NOT CLAIMED: nothing closes the 0.011 accuracy gap, the arms still differ 4.1x in
parameters, this is round 0 only, and it does not localise where the guard lives.
J2 seed 2 is a weak-but-real guard (broken +0.331 vs placebo -0.011), so "J2 has
the guard" is 2 of 3, not 3 of 3, and the deck says so on its own page.

Four draft renders to $SCRATCH and two read-backs of every figure caught six
defects with no exit code between them, listed in the publish sbatch header. The
worst was structural rather than cosmetic: a claim page asserted that three rates
coincide while plotting one UNCONDITIONAL rate against two rates CONDITIONED on
it. Different denominators cannot coincide, so the title was false on a figure
that rendered clean and linted clean. The deck now plots the unconditional rates,
tables the conditional ones, writes conditionals as "broken | clean", and carries
a page defining both denominators.
MSG
    echo "commit rc=$?"
fi
COMMIT=$(git rev-parse --short HEAD)
echo "builder commit: $COMMIT"

echo
echo "=== 3. render into ~/slides ======================================="
mkdir -p "$DECK"
python3 -u notebooks/make_ddguard_web.py --out "$DECK" --commit "$COMMIT" \
    --job 52211808 || { echo "RENDER FAILED"; exit 1; }

echo
echo "=== 4. build (KaTeX delimiter + macro warnings) ==================="
cd /home/users/hshunt/slides || exit 1
./tools/build.sh spin-gnn/guard
echo "build rc=$?"

echo
echo "NOT PUBLISHED by this job -- claude-slides-publish.sh is the next step."
date +'end %F %T'
