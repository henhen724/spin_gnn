#!/bin/bash
#SBATCH --job-name=provPublish
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/provPublish_%j.out
#
# FINAL RENDER of decks/spin-gnn/provenance (finding 71), after three draft
# renders to $SCRATCH and two read-backs of the prose and every figure.
#
# Pass 1 found eleven defects that all exited 0, among them: a G6 title refuted
# by its own table; a caveat page claiming the position/section confound had
# been registered in advance when it was noticed on the read-back; a page
# saying the hints "are then identical" when the given-side edit recolours a
# hint; a column ("reverts to c") used in a table and defined nowhere.
#
# Pass 2 found four more, three of them substantive: a figure note asserting
# the tied arm is 1.0000 in every bin when one bin is 0.9800; a note calling
# the clash-free column "finding 68's construction exactly" when it differs in
# the one way this deck exists to separate (round 0 vs rounds 1-15); a
# comparison title saying "both rates here are far higher" on a table whose
# three tied rows are equal or lower. Pass 2 again found more than a clean
# render would suggest, which is why it is not skipped.
#
# The 13-word conditional title that killed job 52268621 was invisible to
# deck_lint.py, because the lint only read string LITERALS. The lint now
# follows conditionals, dict lookups and names bound to either, and checks
# every branch; that change is committed here with the builder.
#
# The builder is COMMITTED FIRST and its hash passed back in as --commit, so
# the deck's provenance line names the exact code that drew it. --job names the
# MEASUREMENT job (52267370), not this render.
#
# This job does NOT git add anything in ~/slides: that repo is shared with other
# sessions, and claude-slides-publish.sh stages only decks/<slug> and
# lib/passwords.js. The job renders and validates; publishing is the next step,
# as its own Bash call, because the publish pushes.

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

DECK=/home/users/hshunt/slides/decks/spin-gnn/provenance

echo
echo "=== 1. lint ======================================================="
python3 scripts/deck_lint.py notebooks/make_prov_web.py || {
    echo "LINT FAILED -- nothing rendered, nothing committed"; exit 1; }

echo
echo "=== 2. commit the builder ========================================="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged or committed."
    exit 1
fi
git add notebooks/make_prov_web.py \
        scripts/deck_lint.py \
        scripts/slurm/prov_deck_2026-11-13.sbatch \
        scripts/slurm/prov_publish_2026-11-13.sbatch
git diff --cached --name-only | sed 's/^/  staged /'
if ! git diff --cached --quiet; then
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
Deck for finding 71: the guard is provenance-specific, but depth is larger

Builder for decks/spin-gnn/provenance, rendering the provenance-paired guard
test job 52267370 measured. check_claims() rebuilds every rate the deck prints
from the six per-case CSVs through the scorer's own helpers and asserts each
one against prov_summary.json to 1e-12, so the deck cannot disagree with the
appendix it is drawn from.

RESULT (400 paired cases per cell, rounds 1-15, 6 checkpoints, nothing trained):
  G2 rule, GATE      PASS  every cell >= 0.9975 on both sides, bar 0.60
  G3 placebo, GATE   PASS  0.9876-1.0000 conditional, floor 0.85
  G1 headline        PASS  2 of 3 untied seeds, bar was 2 of 3, 0 reversed
                     s0 given 0.8050 derived 0.9907 (+0.1858, b/c 61/1,
                        Holm 8.2e-17)
                     s1 given 0.6727 derived 0.8559 (+0.1832, b/c 88/27,
                        Holm 1.99e-08)
                     s2 given 0.9809 derived 0.9809 (+0.0000, b/c 5/5, p=1)
  G4 tied control    QUIET |diff| <= 0.0033 in all three seeds
  G6 clash-matched   the same two seeds separate on the 341 exactly matched
                     cases (+0.1786, +0.1767); the summary's all-three
                     G6_agrees flag is False only because s2 misses there too

So breaking one of the model's OWN earlier deductions is substantially less
likely to make it withhold than breaking a hint that does the same work, inside
the same case, at the same node, at the same depth, with the clash count
matched. The seed that misses is J2 s2, which finding 68 already measured at
0.8444 -- the seed with the weakest guard to lose.

NOT THE WHOLE GAP, and the deck says so on three pages. The post-hoc lead this
run came from was 0.0000/0.0000/0.8444 at round 0 against 0.9903/0.8619/0.9830
on chained cases. The paired provenance step is +0.19, the within-run depth
rise from round 1 to rounds 3-15 is +0.21/+0.27, and the step from round 0 to
this run's given side is the largest of the three -- but it crosses case lists
and mixes depth with construction, so it is marked POST-HOC on the figure.

CLASH IS NOT THE MECHANISM. Finding 68's exact edit type -- a hint edit that
clashes with nothing -- gives 0.8571/0.6296/0.9600 here, nowhere near its
round-0 0.0000. What differs is the depth, not how visibly broken the prefix
looks.

TWO THEOREMS came out of the construction and are stated as pages:
  - the CLASH theorem (finding 70): w deduced to colour a has every other
    colour clashing at w, by the very fact that made it deducible, so the set
    of locally proper single-token edits to the reasoning section is EMPTY.
    The edit therefore minimises and records the clash count instead.
  - the ROUND-0 PAIRING theorem: a round-0 v is forced by hints alone, so its
    hint neighbours already cover q-1 colours and any neighbour deduced in that
    same round repeats one of them, giving it two holders. A round-0 case with
    both a hint single holder and a deduced single holder cannot exist. The
    census confirmed it before any model ran: 6828 paired cases, none at
    round 0.

NOT CLAIMED: nothing was trained and rho_50 did not move; the arms still differ
4.1x in parameters; provenance is confounded with POSITION, since the hint sits
in the hints section and the deduction sits later and closer to the readout,
and nothing here separates those -- that confound was noticed on the read-back,
not registered, and the deck says which it is; and nothing localises where in
the network the guard lives.

Also commits a deck_lint.py fix: computed titles and bullets (a verdict
conditional, a dict lookup, a name bound to either) were invisible to it, so it
reported a clean bill while a 13-word title waited to kill the render -- which
it did, job 52268621. It now resolves those shapes and checks every branch.
MSG
    echo "commit rc=$?"
fi
COMMIT=$(git rev-parse --short HEAD)
echo "builder commit: $COMMIT"

echo
echo "=== 3. render into ~/slides ======================================="
mkdir -p "$DECK"
python3 -u notebooks/make_prov_web.py --out "$DECK" --commit "$COMMIT" \
    --job 52267370 || { echo "RENDER FAILED"; exit 1; }

echo
echo "=== 4. build (KaTeX delimiter + macro warnings) ==================="
cd /home/users/hshunt/slides || exit 1
bash tools/build.sh spin-gnn/provenance
echo "build rc=$?"

echo
echo "NOT PUBLISHED by this job -- claude-slides-publish.sh is the next step."
date +'end %F %T'
