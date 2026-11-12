#!/bin/bash
#SBATCH --job-name=stageuWeb
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stageuWeb_%j.out
#
# STAGE U AS A WEB DECK -- render only, NO publish.
#
# Stage U is the finished run: array 51922710, six cells, all COMPLETED 0:0,
# scored by the registered scorer in 52000549 and closed by the accuracy-to-solve
# curve in 52025112. This renders that reading into ~/slides/decks/spin-gnn/,
# replacing stage T's deck in the project's single gated deck (re-publishing
# keeps the password, so the link already shared keeps working; stage T's pages
# stay in the slides repo's git history).
#
# Nothing is committed or pushed to the slides repo here, and nothing is
# published. The read-back comes first -- deck.md and every figure, twice, as
# this project requires -- because a web deck is a URL and a wrong sentence on a
# URL keeps serving.
#
# In a job and not on the login node for three reasons: it imports torch through
# scripts.minarch_analysis, it bootstraps every pooled cell at 2000 resamples
# twice over (the scorer's loaders and then acc_vs_solve's own collect), and it
# writes into $HOME, where sh02-ln02's I/O wedges intermittently.
#
# The builder is committed by the SAME job before it renders, because the deck
# prints its own commit and that has to be the commit of the code that drew it.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
# viz + py-matplotlib are NOT optional: stage S's 51905177 committed the builders
# and then died 19 s later on `import matplotlib`, rendering nothing.
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists."
    exit 1
fi

echo
echo "=== compile ==========================================================="
python3 -m py_compile notebooks/make_stageu_web.py || exit 1
echo "ok"

echo
echo "=== commit the builder BEFORE rendering ==============================="
PATHS="notebooks/make_stageu_web.py
scripts/slurm/stageu_score_2026-11-11.sbatch
scripts/slurm/stageu_curve_2026-11-11.sbatch
scripts/slurm/stageu_web_2026-11-11.sbatch"
git add $PATHS || exit 1
staged=$(git diff --cached --name-only)
printf '%s\n' "$staged" | sed 's/^/  /'
if [ -n "$staged" ]; then
    while IFS= read -r p; do
        [ -z "$p" ] && continue
        printf '%s\n' "$staged" | grep -qx -- "$p" && continue
        # Not staged has two causes and only one is a fault: an IGNORED path
        # stages nothing and is a fault, an UNCHANGED path stages nothing and is
        # ordinary (stage S's 51922718 failed closed on an unchanged file).
        if git ls-files --error-unmatch -- "$p" >/dev/null 2>&1; then
            echo "  ($p unchanged since HEAD -- already committed)"
            continue
        fi
        echo "FAILING CLOSED: $p is not tracked (gitignored?) and did not stage."
        exit 1
    done <<< "$PATHS"
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
Stage U web deck: the gate negative, and the bimodality the median hid

Stage U asked whether the 0.017 between one shared block and the free line is
about ROLE COUNT: k distinct blocks, each applied 16/k times, every rung still
one map iterated. Array 51922710, six cells, scored by the registered scorer
(committed b72a74c before the first cell landed) in 52000549, with U4 closed by
52025112.

The registered gate prediction is FALSIFIED: k=2 reads 0.752 and k=4 reads 0.755
against the control's 0.757, both 3/3 crossing and both within 0.01 of it, which
is the competing hypothesis' prediction and not this stage's. The train NLL
floor is flat across k (span 0.0067 against the registered 0.01 band), so the
floor belongs to the recipe and not to the role count.

What the bound-aware median hides is the result worth publishing. One seed in
EACH k > 1 arm left the plateau -- k=2 seed 2 at rho_50 0.603 and k=4 seed 1 at
0.579, against their own arms' medians -- while no k=1 seed did in three tight
seeds (0.754 / 0.757 / 0.765). Both of those seeds sit ON the N=48
accuracy-to-solve curve (+0.0017 and -0.0025 against the reference's own
leave-one-out bar of 0.0423, both weakly determined and labelled so), so nothing
architectural happened: they reached J2-level token accuracy and collected the
solve rate the curve pays there. Both were also still descending four to ten
times faster than the plateau seeds when the fifty-epoch budget ended.

Nothing is re-derived. Levels, accuracies and the control come from the scorer's
own load_pooled / rung / summarize / acc_of; U4's residuals are recomputed by
calling scripts.acc_vs_solve's own collect / pava / predict / rms on exactly the
dir list score_u4() prints, and check_claims() asserts the bar it finds equals
the registered LOO_BAR. Only history.json is read straight off disk, for the
per-epoch learning curves and tail slopes, and its final-epoch train NLL median
is checked against the scorer's frozen S1_NLL_MEDIAN.

check_claims() also asserts the parts a reader has to trust: that exactly two
cells are in the basin and both are k > 1, that no arm cleared the gate, that
both basin residuals are inside the bar, and that no basin seed beat the
standing minimum by more than rounding -- which is what makes the last page's
"0.580 still stands" a measurement rather than a hedge.
MSG
    echo "commit rc=$?"
else
    echo "  (builder already committed)"
fi
SHA=$(git rev-parse --short HEAD)
echo "builder commit: $SHA"

echo
echo "=== render deck.md + figures =========================================="
python3 -u -m notebooks.make_stageu_web --commit "$SHA"
rc=$?
echo "render rc=$rc"
[ "$rc" -ne 0 ] && exit "$rc"

echo
ls -la /home/users/hshunt/slides/decks/spin-gnn/ \
       /home/users/hshunt/slides/decks/spin-gnn/figures/ 2>/dev/null
echo
echo "pages: $(grep -c '^---$' /home/users/hshunt/slides/decks/spin-gnn/deck.md)"
echo "end $(date '+%F %T')"
