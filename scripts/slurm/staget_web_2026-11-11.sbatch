#!/bin/bash
#SBATCH --job-name=stagetWeb
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagetWeb_%j.out
#
# STAGE T AS A WEB DECK -- render only, NO publish.
#
# ~/CLAUDE.md now makes a password-gated web deck at
# https://slides-hshunt.pages.dev/<project>/ the DEFAULT delivery and a PDF
# ask-only. Stage T went out as a PDF to OneDrive before that was read, so this
# renders the same reading into ~/slides/decks/spin-gnn/ where it belongs.
#
# Nothing is committed or pushed here. The read-back comes first: deck.md and
# every figure are read back before claude-slides-publish.sh runs, because a web
# deck is a URL and a wrong sentence on a URL is worse than one in a mailed PDF.
#
# In a job and not on the login node for two reasons: this imports torch through
# scripts.minarch_analysis and bootstraps 18 cells x 2000 resamples, and it
# writes into $HOME, where sh02-ln02's I/O wedges intermittently.
#
# The builder is committed by the SAME job before it renders, because the deck
# prints its own commit and that has to be the commit of the code that drew it
# (stage S's and stage T's two-commit pattern).

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
python3 -m py_compile notebooks/make_staget_web.py || exit 1
echo "ok"

echo
echo "=== commit the builder BEFORE rendering ==============================="
PATHS="notebooks/make_staget_web.py
scripts/slurm/staget_web_2026-11-11.sbatch"
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
Stage T web deck: the delivery ~/CLAUDE.md now asks for

A password-gated deck at slides-hshunt.pages.dev/spin-gnn/ is now the default
way results reach hshunt, and a PDF is ask-only. Stage T had already gone out as
a PDF, so this is the same reading rendered into ~/slides/decks/spin-gnn/.

Nothing is re-derived. Every figure and every number comes from
notebooks.make_staget_deck, which reads them through the registered scorer
(scripts/staget_score.py, committed by 51904633 before the first grid, scored by
51919538), and check_claims() runs here too -- so this builder fails closed on
exactly the two sentences the PDF's read-back corrected, rather than publishing
them to a URL:

  p5  "the easy end never reaches the 0.90 bar, at any T" -- its own figure
      refutes it; seed 2 sits exactly ON 0.900 at the trained count, so what
      fails is unanimity.
  p9  "all three seeds peak at 16" -- seed 2 peaks at T=20, by 0.0004.

Titles and notes are the PDF's and go through the shared library's own
_check_claim/_check_note/_check_points, so the web deck cannot drift past the
same 12-word title and 25-word body caps.

Two deliberate differences from the PDF. The comparison against the previous
best is LAST, which ~/CLAUDE.md requires and the PDF did not do (it ended on the
next cut). And the tokenization and trace pages get finding-sentence headings of
their own, because those figures already carry their in-figure titles and
repeating one as the slide heading wastes the line that makes the claim.
MSG
    echo "commit rc=$?"
else
    echo "  (builder already committed)"
fi
SHA=$(git rev-parse --short HEAD)
echo "builder commit: $SHA"

echo
echo "=== render deck.md + figures =========================================="
python3 -u -m notebooks.make_staget_web --commit "$SHA"
rc=$?
echo "render rc=$rc"
[ "$rc" -ne 0 ] && exit "$rc"

echo
ls -la /home/users/hshunt/slides/decks/spin-gnn/ \
       /home/users/hshunt/slides/decks/spin-gnn/figures/ 2>/dev/null
echo
echo "pages: $(grep -c '^---$' /home/users/hshunt/slides/decks/spin-gnn/deck.md)"
echo "end $(date '+%F %T')"
