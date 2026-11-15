#!/bin/bash
#
# Render decks/spin-gnn/counterfactual from stage Z's artifacts.
#
# A job rather than a login-node run because the builder imports torch through
# deck_tokenization, which generates the real example instance and its trace,
# and because every other deck in this project was rendered the same way.
#
# NOTHING IS MEASURED HERE and nothing is published. The job renders PNGs and
# deck.md into ~/slides/decks/spin-gnn/counterfactual/, then checks every PNG
# it wrote is a readable, non-trivial image, and stops. The build and the
# publish are done by hand afterwards, because claude-slides-publish.sh pushes
# -- and the push IS the deploy -- so it never belongs inside a batch job.
#
# No GPU: the builder loads no checkpoint. All three seeds' numbers were
# measured by 52413604 and are read from JSON and CSV.
#
#SBATCH --job-name=stagezDeck
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache

OUT=/home/users/hshunt/slides/decks/spin-gnn/counterfactual
COMMIT=e87bf56

echo
echo "=== 1. static cap lint ==========================================="
python3 scripts/deck_lint.py notebooks/make_stagez_web.py || {
    echo "LINT FAILED -- a page would die inside slides.py"; exit 1; }

echo
echo "=== 2. render ===================================================="
python3 -u -m notebooks.make_stagez_web --commit "$COMMIT" --job 52413604 \
    --out "$OUT"
RC=$?
echo "render rc=$RC"
[ $RC -ne 0 ] && { echo "render FAILED -- nothing to publish"; exit $RC; }

echo
echo "=== 3. every PNG is a readable image ============================="
python3 - "$OUT" <<'PY'
import sys, os, glob
from PIL import Image
d = sys.argv[1]
pngs = sorted(glob.glob(os.path.join(d, "figures", "*.png")))
if not pngs:
    sys.exit("no PNGs written")
bad = []
for p in pngs:
    kb = os.path.getsize(p) / 1024.0
    try:
        im = Image.open(p); im.load()
        w, h = im.size
    except Exception as e:
        bad.append("%s unreadable: %s" % (p, e)); continue
    print("  %-28s %5.0f kB  %dx%d" % (os.path.basename(p), kb, w, h))
    # A figure that rendered empty still writes a file, so size is checked too.
    if kb < 15 or w < 800 or h < 400:
        bad.append("%s looks degenerate (%.0f kB, %dx%d)" % (p, kb, w, h))
if bad:
    sys.exit("\n".join(bad))
print("%d PNGs ok" % len(pngs))
PY
echo "png rc=$?"

echo
echo "=== 4. deck.md shape ============================================="
# KaTeX renders in the browser and nothing on Sherlock can open the deck, so a
# broken equation stays invisible until hshunt does. tools/build.sh warns about
# these; this catches them a step earlier, with exact literals rather than a
# grep pattern whose own backslashes have to be guessed at.
python3 - "$OUT/deck.md" <<'PY'
import sys
md = open(sys.argv[1]).read()
print("pages: %d" % (md.count("\n---\n") + 1))
print("figures referenced: %d" % md.count("![](figures/"))
bad = [t for t in ("<<FIGURE>>", "\\bm", "\\mathscr", "\\mathbbm",
                   "\\DeclareMathOperator", "\\eqref", "\\(", "\\[")
       if t in md]
print("unsupported or unreplaced: %s" % (bad if bad else "none"))
sys.exit(1 if bad else 0)
PY
echo "md rc=$?"

echo "end $(date '+%F %T')"
exit $RC
