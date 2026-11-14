#!/bin/bash
#SBATCH --job-name=lensDeck
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/lensDeck_%j.out
#
# Render the finding-72 deck: draft into $SCRATCH by default, into ~/slides
# when OUT and COMMIT are exported (see below).
#
# In a job rather than on a login node because the builder imports torch (the
# shared tokenization figure needs a real instance and a real vocabulary) and
# because $HOME I/O on the login nodes has wedged before. The draft goes to
# $SCRATCH so nothing in the shared slides repo is touched until the prose and
# every figure have been read back twice.
#
# Lint first: deck_lint.py resolves conditional and dict-lookup titles as well
# as literals, which it learned to do after an invisible 13-word title killed
# a render job on 2026-11-13.

set -uo pipefail
cd /scratch/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
date +'start %F %T'

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# Draft by default. The final render passes the real destination and the real
# commit through the environment --
#   sbatch --export=ALL,OUT=$HOME/slides/decks/spin-gnn/localisation,COMMIT=<sha>
# -- so the deck that gets published is produced by this same lint-then-render
# path rather than by a one-off command typed on a login node.
OUT=${OUT:-/scratch/users/hshunt/spin_gnn/deckdraft/localisation}
COMMIT=${COMMIT:-draft}

echo
echo "=== lint ==========================================================="
python3 scripts/deck_lint.py notebooks/make_lens_web.py || {
    echo "LINT FAILED -- nothing rendered"; exit 1; }

echo
echo "=== render draft ==================================================="
mkdir -p "$OUT"
python3 -u -m notebooks.make_lens_web --out "$OUT" \
    --commit "$COMMIT" --job 52271112 || { echo "RENDER FAILED"; exit 1; }

echo
echo "=== pages =========================================================="
grep -c '^---$' "$OUT/deck.md"
ls -1 "$OUT/figures"

date +'end %F %T'
