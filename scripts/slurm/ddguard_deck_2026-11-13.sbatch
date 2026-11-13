#!/bin/bash
#SBATCH --job-name=ddgDeck
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/ddgDeck_%j.out
#
# DRAFT RENDER of decks/spin-gnn/guard. Writes to $SCRATCH, NOT to ~/slides, and
# stamps --commit draft, so a builder nobody has read back yet cannot be what a
# commit contains (LOG 2026-11-12). The real render is a second job, after the
# figures have been read.
#
# NO GPU: the builder loads no model. It reads the twelve per-case CSVs job
# 52211808 already wrote and generates one example instance on CPU for the shared
# tokenization and trace figures.
#
# Steps fail closed in order -- lint before render, render before anything is
# printed -- because the expensive failure mode here is not a crash, it is a deck
# that renders clean and says something false.

set -uo pipefail
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
date +'start %F %T'

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

DRAFT=/scratch/users/hshunt/spin_gnn/runs/2026-11-13_ddguard/deck_draft

echo
echo "=== 1. page-cap lint (claims 12, notes 25, bullets 14, 3 per points) ==="
python3 scripts/deck_lint.py notebooks/make_ddguard_web.py || {
    echo "LINT FAILED -- nothing rendered"; exit 1; }

echo
echo "=== 2. draft render ==============================================="
# check_claims inside the builder asserts every rate it prints against
# ddguard_summary.json, which the scorer wrote inside job 52211808. If the two
# modules ever disagree the render raises instead of publishing a deck that
# argues with its own appendix.
python3 -u notebooks/make_ddguard_web.py --out "$DRAFT" --commit draft \
    --job 52211808 || { echo "RENDER FAILED"; exit 1; }

echo
echo "=== 3. figures written ============================================"
ls -o "$DRAFT/figures" | sed 's/^/  /'

echo
echo "=== 4. the deck source, for the prose read-back ==================="
cat "$DRAFT/deck.md"

date +'end %F %T'
