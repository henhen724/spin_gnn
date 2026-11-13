#!/bin/bash
#SBATCH --job-name=dissocScore
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/dissocScore_%j.out
#
# RETRY of the SCORING half of job 52224240, which failed at 00:01:14 with
#   ValueError: could not convert string to float: 'True'
# after all six measurement cells had already succeeded. Every CSV cell is a
# string and the boolean columns arrive as 'True', so fmean's float() raised on
# the first boolean column it was handed. Fixed by coercing through the module's
# own truthy() instead; the measurement is NOT re-run, because nothing about it
# failed and re-running it would change nothing except the electricity bill.
#
# NO GPU for the same reason: the six per-case CSVs already exist, and this job
# only reads them.
#
# Also fixes a second defect the failure exposed before it crashed. H5 compared
# this run's UNCONDITIONAL mean against finding 68's CONDITIONAL published rate --
# the exact denominator mistake the guard deck's read-back caught on 2026-11-13.
# At J2 s2 that is 0.6571 against 0.8444, which would have read as a reproduction
# failure when both numbers are right and only the denominators differed. H5 now
# computes broken | clean with ddguard_score's own ratio(), and both rates are
# written to the summary so a figure can never mix them again.
#
# Scoring and the draft render are one job: each is seconds, and Scheduler
# etiquette asks short tasks to be aggregated.

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

RUN=/scratch/users/hshunt/spin_gnn/runs/2026-11-13_dissoc
DRAFT=$RUN/deck_draft

echo
echo "=== 0. the six cells are already on disk =========================="
for d in "$RUN"/S1_s* "$RUN"/J2_s*; do
    printf '  %s  %s rows\n' "$(basename "$d")" "$(($(wc -l < "$d/dissoc_behavior.csv") - 1))"
done

echo
echo "=== 1. score ======================================================"
python3 -u scripts/dissoc_score.py || { echo "SCORING FAILED AGAIN"; exit 1; }

echo
echo "=== 2. page-cap lint =============================================="
python3 scripts/deck_lint.py notebooks/make_dissoc_web.py || {
    echo "LINT FAILED -- nothing rendered"; exit 1; }

echo
echo "=== 3. draft render to \$SCRATCH ==================================="
python3 -u notebooks/make_dissoc_web.py --out "$DRAFT" --commit draft \
    --job 52224240 || { echo "RENDER FAILED"; exit 1; }

echo
echo "=== 4. figures written ============================================"
ls -o "$DRAFT/figures" | sed 's/^/  /'

echo
echo "=== 5. the deck source, for the prose read-back ==================="
cat "$DRAFT/deck.md"

date +'end %F %T'
