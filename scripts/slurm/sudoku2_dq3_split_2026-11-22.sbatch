#!/bin/bash
#SBATCH --job-name=sud2DQ3split
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/users/hshunt/spin_gnn/runs/stagedq3_2026-11-22/slurm-%A_%a.out
#
# STAGE DQ3 -- the per-section split for the seed-2 cells. Inference only,
# CPU, no training, no checkpoint written.
#
#   sbatch --dependency=afterany:53313361 \
#       scripts/slurm/sudoku2_dq3_split_2026-11-22.sbatch
#
# WHY THIS JOB EXISTS. 53313361 produces the density sweep (E5) and the
# trainability read but NO per-section NLL -- and E1, E2, E3 and E4 are all
# per-section numbers, so four of the five predictions have no data source
# without this. Same split of labour as stage DQ (53243550 then 53264789).
#
# afterANY, NOT afterok, for the reason stage DQ documented: a dead cell under
# afterok parks this forever as DependencyNeverSatisfied with nobody told,
# whereas afterany measures what landed, writes MISSING.txt, and the scorer
# refuses any prediction whose cells are incomplete.
#
# THE SCORERS ARE WRAPPERS, NOT COPIES, AND THAT IS DELIBERATE.
# scripts/stagedq_probe.py and scripts/stagedq_fit.py produced stage DQ's
# published VERDICT.txt. Editing either to accept a second directory would
# mean the code that produced that verdict is no longer the code in the tree.
# So scripts/stagedq3_probe.py rebinds three module globals (SCAN_DQ, OUT,
# CELLS) and calls the SAME main(), and scripts/stagedq3_fit.py imports the
# readers, the Clopper-Pearson helpers and the pooling and passes explicit
# globs. Both carry assertions that the redirect MOVED -- a no-op patch would
# silently re-probe seeds 0 and 1 and write a file that looks like a DQ3
# result.
#
# THE TWO INHERITED GUARDS STILL APPLY, which is the other reason to import
# rather than re-type: load_curves REFUSES a curve whose (degree, n_colors)
# is not (4, 3), and load_sections REFUSES a row whose vocabulary is not
# N + 6 + 3. A q=4 checkpoint cannot be scored against ln 3 by accident.
#
# DQ-P0 and DQ-P1 RUN AGAIN HERE, before a single DQ3 checkpoint is touched,
# because main() calls them first. DQ-P1 reproduces an already published
# NSPLIT CSV on 20 quantities; it passed at 5.0e-08 when this wrapper was
# written on 2026-11-22, which is what proves the redirect rather than
# assuming it.
#
# E1-E5 ARE REGISTERED IN runs/stagedq3_2026-11-22/REGISTERED.md at commit
# 5af3d21, BEFORE 53313361 was submitted. The scorers were dry-run on stage
# DQ's ALREADY SCORED cells before this job was written, which exercised every
# branch: seeds (96,0)+(120,0) gave E1 MISS / E3 HIT / E4 HIT, seeds
# (96,1)+(120,1) gave E1 HIT, and E5 correctly refused to claim an
# improvement over a value equal to the standing 0.99191.
#
# DENSITY GRID unchanged from stage DQ: [0.90 0.70 0.50 0.30 0.18]. Changing
# it would break the per-cell comparison this stage exists to make.
#
set -u
module purge --force
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312

OUT=/scratch/users/hshunt/spin_gnn/runs/stagedq3_2026-11-22
mkdir -p "$OUT"
cd /home/users/hshunt/spin_gnn

echo "=== commit ==="
git log --oneline -1
hostname
date +'start %F %T'

python3 -u scripts/stagedq3_probe.py \
    --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --ntasks "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --out "$OUT" ${DQ3_SPLIT_ARGS:-}

date +'end   %F %T'

# LAST ONE OUT WRITES THE VERDICT, as in stage DQ. Every task tries; only the
# task that finds all the sibling CSVs already on disk goes ahead, so whichever
# finishes last produces a COMPLETE report and the earlier ones produce none.
# Scoring unconditionally would emit one partial VERDICT.txt and one correct
# one, with nothing in the file saying which was which.
if [ -z "${DQ3_SPLIT_ARGS:-}" ]; then
    NCSV=$(ls "$OUT"/dqsplit_*.csv 2>/dev/null | wc -l)
    NTASK=${SLURM_ARRAY_TASK_COUNT:-1}
    if [ "$NCSV" -ge "$NTASK" ]; then
        echo
        echo "=== all $NCSV/$NTASK section CSVs present: scoring E1-E5 ==="
        python3 -u scripts/stagedq3_fit.py --out "$OUT/VERDICT.txt" || true
    else
        echo
        echo "$NCSV/$NTASK section CSVs present; a later task will score."
    fi
fi
