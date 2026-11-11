#!/bin/bash
#SBATCH --job-name=stagesScore
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagesScore_%j.out
#
# Score stage S with its registered scorer (scripts/stages_score.py, committed
# before the first cell landed). Same shape as stagep_score_2026-11-10.sbatch.
#
# SUBMITTED TWICE, ON PURPOSE, AND THE TWO READS ARE NOT THE SAME READ:
#   once bare, while the three d448 cells are still running -- a PARTIAL read
#     that can only speak to the d320 trend point and to S2 (accuracy), and
#     says so by printing how many cells it found;
#   once with --dependency=afterany:51853969 -- the COMPLETE read, which is the
#     one that may be quoted as stage S's result.
# afterany and not afterok because two d448 seeds are sitting at the uniform
# predictor: how a cell exits is a result this stage has to read, not a reason
# to print nothing.
#
# NO deck, NO notebook, NO commit, NO upload, NO STATUS.md edit here. Those come
# after a human has read the verdicts; stage N's write-up chain is the pattern.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

ARRAY=51853969
SCAN=/scratch/users/hshunt/spin_gnn/sudoku_n48_tiedwidth_S_2026-11-10

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
echo
echo "=== how the array exited (array $ARRAY) ================================="
sacct -j "$ARRAY" -X --format=JobID%16,JobName%16,State,ExitCode,Elapsed,MaxRSS

echo
echo "=== which cells have a rho curve on disk ================================"
# A curve per cell is what the scorer actually reads, so count those and not the
# checkpoints: a cell can have trained and not yet been evaluated on the grid.
ls -1 "$SCAN"/rho_curve_*.csv 2>/dev/null | sed "s|$SCAN/||"
n=$(ls -1 "$SCAN"/rho_curve_*.csv 2>/dev/null | wc -l)
echo "found $n of 6 cells"
if [ "$n" -lt 6 ]; then
    echo "PARTIAL READ: the d448 arm is incomplete, so nothing below may be"
    echo "quoted as stage S's result. S3 already says d320 cannot be called on"
    echo "its own for rho_50 (613,818 is below H1's 879,418 censoring floor)."
fi

echo
echo "=== the registered scorer =============================================="
python3 -u -m scripts.stages_score
echo "scorer rc=$?"

echo
echo "end $(date '+%F %T')"
