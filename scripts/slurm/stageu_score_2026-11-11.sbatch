#!/bin/bash
#SBATCH --job-name=stageuScore
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stageuScore_%j.out
#
# Score stage U with its registered scorer (scripts/stageu_score.py, committed as
# b72a74c by job 51922709 BEFORE the first cell landed). Same shape as
# stages_score_2026-11-11.sbatch.
#
# ONE read, not two: all six cells exited COMPLETED 0:0 at about 8h06-8h09, so
# unlike stage S there is no partial-arm problem to read around. The cell count
# is still printed and checked, because a cell can have trained without having
# been evaluated on the grid, and the scorer reads the CURVES.
#
# NO deck, NO commit, NO publish, NO STATUS.md edit here. Those come after the
# verdicts have been read; stage N's write-up chain is the pattern.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

ARRAY=51922710
SCAN=/scratch/users/hshunt/spin_gnn/sudoku_n48_cycled_U_2026-11-11

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
echo
echo "=== how the array exited (array $ARRAY) ================================="
sacct -j "$ARRAY" -X --format=JobID%16,JobName%16,State,ExitCode,Elapsed,MaxRSS

echo
echo "=== which cells have a rho curve on disk ================================"
ls -1 "$SCAN"/rho_curve_*.csv 2>/dev/null | sed "s|$SCAN/||"
n=$(ls -1 "$SCAN"/rho_curve_*.csv 2>/dev/null | wc -l)
echo "found $n of 6 cells"
if [ "$n" -lt 6 ]; then
    echo "PARTIAL READ: fewer than six curves, so U3's ordering is not callable"
    echo "(nothing is called until BOTH new arms land -- the stage O mistake)."
fi

echo
echo "=== the registered scorer =============================================="
python3 -u -m scripts.stageu_score
echo "scorer rc=$?"

echo
echo "end $(date '+%F %T')"
