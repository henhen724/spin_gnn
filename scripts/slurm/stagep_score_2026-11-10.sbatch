#!/bin/bash
#SBATCH --job-name=stagepScore
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagepScore_%j.out
#
# Score stage P with its registered scorer, once the array has finished.
#
# WHY A DEPENDENCY AND NOT A POLL. Scheduler etiquette: a job that waits on
# another job's completion is what --dependency is for, and the array runs ~6 h.
# This job fires on its own and the session does not watch it.
#
# WHY afterany AND NOT afterok. A cell that crashes or hits its time limit is a
# RESULT this stage has to read, not a reason to print nothing: P2 is about
# cells that fail to train, and a partial array still answers P3a. sacct is
# printed first so the exit states are in the same log as the verdicts.
#
# This job deliberately does NO deck, NO notebook, NO commit, NO upload and NO
# STATUS.md edit. Those are written after a human has read the verdicts, and the
# stage N write-up chain is the pattern to copy.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

ARRAY=51815311

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
echo
echo "=== how the array exited (array $ARRAY) ================================="
sacct -j "$ARRAY" -X --format=JobID%16,JobName%16,State,ExitCode,Elapsed,MaxRSS

echo
echo "=== the registered scorer =============================================="
python3 -u -m scripts.stagep_score
echo "scorer rc=$?"

echo
echo "end $(date '+%F %T')"
