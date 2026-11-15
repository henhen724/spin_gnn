#!/bin/bash
#
# Localise why 52423780_[0-2] sat at uniform output. See the docstring of
# scripts/stagepair_diagnose.py for the observation, what is already ruled out
# and the hypothesis. NOT A RESULT: nothing here is read as a measurement.
#
# Two arms, one GPU, from one init and one data order. ~900 steps each at
# 76.55 ms (control) and 151.43 ms (pair) per the 52415691 pricing is about
# 3.5 minutes of stepping; instance generation for 20,000 dominates.
#
#SBATCH --job-name=stagepairDiag
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -euo pipefail
cd /home/users/hshunt/spin_gnn

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

python3 scripts/stagepair_diagnose.py

echo
echo "end $(date '+%F %T')"
