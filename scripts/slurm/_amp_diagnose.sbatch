#!/bin/bash
#SBATCH -J spin_gnn_amp_diag
#SBATCH -p sganguli
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out
set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
python scripts/_amp_diagnose.py
