#!/bin/bash
#SBATCH -J spin_gnn_amp
#SBATCH -p sganguli
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# AMP teacher-student learning curves (spin_gnn/amp_sweep.py), CPU only.
# Runs tests/test_amp.py first so a broken denoiser fails fast.
# Usage: sbatch scripts/slurm/amp_sweep.sbatch [extra args to amp_sweep.py]

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python -m pytest tests/test_amp.py -q
python -m spin_gnn.amp_sweep --device cpu --out "amp_results/sweep_${SLURM_JOB_ID}.csv" "$@"
