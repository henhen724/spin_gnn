#!/bin/bash
#SBATCH -J sat2_step_acc
#SBATCH -p sganguli
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH -o %x-%j.out

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "${SLURM_SUBMIT_DIR}"
python scripts/_sat2_step_accuracy_probe.py
