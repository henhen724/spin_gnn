#!/bin/bash
#SBATCH -J spin_gnn_test
#SBATCH -p normal
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH -o %x-%j.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Runs the test suite. Cheap CPU-only sanity check -- run this after any
# change to model.py/graph.py/simulate.py before trusting a training result.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
python -m pytest tests/ -q
