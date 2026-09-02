#!/bin/bash
#SBATCH -J sat_notebook
#SBATCH -p sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH -o %x-%j.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Executes notebooks/sat_unit_propagation_results.ipynb in place, refreshing
# every cell's outputs (test suite run, sweep aggregation/plots, checkpoint
# load + generation) against the current state of $SCRATCH/spin_gnn.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
# The module tree's own (older) typing_extensions otherwise shadows the
# venv's newer one ahead of it on sys.path, breaking jupyter_client's import.
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "${SLURM_SUBMIT_DIR}"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  notebooks/sat_unit_propagation_results.ipynb
