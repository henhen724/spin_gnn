#!/bin/bash
#SBATCH -J sat_notebook
#SBATCH -p sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH -o %x-%j.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Executes a notebook in place, refreshing every cell's output against the
# current state of $SCRATCH/spin_gnn. Pass the notebook path as $1;
# defaults to sat_unit_propagation_results.ipynb for backward compatibility.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 libjpeg-turbo/3.1.2
source "${SCRATCH}/spin_gnn_venv/bin/activate"
# The module tree's own (older) typing_extensions otherwise shadows the
# venv's newer one ahead of it on sys.path, breaking jupyter_client's import.
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "${SLURM_SUBMIT_DIR}"
NOTEBOOK="${1:-notebooks/sat_unit_propagation_results.ipynb}"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  "$NOTEBOOK"
