#!/bin/bash
#SBATCH -J sat2_smoke
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "${SLURM_SUBMIT_DIR}"

python -m spin_gnn.sat2_hparam_sweep \
    --n-vars 16 --alpha 1.25 --d-model 64 --nhead 4 --num-layers 2 --lr 1e-3 \
    --n-train-instances 3000 --n-val-instances 300 --batch-size 64 --epochs 10 \
    --out-dir "${SCRATCH}/spin_gnn/_sat2_smoke"
