#!/bin/bash
#SBATCH -J sudoku2_attn_evo
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# Retrains the best sudoku-dd hparam-sweep config (d_model=64, nhead=4,
# num_layers=12, dim_feedforward=1024, 300 epochs) with checkpoint
# snapshots, and tracks spin_gnn.sudoku_attention_distance's attention-
# targeting enrichment metric at each snapshot -- is the attention head's
# targeting of DD-relevant tokens still improving over training, or stuck
# near its untrained (~1.0x baseline) value? See sudoku_attention_evolution.py.
# Usage: sbatch sudoku2_attention_evolution.sbatch [extra args]

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_attention_evolution/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_attention_evolution \
    --d-model 64 \
    --nhead 4 \
    --num-layers 12 \
    --dim-feedforward 1024 \
    --n-train-instances 5000 \
    --n-val-instances 300 \
    --batch-size 32 \
    --epochs 300 \
    --checkpoint-epochs 0 5 10 25 50 75 100 150 200 250 300 \
    --out-dir "$OUT_DIR" \
    "$@"
