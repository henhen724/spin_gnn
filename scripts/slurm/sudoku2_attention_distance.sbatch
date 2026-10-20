#!/bin/bash
#SBATCH -J sudoku2_attn_dist
#SBATCH -p sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH -o %x-%j.out

# Compares DD's required token-reference distance per deduction against the
# trained transformer's actual attention-weight decay with distance, over
# the best sudoku-dd hparam-sweep checkpoint. See
# spin_gnn/sudoku_attention_distance.py / sudoku_attention_distance_analysis.py.
# Usage: sbatch sudoku2_attention_distance.sbatch [extra args]

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
# Slurm stdout is block-buffered, so progress lines otherwise stay invisible
# for the whole run and cannot be polled.
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

CKPT_DIR="${SCRATCH}/spin_gnn/sudoku_hparam_sweep/d0064_h04_L12_ff01024_lr1e-03_pre"
OUT_DIR="${SCRATCH}/spin_gnn/sudoku_attention_distance/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_attention_distance_analysis \
    --checkpoint-dir "$CKPT_DIR" \
    --n-instances 300 \
    --seed-start 5000 \
    --out-dir "$OUT_DIR" \
    "$@"
