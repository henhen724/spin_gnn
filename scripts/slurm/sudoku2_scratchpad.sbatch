#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_sp
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# Hypothesis: the binding constraint on agg_solve_rate (stuck at 0.72-0.77
# across every lever tried -- nhead, d_model, 3x training budget) is
# long-range RETRIEVAL. A deduction needs its node's neighbor colors, which
# live back in BEGIN_GRAPH a median 179 tokens earlier, and the measured
# attention-targeting enrichment on exactly that token is only 1.13-1.26x a
# distance-matched baseline, i.e. barely better than attending by recency.
#
# The scratchpad grammar restates each deduction's evidence (the neighbors'
# colors known BEFORE that round) immediately before the deduction, making
# every step locally determined: short-range copy + a local "which color is
# missing" step. Prediction: agg_solve_rate rises above 0.767. If it does
# not, retrieval is not the constraint.
#
# This also gates the scale-up that the hint-density transition needs: the
# transition only appears at much larger N (at N=24 the DD cascade peaks at
# 1.7 rounds), which means ~11*N tokens -- if retrieval already fails at 210
# tokens it is hopeless at 1700, and the scratchpad is the fix that keeps
# deductions local as N grows.
#
# ONE variable vs. the current best (Slurm 48513926_1: d_model=128, nhead=4,
# L=12, ff=1024, 900 epochs, agg_solve_rate=0.767, mean_o1=0.968) -- same
# config, same seed, same eval, only --scratchpad added. That run is the
# control; no rerun needed. New dated out-dir because the checkpoint tag
# encodes neither the epoch count nor (before now) the scratchpad flag.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_hparam_sweep_2026-10-19_scratchpad"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 24 \
    --degree 5 \
    --n-colors 4 \
    --d-model 128 \
    --nhead 4 \
    --num-layers 12 \
    --dim-feedforward 1024 \
    --n-train-instances 5000 \
    --n-val-instances 300 \
    --batch-size 32 \
    --epochs 900 \
    --n-per-bin 100 \
    --scratchpad \
    --out-dir "$OUT_DIR" \
    "$@"
