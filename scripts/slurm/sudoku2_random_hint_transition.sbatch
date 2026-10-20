#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_rhint
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# First transformer-vs-DD hint-density curve. Trains on INDEPENDENT RANDOM
# hints (--hint-mode random), the only distribution in which a solvability
# transition exists at all: the default removal-path protocol accepts a hint
# removal only if DD still recovers the whole coloring, so every instance is
# resolvable at every rho and no threshold can be observed.
#
# Under random hints the model must also learn to CALL unsolvability, so
# outcome_accuracy(rho) and the model's solve_rate(rho) can be read directly
# against dd_resolved_frac(rho)/dd_frac_recovered(rho), which
# eval_solve_by_rho now measures on the very same instances in each bin.
#
# Caveat recorded up front: a pilot (scripts/dd_random_hint_transition.py)
# shows the DD cascade at N=24 peaks at only ~1.7 rounds and full recovery
# lifts off only above rho~0.7, so the N=24 transition is heavily finite-size
# smeared. This run is therefore the METHOD at a size already known to train,
# not the final physics; Slurm 48579017 maps N=24..384 to pick the size where
# the transition is actually sharp.
#
# Config matches the current best (Slurm 48513926_1: d_model=128, nhead=4,
# L=12, ff=1024, 900 epochs) so the only change vs. that baseline is the hint
# distribution. Plain grammar, not --scratchpad: that is a separate
# one-variable test running as 48579019.

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

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_random_hint_transition_2026-10-19"
mkdir -p "$OUT_DIR"

# 21-point rho grid at 80 instances/bin = 1680 free generations for the final
# checkpoint; the 900-epoch baseline needed 46 min for training plus ~800
# generations, so 5h is comfortable.
python -m spin_gnn.sudoku_transition_sweep \
    --n-nodes 24 \
    --degree 5 \
    --n-colors 4 \
    --rho 0.10 \
    --rho-max 1.0 \
    --hint-mode random \
    --d-model 128 \
    --nhead 4 \
    --num-layers 12 \
    --dim-feedforward 1024 \
    --n-train-instances 5000 \
    --n-val-instances 300 \
    --batch-size 32 \
    --epochs 900 \
    --checkpoint-epochs 900 \
    --rho-grid 0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00 \
    --n-per-bin 80 \
    --out-csv "${OUT_DIR}/transition.csv" \
    "$@"
