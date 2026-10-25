#!/bin/bash
#SBATCH --job-name=sud2n96Lscan
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=36:00:00
#SBATCH --output=%x_%j.out
#
# The N=96 arm. DO NOT SUBMIT until BOTH the N=24 scan (49433454) and the N=48
# arm have reported. This is the most expensive job in the project and it only
# earns its cost if the two cheaper arms disagree about L*(T) -- see below.
#
# WHY IT MIGHT NOT BE WORTH RUNNING AT ALL. The cross-N comparison needs at
# least two points, and N=24 (mean DD depth 4.25) plus N=48 (6.10) already
# provide them. N=96 adds 7.45, a 22% increase in mean depth over N=48 for
# ~4x the compute per instance. The case for spending that is narrow:
#   * If N=24 and N=48 agree that L*(T) is FLAT, finding 2 is confirmed twice
#     and a third point buys nothing. Do not run this.
#   * If they agree that L*(T) RISES with T, the slope is the quantity of
#     interest and a third N pins whether it is linear in T (one layer per
#     deduction round) or sublinear. Then run this.
#   * If they DISAGREE -- flat at N=24, rising at N=48 -- the effect only
#     appears once there is depth to resolve, and N=96 is the test of that.
#     Then run this, and it is the most informative job in the project.
# Writing the script now rather than later is free; submitting it is not.
#
# L GRID. {2, 4, 8, 12} only -- four points, a strict subset of both other
# arms. L=1 is dropped: it fails at 0.067 at N=24, and there is no reason to
# spend 2 GPU-h confirming that one layer also fails on a harder problem.
# L=3 and L=6 are dropped because at N=96 each costs ~2-5 GPU-h and the knee
# they would resolve will already be located by the N=24 scan (which puts it
# sharply at L=3: 0.067 -> 0.467 -> 0.957 -> 0.977 for L=1,2,3,4).
#
# EVERYTHING ELSE PINNED: d_model=256, nhead=4, ff=1024 (MLP width constant,
# only depth varies), lr=1e-3, pre-norm, same seed. n_train=10000 and 100
# epochs, both reduced: sequences are ~787 tokens at N=96 against 402 at N=48
# and 204 at N=24, so per-instance cost is ~4x N=48 and ~15x N=24. This is a
# real confound against the other two arms and is NOT hidden: the L=12 cell is
# the control that measures it, and if L=12 at N=96 comes in far below the
# N=48 L=12 cell, the comparison is about budget as much as about N. Say so
# rather than reporting an L*(T) curve as if the three arms were matched.
#
# PRE-REGISTERED PREDICTIONS -- to be written before submission, once the two
# cheaper arms have reported. Leaving them blank is deliberate: the whole
# point of gating this job is that what it should predict depends on what
# N=24 and N=48 say, and pre-registering now would just be guessing twice.
#
# Cost: measured 0.058 GPU-h per layer-unit per 150 epochs at N=24. At N=96,
# ~15x per instance but 1/2 the instances and 2/3 the epochs: ~0.29 h per
# layer-unit. 26 layer-units is ~7.5 h, plus four data builds at 10000
# instances (the N=96 removal paths are ~1.5 ms each, so minutes) and four
# depth evals with 787-token autoregressive generation, which is the real
# unknown -- at N=24 a 600-instance depth eval took minutes, and generation
# cost scales with sequence length times sequence length. 36:00:00 is sized
# for the evals being 20x dearer than the N=24 ones, not for the training.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_layer_scan_2026-10-24_n96"
mkdir -p "$OUT_DIR"

for L in 2 4 8 12; do
    echo "=== N=96 num_layers=${L} ==="
    python -m spin_gnn.sudoku_hparam_sweep \
        --n-nodes 96 \
        --degree 5 \
        --n-colors 4 \
        --d-model 256 \
        --nhead 4 \
        --num-layers "$L" \
        --dim-feedforward 1024 \
        --n-train-instances 10000 \
        --n-val-instances 300 \
        --batch-size 16 \
        --epochs 100 \
        --n-per-bin 100 \
        --out-dir "$OUT_DIR"
done

echo
echo "=== solve_rate by DD depth at fixed density, all num_layers ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${OUT_DIR}"/d0256_h04_L*_ff01024_lr1e-03_pre \
    --rho 0.75 0.60 \
    --n-pool 400 \
    --min-per-bin 30 \
    --out-csv "${OUT_DIR}/depth_by_layers.csv"
