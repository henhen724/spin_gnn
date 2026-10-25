#!/bin/bash
#SBATCH --job-name=sud2n48Lscan
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#
# The N=48 arm of the num_layers scan. DO NOT SUBMIT until 49433453 has
# reported: it is the gate. If that baseline comes in below 0.70 the recipe
# did not transfer and this scan would be measuring an optimization failure
# rather than a depth requirement.
#
# THE CROSS-N TEST. The N=24 arm (49433454) asks whether L*(T) -- the smallest
# num_layers reaching >= 0.90 on depth-T instances -- grows with T. Within one
# N that question has little room: mean DD depth at N=24 is 4.25 and the max
# is 8. N=48 gives mean 6.10 and max 14. So the discriminating comparison is
# between the two arms, not inside either:
#
#   serial hypothesis   L*(T) rises with T at BOTH N, and the N=48 curve
#                       extends the N=24 one rather than sitting above it --
#                       depth is a function of T, not of N.
#   finding 2           L*(T) is flat at both N. Whatever the model does, it
#                       does not spend one layer per deduction round, and the
#                       extra num_layers buy accuracy uniformly across T.
#   third possibility   L*(T) flat in T but higher at N=48 than at N=24. Then
#                       what depth buys is graph size, not chain length, and
#                       "reasoning depth" is the wrong axis entirely.
#
# L GRID. {1, 2, 4, 8, 12} -- a strict subset of the N=24 grid
# {1,2,3,4,6,8,12} so the two arms are directly comparable at five shared
# points. 3 and 6 are dropped because they only refine a region the N=24 arm
# will already have resolved, and each layer-unit costs 4x more here.
#
# EVERYTHING ELSE PINNED at the best recipe: d_model=256, nhead=4, ff=1024
# (MLP width constant, only depth varies), lr=1e-3, pre-norm, 150 epochs, same
# seed. n_train=20000 matches 49433453 so the baseline doubles as this scan's
# L=12 cross-check -- if d0256/L12 here disagrees with 49433453 beyond its
# Wilson interval, something other than num_layers is moving.
#
# PRE-REGISTERED PREDICTIONS.
#  (a) L=12 reproduces 49433453's aggregate solve_rate within its Wilson
#      interval despite 150 vs 300 epochs. Falsifier: it does not, in which
#      case 150 epochs is undertrained at N=48 and the whole scan is confounded
#      with epoch budget -- report that and stop rather than interpret L*(T).
#  (b) The N=24 verdict on L*(T) reproduces here in SIGN. Whatever 49433454
#      says about monotonicity, N=48 says the same, because a mechanism that
#      spends layers on deduction rounds should not switch off at one size.
#  (c) L*(T) at N=48 is >= L*(T) at N=24 at every shared T. Larger graphs are
#      harder at fixed chain length, so this is the weak, almost-guaranteed
#      half of the third possibility above; it is listed to make the
#      interesting part -- whether the GRADIENT in T differs -- falsifiable
#      separately from the offset.
#  (d) L=1 fails everywhere at N=48 too (< 0.5 at T >= 2).
#
# Cost: measured at N=24, ~0.058 GPU-h per layer-unit per 150 epochs; N=48 is
# ~4x per instance (2x sequence, quadratic attention), so ~0.23 h per unit.
# 27 layer-units is ~6.2 h, plus five data builds at 20000 instances and five
# depth evals: ~9 h. 20:00:00 leaves room, since the N=48 cost multiplier is
# extrapolated rather than measured.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_layer_scan_2026-10-24_n48"
mkdir -p "$OUT_DIR"

for L in 1 2 4 8 12; do
    echo "=== N=48 num_layers=${L} ==="
    python -m spin_gnn.sudoku_hparam_sweep \
        --n-nodes 48 \
        --degree 5 \
        --n-colors 4 \
        --d-model 256 \
        --nhead 4 \
        --num-layers "$L" \
        --dim-feedforward 1024 \
        --n-train-instances 20000 \
        --n-val-instances 300 \
        --batch-size 32 \
        --epochs 150 \
        --n-per-bin 100 \
        --out-dir "$OUT_DIR"
done

echo
echo "=== solve_rate by DD depth at fixed density, all num_layers ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${OUT_DIR}"/d0256_h04_L*_ff01024_lr1e-03_pre \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT_DIR}/depth_by_layers.csv"
