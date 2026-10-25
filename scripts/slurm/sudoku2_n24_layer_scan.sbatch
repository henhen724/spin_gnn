#!/bin/bash
#SBATCH --job-name=sud2n24Lscan
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#
# Does network depth have to grow with reasoning depth?
#
# THE QUESTION. DD solves an instance in T rounds, each round using the
# colors fixed by the previous one. If the transformer implemented that
# serially -- one round per layer or per fixed block of layers -- then the
# minimum num_layers needed to solve depth-T instances would grow with T,
# roughly linearly. If instead it solves them some other way (finding 2 says
# probes, algorithm-ID, counterfactual and SAE all agree it is NOT running
# DD), solve_rate should saturate at the same num_layers for every T.
#
# This is the N=24 arm, run first because it is cheap and sets the method;
# the N=48 arm follows once sud2n48base shows the recipe transfers. Depth
# itself only grows with N (mean T 4.25 at N=24, 6.10 at N=48, 7.45 at N=96),
# so the cross-N comparison is the real test and this is its baseline.
#
# ONE CHANGE PER RUN. num_layers in {1,2,3,4,6,8,12} with everything else
# pinned at the best-recipe values: d_model=256, nhead=4, ff=1024 (so MLP
# WIDTH is constant and only DEPTH varies), lr=1e-3, pre-norm, n_train=20000,
# 150 epochs, same seed. 150 rather than 300 epochs keeps the whole scan near
# 3.5 GPU-h; the data axis is already characterised and is not what this
# varies.
#
# MEASUREMENT. sudoku_depth_eval, not the k bins: it holds rho FIXED and bins
# the same-density instances by their own T, so a gradient across bins is
# attributable to depth rather than to density. Its realized_rho_spread must
# be read first -- if deep-T instances realize a systematically lower density
# the confound has moved inside the bins and the scan says nothing.
#
# PRE-REGISTERED PREDICTIONS.
#  (a) At T=1 (one deduction round) solve_rate is >= 0.95 for every
#      num_layers >= 2. A single round should not need depth.
#  (b) The serial hypothesis: L*(T), the smallest num_layers reaching >= 0.90
#      at depth bin T, increases monotonically over T=1..5 and L*(5) >=
#      L*(1) + 3. Falsifier, and what finding 2 predicts instead: L*(T) is
#      flat -- the same num_layers (expected 4) suffices at every T, and the
#      deeper models buy accuracy uniformly across T rather than extending
#      reach to deeper instances.
#  (c) L=1 fails everywhere (< 0.5 at every T >= 2): with one layer there is
#      no residual stream to iterate on at all.
#  (d) Aggregate solve_rate saturates by L=8, i.e. L=12 is within 0.01 of
#      L=8, consistent with 12 layers already being past the knee at N=24.
#
# Cost: the d256 x 20000 x 300ep run took ~1.4 h at L=12, so ~0.058 h per
# layer-unit per 150 epochs; 37 layer-units total is ~2.2 h plus 7 data
# builds and 7 evals, ~3.5 h. 10:00:00 leaves room for a slow GPU.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_layer_scan_2026-10-24_n24"
mkdir -p "$OUT_DIR"

for L in 1 2 3 4 6 8 12; do
    echo "=== N=24 num_layers=${L} ==="
    python -m spin_gnn.sudoku_hparam_sweep \
        --n-nodes 24 \
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
echo "=== solve_rate by DD depth at fixed density, all depths ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${OUT_DIR}"/d0256_h04_L*_ff01024_lr1e-03_pre \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT_DIR}/depth_by_layers.csv"
