#!/bin/bash
#SBATCH --job-name=sud2n48dctl
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#
# WHY THIS EXISTS. L*(24)=3 and L*(48)=4 were both measured at n_train=20000.
# The N=96 arm (sudoku2_n96_layer_scan_80k.sbatch, submitted alongside this)
# has to run at 80000, because nothing at 20000 clears 0.90 at N=96 at all
# (49524030: 0.8267, zero of six rho=0.60 bins). So the three-point series
# "L* = 3, 4, ?" would be confounded the moment the third point lands: a change
# in L* between N=48 and N=96 could be the size or could be the 4x data.
#
# This job removes the confound for ~3 h of GPU time by re-measuring the N=48
# knee at 80000. It is the cheap half of a two-job pair and neither settles the
# trend alone.
#
# IT ALSO TESTS SOMETHING IN ITS OWN RIGHT: does training data substitute for
# depth? Width partly does -- narrowing the MLP 4x moves the N=48 knee from
# L=4 to L=6 (49587538). If data does too, then "L*" is not a property of the
# problem at all but of the problem AND the budget, and every L* in this
# project needs restating with its budget attached.
#
# ONE CHANGE from the 49523504/49509466 layer-scan cells: n_train 20000 ->
# 80000. Identical otherwise: N=48, degree=5, n_colors=4, rho=0.3, rho_max=0.95,
# removal hints, d_model=256, nhead=4, ff=1024, pre-norm, lr=1e-3 CONSTANT,
# batch=32, 150 epochs, seed=0.
#
# TAG COLLISION IS INTENTIONAL AND HANDLED. n_train is not part of the
# checkpoint tag, so these cells are named d0256_h04_L0{3,4}_ff01024_lr1e-03_pre
# exactly like their 20000 counterparts. They live in a different out dir, and
# sudoku_agg_eval appends " [run_dir]" to same-tag checkpoints from different
# runs in both the CSV and the npz names, so all four cells stay addressable.
# That is the same mechanism the two N=48 L=12 cells already rely on.
#
# PRE-REGISTERED:
#  (a) L=3 at 80000 still MISSES 0.90 in at least one rho=0.60 bin, so
#      L*(48) stays 4 and the N=96 comparison is not confounded by data.
#      At 20000 its bins are 0.875/0.846/0.860/0.835/0.827 -- a uniform ~0.06
#      shortfall rather than a near miss, so closing it needs a real shift.
#      Falsifier: L=3 clears every bin, which makes L* budget-dependent, blocks
#      reading the N=96 number as a size effect, and would mean data and depth
#      are substitutes in the same way width and depth are.
#  (b) L=4 at 80000 beats L=4 at 20000 (0.9093) on the shared pool at p<0.05.
#      This is the premise of the control, not a discovery; if the data lever
#      does nothing at N=48 then (a) is vacuous and must be reported that way
#      rather than as support for L* stability.
#  (c) The 4x-data gain at N=48 is SMALLER than the +0.1007 it bought at N=96.
#      Reason: at N=48 the 20000-instance runs end 0.0497 to 0.1413 ABOVE their
#      val minimum (finding 14) -- they already fit -- whereas N=96 at 20000 ends
#      0.0289 BELOW val, underfitting, and 80000 is what put it into a normal
#      +0.0239 gap. Falsifier: an equal or larger gain at N=48, which would mean
#      the train-val gap does not predict the data lever's size -- the same
#      inference that was wrong in finding 15, wrong again in the other
#      direction, and I should stop using it.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

SCAN="${SCRATCH}/spin_gnn/sudoku_layer_scan_2026-10-24_n48"
OUT="${SCRATCH}/spin_gnn/sudoku_n48_data_control_2026-10-26"
mkdir -p "$OUT"

for L in 3 4; do
    echo
    echo "=== N=48 num_layers=${L}, n_train=80000 (one change from the scan) ==="
    python -m spin_gnn.sudoku_hparam_sweep \
        --n-nodes 48 --degree 5 --n-colors 4 \
        --d-model 256 --nhead 4 --num-layers "$L" --dim-feedforward 1024 \
        --n-train-instances 80000 --n-val-instances 300 --batch-size 32 \
        --epochs 150 --n-per-bin 100 \
        --out-dir "$OUT"
done

echo
echo "=== does 4x data move the N=48 knee? L*(T) at 80000 ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${OUT}"/d0256_h04_L0[34]_ff01024_lr1e-03_pre \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT}/depth_by_layers_n48_80k.csv"

echo
echo "=== metric of record: 20000 vs 80000 at L=3 and L=4 on one pool ==="
# The 20000 L=4 cell is the reference: it is the published N=48 knee (0.9093).
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "${SCAN}/d0256_h04_L04_ff01024_lr1e-03_pre" \
                 "${SCAN}/d0256_h04_L03_ff01024_lr1e-03_pre" \
                 "${OUT}/d0256_h04_L04_ff01024_lr1e-03_pre" \
                 "${OUT}/d0256_h04_L03_ff01024_lr1e-03_pre" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500.csv"
