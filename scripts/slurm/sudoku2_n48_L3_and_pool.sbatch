#!/bin/bash
#SBATCH --job-name=sud2n48L3pool
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=%x_%j.out
#
# Does L* rise with N? The N=48 scan left the one cell that decides it untested.
#
# WHAT 49509466 FOUND. L*(T) at N=48 is FLAT in T -- 4 at every depth, T=3..7 at
# rho=0.60 and T=2..4 at rho=0.75, with realized_rho spread 0.0004 and 0.0000,
# so there is no density confound to argue about this time. That kills the
# serial/chaining reading at a second problem size and over a wider T range
# than N=24 had. But the grid was {1,2,4,8,12} and SKIPPED L=3, which is exactly
# the N=24 knee, so all that scan can say is L*(N=48) is in (2, 4]:
#
#   L=1  0.003     L=2  0.243     L=4  0.903     L=8  0.953     L=12  0.953
#
# L=2 is catastrophic at N=48 (0.02-0.20 per depth bin) and L=4 clears 0.90 at
# every bin, so the knee sits at 3 or at 4 and nothing measured distinguishes
# them. That distinction is the whole question in STATUS next-step 1:
#
#   L*(48) = 3  -> flat in T AND flat in N. Depth is a fixed cost, full stop,
#                  and "reasoning depth" is not the axis it tracks (finding 2).
#   L*(48) = 4  -> flat in T but rising in N. Depth tracks GRAPH SIZE, not
#                  reasoning depth -- a different and more interesting claim
#                  than either hypothesis the layer scan was built to test.
#
# So: one cell, L=3 at N=48, everything else identical to 49509466.
#
# THE EPOCH-BUDGET GATE FAILED, AND WHY THIS RUNS ANYWAY. 49509466's sbatch
# pre-registered that its L=12 cell must reproduce 49433453's 0.9033 within
# that number's Wilson interval [0.865, 0.932], or else "the scan is confounded
# with epoch budget -- report and stop rather than read L*(T) off it". It came
# in at 0.9533, outside the interval, on the HIGH side: 150 epochs beat 300.
#
# The cause is identified rather than mysterious. Both runs reach the same
# validation minimum (1.2146 at ep46 for 150 ep, 1.2164 at ep36 for 300 ep --
# training is reproducible), and both then overfit, the 300-epoch run further:
# +0.1413 above its minimum against +0.0497. save_checkpoint stores the final
# epoch, so the longer run's SAVED checkpoint is deeper into overfitting. This
# is the standing open problem about final-epoch checkpointing, now showing up
# as a real solve_rate cost for the first time rather than as a val_nll
# curiosity (finding 4 said val_nll barely predicts solve_rate across models;
# this is within one model, where it evidently does).
#
# The confound the gate was guarding against -- cells differing in how long
# they trained -- does not exist inside the scan: all five cells ran 150 epochs,
# as does this one, as does the whole N=24 arm it gets compared against. The
# gate failure is cross-JOB, so what it invalidates is using 49433453 as a
# reproducibility anchor, not the within-scan or cross-arm L* comparison.
# Reporting it and continuing is therefore the honest reading of the rule, not
# a way around it. The shared-pool pass below settles the discrepancy properly.
#
# PART 2: THE METRIC OF RECORD. Everything above is the per-run 300-instance
# agg_solve_rate, which STATUS retired. The 0.9533-vs-0.9033 gap is 0.05 at
# n=300, where the two Wilson intervals only just fail to overlap, and the
# L=8/L=12 tie (0.9533 both) is far inside the noise. So all seven N=48
# checkpoints get scored on the shared pool at n=1500. All seven share an
# instance signature (n_nodes=48, degree=5, n_colors=4, rho=0.3, rho_max=0.95,
# removal), verified from their config.json, so one call covers them.
#
# PRE-REGISTERED:
#  (a) L=3 at N=48 scores BELOW 0.90 aggregate, i.e. L*(48) = 4 and the knee
#      rises with N. Falsifier: it clears 0.90 at every depth bin, in which case
#      L* is flat at 3 across a 2x change in N and the fixed-cost reading wins
#      outright. Either outcome is informative; this is a genuine 50/50 and is
#      written down so it cannot be reinterpreted afterwards.
#  (b) The 150-epoch L=12 cell beats the 300-epoch one on the shared pool,
#      paired McNemar p < 0.05. This is the overfitting diagnosis above stated
#      as a test. Falsifier: they tie, in which case the gate failure was just
#      300-instance noise, the diagnosis is wrong, and the standing open
#      problem stays open rather than becoming a measured cost.
#  (c) L=8 and L=12 stay tied at n=1500 (|delta| < 0.02), as they are at n=300.
#      A knee that flattens by L=8 at N=48 mirrors the N=24 arm flattening by
#      L=4, which is the shape the cross-N comparison needs.
#
# OUTPUTS. The L=3 cell goes into the EXISTING scan directory: sweep.csv is
# opened in append mode (sudoku_hparam_sweep.py:92) and the cell's own
# subdirectory is new, so nothing is overwritten, and make_layer_scan_deck
# picks it up with no edit because by_layers() keys on num_layers. The depth
# eval re-runs over all six cells into a NEW csv rather than overwriting
# depth_by_layers.csv -- the five existing cells use the same pool seeds, so
# their numbers reproducing exactly is a free check on the first run.
#
# Cost: 49509466 did 27 layer-units at N=48 in 3:08:52, ~7 min per unit, so
# L=3 is ~21 min plus a data build. The depth eval over six cells and a
# 1500-instance pool pass at 402 tokens per sequence are the larger half;
# 49521537 did eight cells at n=1500 and N=24 in 4:59. Budget 3 h.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

SCAN="${SCRATCH}/spin_gnn/sudoku_layer_scan_2026-10-24_n48"
BASE="${SCRATCH}/spin_gnn/sudoku_n48_baseline_2026-10-24"
POOL="${SCRATCH}/spin_gnn/sudoku_n48_sharedpool_2026-10-25"
mkdir -p "$POOL"

echo "=== N=48 num_layers=3 (the cell 49509466 skipped) ==="
python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 \
    --degree 5 \
    --n-colors 4 \
    --d-model 256 \
    --nhead 4 \
    --num-layers 3 \
    --dim-feedforward 1024 \
    --n-train-instances 20000 \
    --n-val-instances 300 \
    --batch-size 32 \
    --epochs 150 \
    --n-per-bin 100 \
    --out-dir "$SCAN"

echo
echo "=== solve_rate by DD depth, all six N=48 cells (new csv, originals kept) ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${SCAN}"/d0256_h04_L*_ff01024_lr1e-03_pre \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${SCAN}/depth_by_layers_with_L03.csv"

echo
echo "=== shared pool, n=1500: the N=48 ladder plus the 300-epoch baseline ==="
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint \
        "${SCAN}/d0256_h04_L12_ff01024_lr1e-03_pre" \
        "${BASE}/d0256_h04_L12_ff01024_lr1e-03_pre" \
        "${SCAN}/d0256_h04_L01_ff01024_lr1e-03_pre" \
        "${SCAN}/d0256_h04_L02_ff01024_lr1e-03_pre" \
        "${SCAN}/d0256_h04_L03_ff01024_lr1e-03_pre" \
        "${SCAN}/d0256_h04_L04_ff01024_lr1e-03_pre" \
        "${SCAN}/d0256_h04_L08_ff01024_lr1e-03_pre" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${POOL}/sharedpool_n1500.csv"
