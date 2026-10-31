#!/bin/bash
#SBATCH --job-name=sud2N48deep
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#
# N=48 LEAK-FREE L* LADDER, DEEP END: DOES THE 0.90 GATE GET CLEARED AT ALL?
#
# ===========================================================================
# Set LAYERS (and a matching --time) in the submitting environment:
#
#   LAYERS=12 sbatch --time=14:00:00 scripts/slurm/sudoku2_n48_deep_lstar.sbatch   # ANCHOR, read first
#   LAYERS=8  sbatch --time=10:00:00 scripts/slurm/sudoku2_n48_deep_lstar.sbatch
#
# One cell per job on purpose: at N=48 a cell costs ~50 min per layer, so the
# two rungs in one job would be ~17.5 h and a timeout would cost the anchor.
# NNODES, NTRAIN, EPOCHS are fixed here (not parameters) because they are the
# 10-29 ladder's values and this series is only meaningful paired against it.
# ===========================================================================
#
# WHY. Job 50099945 ran the N=48 leak-free ladder at L=6, 4, 3 and the result
# is the one its own header warned about: nothing clears the gate.
#
#     shared pool n=1500, seed_start 5e10, --shuffle-adj
#     L=3  solve_rate 0.0093   [0.0056, 0.0156]
#     L=4             0.3153   [0.2923, 0.3393]   +0.306 over L=3
#     L=6             0.6153   [0.5904, 0.6396]   +0.150 per layer over L=4
#
# L=6 was the anchor and it missed 0.90 by 0.285, so L*(48) is currently
# UNDEFINED, not large -- finding 15's error at N=96 is exactly this, and the
# fix is to extend the ladder up rather than to write 6/4/3 up as an L*.
#
# ONE CHANGE PER CELL. Every argument below is job 50099945's verbatim
# (N=48, degree 5, q=4, d_model 256, nhead 4, ff=1024, pre-norm, lr 1e-3,
# batch 32, 240000 x 100ep, --shuffle-adj). `num_layers` is the only
# difference, and it is not an INSTANCE_KEY, so the new cells pool with the
# three 10-29 cells on the same n=1500 instances and on the same depth bins.
#
# PRE-REGISTERED, written before either job ran:
#  (a) ANCHOR, moderate confidence: L=12 clears the gate, solve_rate >= 0.90
#      with Wilson lo >= 0.88. Reason: the measured increments are +0.306
#      (3->4) and +0.150/layer (4->6); six further layers cover the remaining
#      0.285 even if the per-layer gain keeps decaying at that rate.
#      Falsifier: L=12 < 0.90. Then the binding constraint at N=48 is the
#      240000 x 100ep budget, not depth, and the honest write-up is the
#      solve-rate-vs-depth curve with "the gate was never reached" -- NOT an
#      L*(48) of "greater than 12", and NOT a T-slope, because L*(T) is
#      undefined in every bin when no cell reaches 0.90.
#  (b) L=8 lands strictly between L=6 and L=12, in [0.70, 0.84]. Reason: the
#      4->6 slope extrapolated linearly gives 0.92 at L=8, which I expect to
#      overshoot because 3->4->6 is already decelerating.
#      Falsifier A: L=8 >= 0.90, i.e. the gate is at or below 8 and the next
#      rung to run is L=7, downward, not upward.
#      Falsifier B: L=8 <= 0.615, i.e. no better than L=6 -- depth saturates
#      between 6 and 8 and (a) should be disbelieved before it is read.
#  (c) Finding 17's monotone depth cost survives at the deep end: inside the
#      L=12 cell, per-T solve_rate at rho=0.60 falls monotonically in T.
#      Falsifier: flat or non-monotone, which would say what limits N=48 is
#      not deduction-chain length.
#  (d) Low confidence, stated so a surprise counts: L=12's per-T curve is
#      FLATTER in T than L=6's, the sign "more layers buys more rounds"
#      implies. Falsifier: parallel curves (an additive difficulty term with
#      no architectural remedy).
#
# COST, measured from job 50099945's own cell mtimes (not guessed): L=3
# 170 min, L=4 219 min, L=6 319 min, i.e. 20 min fixed + 49.9 min per layer at
# this N and budget. L=8 ~7.0 h, L=12 ~10.3 h; the two evals took 8.5 min for
# three checkpoints and run over six here, so ~15 min. The --time values at the
# top leave ~40% headroom. sganguli's maximum is 7 d (sh_part, 2026-10-31).
#
# ENVIRONMENT. ${SCRATCH}/spin_gnn_venv is dead -- the 90-day purge has taken
# its pip, cffi and now pandas -- so this script uses the module stack only and
# calls python3 explicitly: without the venv on PATH, bare `python` is the
# system 2.7 and torch fails to even parse. The smoke check below costs seconds
# and turns an environment failure into a 30-second job instead of a 10-hour
# one that dies in the evals.

set -euo pipefail
LAYERS="${LAYERS:?set LAYERS to a single rung, 12 (anchor) or 8}"
NNODES=48
NTRAIN=240000
EPOCHS=100

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

echo "=== smoke check ==="
python3 - <<'PY'
import torch, numpy, numba
import spin_gnn.sudoku_hparam_sweep, spin_gnn.sudoku_depth_eval, spin_gnn.sudoku_agg_eval
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "no GPU visible to torch"
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n${NNODES}_leakfree_lstar_$(date +%Y-%m-%d)"
mkdir -p "$OUT"
echo "N=${NNODES} layers=[${LAYERS}] n_train=${NTRAIN} epochs=${EPOCHS} -> ${OUT}"

COMMON=(--n-nodes "$NNODES" --degree 5 --n-colors 4 --shuffle-adj
        --d-model 256 --nhead 4 --dim-feedforward 1024
        --n-train-instances "$NTRAIN" --epochs "$EPOCHS"
        --n-val-instances 300 --batch-size 32 --n-per-bin 100)

CKPTS=()
for L in $LAYERS; do
    cell="${OUT}/L$(printf '%02d' "$L")"
    echo
    echo "=== cell L=${L} ==="
    date +'cell start %F %T'
    python3 -m spin_gnn.sudoku_hparam_sweep "${COMMON[@]}" \
        --num-layers "$L" --out-dir "$cell"
    date +'cell end   %F %T'
    CKPTS+=("${cell}/d0256_h04_L$(printf '%02d' "$L")_ff01024_lr1e-03_pre_shuf")
done

# The 10-29 ladder, added as references so every comparison is paired on the
# same instances. Guarded: a missing cell must not fail the evals.
REF_DIR="${SCRATCH}/spin_gnn/sudoku_n48_leakfree_lstar_2026-10-29"
for L in 06 04 03; do
    ref="${REF_DIR}/L${L}/d0256_h04_L${L}_ff01024_lr1e-03_pre_shuf"
    if [ -d "$ref" ]; then CKPTS+=("$ref"); else echo "skipping missing reference $ref"; fi
done

echo
echo "=== pre-regs (c)+(d): per-T solve_rate on ONE set of depth bins ==="
python3 -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${CKPTS[@]}" \
    --rho 0.75 0.60 --n-pool 1500 --min-per-bin 60 \
    --out-csv "${OUT}/depth_lstar_${SLURM_JOB_ID}.csv"

echo
echo "=== pre-regs (a)+(b): the shared n=1500 pool, disjoint from training ==="
python3 -m spin_gnn.sudoku_agg_eval \
    --checkpoint "${CKPTS[@]}" \
    --n-instances 1500 --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500_${SLURM_JOB_ID}.csv"

echo
echo "=== summaries ==="
cat "${OUT}/sharedpool_n1500_${SLURM_JOB_ID}.csv"
