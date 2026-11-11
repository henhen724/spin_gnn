#!/bin/bash
#SBATCH --job-name=sud2N48TiedT
#SBATCH --partition=sganguli
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -G 1
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/users/hshunt/spin_gnn/sud2N48TiedT_%A_%a.out
#
# STAGE T -- apply stage S's ONE shared block T times on FIXED WEIGHTS.
# Registered by scripts/staget_score.py; this array runs --dependency=afterok on
# the commit of that registration, so no cell can land ahead of its reading.
#
# NO TRAINING HAPPENS HERE. One cell = one trained seed, re-evaluated on the
# same 14-point grid at T in {4, 8, 12, 16, 20, 24}. T=16 is the trained count
# and is re-run rather than copied from stage S, so all six curves come from one
# code path on one GPU: a comparison whose reference was produced by different
# code is not a comparison.
#
# WHY THIS IS CHEAP AND WHY IT IS WORTH A GPU AT ALL. Stage S's grid took ~11
# minutes per cell on an H100; six grids with a mean T of 14 is roughly an hour.
# The same sweep on CPU would be a day. Rule 10: each cell is ~60 minutes of
# real work, well past the 10-minute floor.
#
# WHY d320 AND NOT d448. d320 is the width that trained 3 of 3 (train NLL
# 1.5851/1.5915/1.5864, all three crossing 0.5, rho_50 0.757). The d448 arm lost
# two seeds to an epoch-0 divergence that diagnostic 51860479 traced to warmup
# rather than to the cell, so it has one usable seed and cannot carry a 3-seed
# reading.
#
# INSTANCES ARE IDENTICAL ACROSS T: --seed-start is left at its default
# 60000000000, the same value stage S used, so every T sees the same 400
# instances per bin and the per-bin differences are paired (guard T4.2).

set -u
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

SRC=/scratch/users/hshunt/spin_gnn/sudoku_n48_tiedwidth_S_2026-11-10
OUT=/scratch/users/hshunt/spin_gnn/sudoku_n48_tiedT_T_2026-11-11
mkdir -p "$OUT"

SEED=${SLURM_ARRAY_TASK_ID}
CKPT="$SRC/t0${SEED}_d0320_ff01024_h04_tied_no_mlp_L16_s${SEED}/d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied"

echo "host=$(hostname)  job=${SLURM_JOB_ID}  array task=${SLURM_ARRAY_TASK_ID}"
echo "start $(date '+%F %T')"
echo "checkpoint: $CKPT"

if [ ! -d "$CKPT" ]; then
    echo "FAILING CLOSED: no such checkpoint directory."
    exit 1
fi

# The grid is stage S's grid, not a new one: 14 points to the 0.33 floor at 400
# per bin. Changing it here would make the T=16 curve incomparable to stage S's.
GRID="0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 0.55 0.5 0.45 0.4 0.35 0.33"

rc=0
for T in 4 8 12 16 20 24; do
    echo
    echo "=== seed ${SEED}, T=${T} ==========================================="
    python3 -u -m scripts.rho_transition_eval \
        --checkpoint "$CKPT" \
        --n-iters "$T" \
        --rho-grid $GRID \
        --n-per-bin 400 \
        --out-csv "$OUT/rho_curve_s${SEED}_T$(printf '%02d' "$T")_${SLURM_JOB_ID}.csv" \
        || rc=$?
done

echo
echo "=== T4.1 reminder: fixed weights, so the only spread is the three trained"
echo "    seeds. This cell is seed ${SEED} of 3."
echo "score with: python3 -u -m scripts.staget_score"
echo "end $(date '+%F %T')  rc=$rc"
exit "$rc"
