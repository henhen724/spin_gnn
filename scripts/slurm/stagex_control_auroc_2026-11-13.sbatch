#!/bin/bash
# =============================================================================
# X3's CONTROL -- what determinacy AUROC do the lambda=0 cells already reach?
#
# X3 asks whether the determinacy term makes the readout's node logits a
# better answer to "which open nodes are deducible here". That is only a
# question if the control is measured, and the control is NOT 0.5: the main
# cross-entropy already lifts the logit of the node about to be emitted, and
# the node about to be emitted is always deducible, so a cell that never saw
# the term is expected to score above chance by construction.
#
# This measures it on stage S's three d320 cells -- the exact lambda=0 arm the
# stage-X seeds are paired against -- on ONE fixed held-out instance set, so
# when the array lands its cells can be scored on the same set and the two
# groups differ in the model and nothing else.
#
# Run BEFORE the arm finishes on purpose. A control measured after the arm is
# a control whose bar can be chosen, and the whole point of registering
# scripts/stagex_score.py at 1b190a0 was to make that impossible.
#
# Cheap: 600 instances generated once, three forward passes, no training.
# =============================================================================

#SBATCH --job-name=stagexCtrlAuroc
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -euo pipefail

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"

S="${SCRATCH}/spin_gnn/sudoku_n48_tiedwidth_S_2026-11-10"
OUT="${SCRATCH}/spin_gnn/sudoku_n48_determinacy_X_2026-11-13"
mkdir -p "$OUT"

date +'start %F %T'

python3 -m scripts.determinacy_eval \
    --checkpoint \
      "${S}/t00_d0320_ff01024_h04_tied_no_mlp_L16_s0/d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied" \
      "${S}/t01_d0320_ff01024_h04_tied_no_mlp_L16_s1/d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_s01" \
      "${S}/t02_d0320_ff01024_h04_tied_no_mlp_L16_s2/d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_s02" \
    --n-instances 600 \
    --out-csv "${OUT}/determinacy_control_stageS.csv"

echo
echo "=== the three numbers above are X3's bar. The arm must beat ALL of them. ==="
date +'%F %T'
