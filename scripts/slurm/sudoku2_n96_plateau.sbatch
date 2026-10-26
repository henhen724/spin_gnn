#!/bin/bash
#SBATCH --job-name=sud2n96plat
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --output=%x_%j.out
#
# HYPOTHESIS. The N=96 ceiling is an OPTIMIZATION plateau, not data starvation.
# Every run in this project has trained at a constant lr -- `torch.optim.Adam(
# model.parameters(), lr=train_cfg.lr)` with no scheduler anywhere in the
# codebase -- and at N=96 that is where the loss stops.
#
# EVIDENCE, from 49524030's own history.json. val_nll bottoms at 1.4588 at
# epoch 129 of 150, final 1.4621, span over the last 30 epochs 0.0043: flat.
# And train_nll ends at 1.4332 against val_nll 1.4621, a gap of -0.029 -- the
# model does not fit its own training set, let alone overfit it. Compare N=48,
# where the same recipe ended +0.0497 to +0.1413 ABOVE its val minimum. A
# data-limited model overfits; this one cannot descend.
#
# THIS CONTRADICTS MY OWN PRE-REGISTRATION, and I am not going to quietly drop
# either side of that. 49524030's sbatch said: "the next move is data (n_train
# 40000/80000 at N=96), not layers". The reason given was finding 6 -- training
# set size has been the binding constraint and is not saturated at 80000. That
# reason is premised on a generalization gap, and the run came back without
# one. So the pre-registered lever is still tested here, as cell B, against the
# same control and on the same pool; it is demoted from "the next move" to "one
# of two candidate levers", and if it wins, it wins on the numbers.
#
# THREE CELLS, each exactly ONE change from 49524030 (d0256/L12/ff1024/
# nhead=4/pre/lr=1e-3/150 ep/n_train=20000/seed=0/N=96/degree=5/n_colors=4):
#   SMOKE  N=12, 2 epochs, cosine. Proves the new --lr-schedule plumbing end to
#          end before 10 h of GPU time depends on it. ~1 min, throwaway dir.
#   A      --lr-schedule cosine. Nothing else touched.
#   B      --n-train-instances 80000. Constant lr, as before.
# A runs before B because A is the cheap one and the one that tests the
# hypothesis; a wall-clock overrun then still leaves the decisive cell measured.
#
# WHY 80000 AND NOT 40000 FOR CELL B. At N=24 a 4x data step (20000 -> 80000)
# cut the error roughly in half, 0.0163 -> 0.0045. Halving N=96's error of
# 0.1733 lands at ~0.913, i.e. 4x is about what it would take to clear the
# threshold at all; 2x would land near 0.87 and settle nothing. If the data
# lever is going to be tested, it has to be tested at a size that could win.
#
# THE CODE CHANGE, and why it cannot disturb anything already measured.
# `lr_schedule` defaults to "constant" in SudokuReasoningTrainConfig and in the
# sweep CLI, which is exactly the old behaviour, and the checkpoint tag only
# gains a "_cos" suffix when it is non-default -- so every tag written before
# today is byte-identical and the decks and agg_eval paths that address cells
# by that string still resolve. Two other things were fixed while in there:
#   * save_checkpoint now records seed, epochs, lr, n_train and the schedule in
#     config.json. None of them were saved before, so a results notebook had to
#     recover the seed and the training budget from the sbatch that produced
#     the run. Additive; agg_eval's INSTANCE_KEYS is unaffected.
#   * sweep.csv is append-only, but the writer used `fieldnames=list(row.keys())`
#     on every append. Adding any column would therefore have written ragged
#     rows into every sweep.csv that predates it -- silently, since the header
#     already on disk would no longer match. The writer now reads the existing
#     header, reuses it, and prints which fields it cannot hold. Verified on a
#     synthetic old-format file: rows stay aligned and the new field is dropped
#     with a note instead of corrupting the file.
#
# PRE-REGISTERED:
#  (a) Cell A's val_nll minimum falls at least 0.02 below the control's 1.4588,
#      and its shared-pool rate beats 0.8267 paired at p < 0.05. Falsifier: the
#      plateau does not move, in which case constant lr was NOT the ceiling,
#      this diagnosis is wrong, and the remaining candidates are capacity and
#      the data lever of cell B.
#  (b) Cell B gains LESS than cell A, because there is no generalization gap at
#      N=96 for more data to close. Falsifier: B >= A, which would mean
#      train ~ val did not imply data-insensitivity here and finding 6 extends
#      to N=96 after all; then the data ladder is the route, my diagnosis above
#      is wrong, and the pre-registration I argued down was right.
#  (c) NEITHER cell clears 0.90 on the shared pool. This is a live outcome, not
#      a formality: the control is 0.1733 short. If it happens, the N=96 layer
#      scan stays unsubmitted and the cross-N claim rests on N=24 vs N=48 --
#      write that down rather than reporting 0.87 as "nearly there".
#  (d) If a cell does clear 0.90 aggregate, it must also clear 0.90 in EVERY
#      rho=0.60 depth bin (T=4..9), since that is what makes L*(T) measurable
#      at N=96 and it is the whole reason for the exercise. The control clears
#      it at rho=0.75 (1.000/0.984/0.983/0.947) but nowhere at rho=0.60
#      (0.860/0.822/0.839/0.754/0.794/0.837). Aggregate alone is not the gate.
#
# Cost: 49524030 was 02:59:15 end to end for one cell plus both evals, so cell A
# is ~2 h of training. Cell B at 4x data is ~8 h plus a longer instance build
# (~80000 N=96 removal paths). Two depth evals and one 3-checkpoint pool pass at
# 787-token generation are ~2 h. 18 h leaves real margin on an 8 h unknown.
# 128G/4 cores because the 80000-instance build holds 4x the tokens of the
# 20000-instance one, which ran in 64G; sganguli caps mem at 32G per core.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

CTRL="${SCRATCH}/spin_gnn/sudoku_n96_baseline_2026-10-25/d0256_h04_L12_ff01024_lr1e-03_pre"
OUT="${SCRATCH}/spin_gnn/sudoku_n96_plateau_2026-10-26"
SMOKE="${SCRATCH}/spin_gnn/_smoke_lr_schedule"
A="${OUT}/d0256_h04_L12_ff01024_lr1e-03_pre_cos"
B="${OUT}/d0256_h04_L12_ff01024_lr1e-03_pre"
mkdir -p "$OUT"

echo "=== SMOKE: new --lr-schedule plumbing, N=12, 2 epochs ==="
python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 12 --degree 5 --n-colors 4 \
    --d-model 64 --nhead 4 --num-layers 2 --dim-feedforward 128 \
    --lr-schedule cosine \
    --n-train-instances 200 --n-val-instances 50 --batch-size 32 \
    --epochs 2 --n-per-bin 5 \
    --out-dir "$SMOKE"
echo "--- smoke config.json (seed/epochs/lr_schedule must be present):"
python -c "
import json,glob
p=sorted(glob.glob('${SMOKE}/*_cos/config.json'))[-1]
c=json.load(open(p))
print({k:c.get(k) for k in ('seed','epochs','lr','lr_schedule','lr_min_factor','n_train_instances')})"

echo
echo "=== CELL A: cosine lr, one change from 49524030 ==="
python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 96 --degree 5 --n-colors 4 \
    --d-model 256 --nhead 4 --num-layers 12 --dim-feedforward 1024 \
    --lr-schedule cosine \
    --n-train-instances 20000 --n-val-instances 300 --batch-size 32 \
    --epochs 150 --n-per-bin 100 \
    --out-dir "$OUT"

echo
echo "=== CELL B: 4x data at constant lr (the pre-registered lever) ==="
python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 96 --degree 5 --n-colors 4 \
    --d-model 256 --nhead 4 --num-layers 12 --dim-feedforward 1024 \
    --n-train-instances 80000 --n-val-instances 300 --batch-size 32 \
    --epochs 150 --n-per-bin 100 \
    --out-dir "$OUT"

echo
echo "=== depth bins: is L*(T) measurable at N=96 at rho=0.60 at all? ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "$A" "$B" \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT}/depth_n96_plateau.csv"

echo
echo "=== metric of record: both cells against the control on one pool ==="
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "$CTRL" "$A" "$B" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500.csv"
