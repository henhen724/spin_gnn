#!/bin/bash
# =============================================================================
# STAGE S DIAGNOSTIC -- is the d448 failure an INIT INSTABILITY or the cell?
#
# NOT A SCORED CELL. This is a bounded screen that writes to its own dated
# directory, outside the stage S run dir, so it can never be picked up as one of
# stage S's six cells or enter a pooled yardstick.
#
# WHAT WAS SEEN (array 51853969, tasks 4 and 5, read at 20:40 on 11-10)
#
#   d320 s0/s1/s2  epoch 4  train_nll 1.6263 / 1.6871 / 1.6738   healthy
#   d448 s0        epoch 3  train_nll 1.6873                     healthy
#   d448 s1        epoch 3  train_nll 3.9563  val_accuracy 0.0313 DIVERGED
#   d448 s2        epoch 3  train_nll 3.9696  val_accuracy 0.0330 DIVERGED
#
# Both failures are already at ~3.89 at the END OF EPOCH 0 and flat or drifting
# UP after that, while the healthy cells are at 1.75 by then. Train NLL near 3.9
# with token accuracy near 0.03 is the uniform predictor over the whole
# vocabulary -- worse than chance on colours -- so these two did not train
# slowly, they blew up inside the first epoch and settled there. Same width as
# d448 s0, which is fine, so it is not structural (448/4 = 112 head dim): it is
# the stochastic early-step instability that warmup exists for.
#
# HYPOTHESIS, stated before the run: the divergence is an instability of the
# first few hundred steps at lr 1e-3 with no ramp. With --warmup-epochs 5 the
# same width and the SAME SEED trains, i.e. train NLL is below 2.0 by the last
# epoch here, against 3.956 where warmup 0 sat at the same epoch. If it still
# sits near 3.9, warmup is not the mechanism and the next knob is the lr (S5's
# other arm, 3e-4 cosine), not another warmup length.
#
# ONE KNOB: --warmup-epochs 0 -> 5. Width, seed, arch, heads, ff, lr, batch,
# instance count and flags are copied from the stage S array verbatim.
#
# WHY ONLY 8 EPOCHS: 5 of ramp plus 3 at the full 1e-3, which is where a warmup
# that merely DELAYS the blowup would show it. A cell that needs scoring needs
# 50 epochs and three seeds; that is S5's registered schedule screen and it is
# hshunt's call, not something to start unasked at 2 GPU-days.
#
# WHY NOT CANCEL THE TWO DIVERGED TASKS: stage S's own registration pre-empts
# that mistake twice (S3, guard 6) -- stage O was called on one arm because the
# other was cancelled mid-flight, and a cell with no final epoch cannot be
# scored at all. "d448 s1/s2 never left the uniform state under warmup 0" is a
# result and needs its final-epoch record.
# =============================================================================
#SBATCH --job-name=sud2N48TiedWarmSdiag
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -euo pipefail

DMODEL=448
LAYERS=16
SEED=1
FF=1024
NHEAD=4
WARMUP=5          # the one knob; stage S runs this cell at 0
ARCH=tied_no_mlp
EPOCHS=8

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"

echo "host=$(hostname)  job=${SLURM_JOB_ID}"
echo "start $(date '+%F %T')"

echo "=== smoke check ==="
python3 - "$DMODEL" "$NHEAD" <<'PY'
import sys
import torch, numpy, numba
import spin_gnn.sudoku_hparam_sweep  # noqa: F401
from spin_gnn.sat_transformer2 import ARCH_MODES
from spin_gnn.sudoku_hparam_sweep import ARCH_TAG
d, nhead = int(sys.argv[1]), int(sys.argv[2])
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
assert "tied_no_mlp" in ARCH_MODES, ARCH_MODES
assert ARCH_TAG["tied_no_mlp"] == "_noff_tied", ARCH_TAG
assert d % nhead == 0, (d, nhead)
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_tiedwarm_Sdiag_2026-11-10"
cell="${OUT}/diag_d0448_ff01024_h04_${ARCH}_L16_s${SEED}_w$(printf '%02d' "$WARMUP")"
mkdir -p "$cell"

echo "DIAGNOSTIC  d_model=${DMODEL} seed=${SEED} arch=${ARCH} warmup=${WARMUP} epochs=${EPOCHS}"
echo "cell -> ${cell}"
echo "the cell this is diagnosing: stage S array 51853969 task 4, warmup 0,"
echo "train_nll 3.8873 / 3.9402 / 3.9588 / 3.9563 over epochs 0-3"

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model "$DMODEL" --nhead "$NHEAD" --dim-feedforward "$FF" \
    --num-layers "$LAYERS" --arch-mode "$ARCH" --seed "$SEED" \
    --warmup-epochs "$WARMUP" \
    --n-train-instances 240000 --epochs "$EPOCHS" \
    --n-val-instances 300 --batch-size 32 --n-per-bin 100 \
    --out-dir "$cell"

echo
echo "=== VERDICT (the hypothesis was: below 2.0 by the last epoch) ==="
log="/scratch/users/hshunt/spin_gnn/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
grep -E '^epoch' "$log" || true
python3 - "$log" <<'PY'
import re, sys
rows = []
for line in open(sys.argv[1]):
    m = re.match(r"epoch (\d+): train_nll=([0-9.]+).*val_accuracy=([0-9.]+)", line)
    if m:
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
if not rows:
    print("NO EPOCH LINES -- cannot call the diagnostic")
    raise SystemExit(1)
e, nll, acc = rows[-1]
# Warmup 0 at this width and seed: 3.9563 at epoch 3, drifting up, accuracy 0.031.
print("last epoch %d: train_nll %.4f  val_accuracy %.4f" % (e, nll, acc))
if nll < 2.0:
    print("WARMUP FIXES IT. The d448 divergence is an init instability, not the")
    print("cell. S5's schedule screen is the right follow-up and warmup 5 is the")
    print("knob to run it with -- 3 seeds, 50 epochs, hshunt's call on the GPUs.")
elif nll > 3.5:
    print("WARMUP DOES NOT FIX IT. Still at the uniform predictor, so the next")
    print("knob is the lr (S5's 3e-4 cosine arm), NOT a longer ramp.")
else:
    print("PARTIAL: off the uniform state but not where the healthy seeds are.")
    print("Report the number; extrapolate nothing from 8 epochs.")
PY
echo "end $(date '+%F %T')"
