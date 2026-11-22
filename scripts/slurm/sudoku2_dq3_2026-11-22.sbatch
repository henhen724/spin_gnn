#!/bin/bash
#SBATCH --job-name=sud2DQ3
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --array=0-1%2
#SBATCH --output=%x_%A_%a.out
#
# STAGE DQ3 -- a third seed at each size of stage DQ. ONE knob moves against
# runs/stagedq_2026-11-21/: the seed, 0/1 -> 2. Same ensemble (d=4 q=3), same
# sizes, same architecture, same data budget, same eval grid.
#
#   sbatch scripts/slurm/sudoku2_dq3_2026-11-22.sbatch
#
# PREDICTIONS E1-E5 ARE REGISTERED IN runs/stagedq3_2026-11-22/REGISTERED.md
# BEFORE this job was submitted. Read that file, not this header.
#
# WHY. Stage DQ's VERDICT.txt scored D1 PARTIAL at a POOLED 0.2234 nats and
# said in its own NOTE that the pooled number is not seed-robust. Per cell,
# reason_color is:
#     N=96  s0  0.4468  acc 0.7671
#     N=96  s1  0.0000  acc 1.0000
#     N=120 s0  1.0987  acc 0.3297   <- ln 3 to 4 dp, accuracy exactly 1/q
#     N=120 s1  0.0001  acc 1.0000
# Three of four cells sit at an extreme and one is intermediate. D2's
# "+0.3260 nats, deeper is worse" is the mean of a collapsed cell and a
# perfect one -- it reads as a depth trend only because the pair was averaged.
# HYPOTHESIS: the colour head's state is a BIMODAL, SEED-DEPENDENT training
# outcome, not a function of N or of DD depth T. The competing reading, which
# this run can confirm instead, is that N=120 is past a threshold and N=120 s1
# was lucky. A third seed separates them; two cannot.
#
# SEED COUNT IS NOT A CLOSED KNOB and the project's non-negotiables ask for 3
# seeds, so this is completing stage DQ rather than opening a new lever. It
# re-tests none of width, cosine decay, more data, best-val selection, role
# count, schedule length or training budget.
#
# WALLTIME IS MEASURED, NOT ESTIMATED, unlike the parent job. sacct -X on
# 53243550: N=96 09:58:54 and 10:03:28, N=120 13:15:05 and 13:11:43, all
# COMPLETED 0:0. --time=18:00:00 is ~36% headroom over the larger measurement.
# The parent asked for 30 h and used 13.25.
#
# TRAPS CARRIED FORWARD
#  1. Chance is ln 3 = 1.098612 and chance accuracy is 1/3. NOTHING FROM q=4
#     IS QUOTED: not the free line 0.740, s* 0.1484/0.119, the val_nll floors
#     1.371/1.625/1.861, (5/16) log N, nor ln 4 itself.
#  2. Scoring is PER CELL. Pooling across seeds is the thing that produced the
#     misleading D2, so any pooled number is reported beside its parts.
#  3. IF A BIN SOLVES 0 OF 400, a_eff = 0 IS NOT REPORTED. Zero successes
#     bounds the rate: p < 1 - 0.05^(1/400) = 0.00746, a_eff < p^(1/n_blank).
#     The bound RISES with T purely because n_blank grows -- an artefact, and
#     it must be labelled as one wherever it is drawn.
#  4. round_mark was NEVER covered by D3's 0.01 bar even though its data
#     entropy is exactly zero. It read 0.1075-0.5410 on all four parent cells.
#     E4 registers it here; the parent's D3 verdict is NOT retroactively
#     adjusted.
#  5. D5's FLAG on both parent N=120 cells fired on a +0.0002-nat ordering
#     (best at epoch 47 of 50), not on the registered "flat at its initial
#     plateau" condition -- they fell 3.06 -> 1.74. Reported as it fell.
#
set -euo pipefail

cd /home/users/hshunt/spin_gnn

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 \
    py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312

# task -> N. The seed is 2 on BOTH cells; that is the only thing this stage
# changes relative to the parent array.
case "$SLURM_ARRAY_TASK_ID" in
    0) N=96  ;;
    1) N=120 ;;
    *) echo "bad array index"; exit 1 ;;
esac
SEED=2
DEG=4
NCOL=3
RHO0=0.14

# A NEW DATED DIRECTORY: nothing is written next to the parent's cells, so the
# scored stage DQ artefacts stay exactly as they were scored.
SCAN=$SCRATCH/spin_gnn/sudoku_dq3_2026-11-22
CELL=$SCAN/n${N}_d${DEG}_q${NCOL}_L12_s${SEED}
mkdir -p "$CELL"

echo "=== commit at submit ==="
git log --oneline -1
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cell: N=$N d=$DEG q=$NCOL seed=$SEED"

python3 - "$N" "$DEG" "$NCOL" <<'PY'
import sys
import torch
from spin_gnn.config import SudokuReasoningConfig, SudokuReasoningTrainConfig
from spin_gnn.sudoku_tokens import Vocab
from spin_gnn.train_sudoku_transformer import resolve_max_len_margin
n, d, q = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
assert torch.cuda.is_available(), "no GPU visible to torch"
assert n % q == 0 and (d * n) % (q * (q - 1)) == 0, f"illegal cell {n},{d},{q}"
m = resolve_max_len_margin(SudokuReasoningTrainConfig(),
                           SudokuReasoningConfig(n_nodes=n, degree=d, n_colors=q))
v = Vocab(n, q, d)
assert v.node_offset == 6 + q, f"node_offset {v.node_offset} != {6 + q}"
assert v.size == n + 6 + q, f"vocab {v.size} != {n + 6 + q}"
print("cuda", torch.cuda.get_device_name(0),
      "| margin", m, "| vocab", v.size, "| node_offset", v.node_offset)
PY

date +'cell start %F %T'
python3 -u -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes "$N" --degree "$DEG" --n-colors "$NCOL" --shuffle-adj --fall \
    --rho "$RHO0" --rho-max 0.95 \
    --d-model 256 --nhead 4 --dim-feedforward 1024 \
    --num-layers 12 --seed "$SEED" \
    --n-train-instances 240000 --epochs 50 \
    --n-val-instances 300 --batch-size 32 --n-per-bin 100 \
    --out-dir "$CELL"
date +'cell end   %F %T'

CKPT=$(dirname "$(find "$CELL" -name model.pt | head -1)")
[ -f "${CKPT}/model.pt" ] || { echo "no checkpoint under $CELL"; ls -R "$CELL"; exit 1; }
echo "checkpoint: $CKPT"

echo
echo "=== E5: the transition curve, identical grid to stage DQ ==="
# --seed-start left at its default, as in the parent, so seed 2 is evaluated
# on the SAME 400 instances per bin as seeds 0 and 1 and is paired with them
# point for point. The grid values are unchanged; changing them would break
# the comparison this stage exists to make.
python3 -u -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.60 0.50 0.40 0.32 0.27 0.22 0.18 0.15 \
    --n-per-bin 400 \
    --out-csv "$SCAN/rho_curve_n${N}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== trainability (same readout as the parent's D5) ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
best = min(h, key=lambda r: r["nll"])
print("epochs %d  final train_nll %.4f  final val_nll %.4f  best val_nll %.4f at epoch %d"
      % (len(h), h[-1]["train_nll"], h[-1]["nll"], best["nll"], best["epoch"]))
print("overfit gap (final - best) = %+.4f" % (h[-1]["nll"] - best["nll"]))
tail = [r["nll"] for r in h[-5:]]
print("last 5 val_nll:", " ".join("%.4f" % x for x in tail),
      "  drop over them %+.4f" % (tail[-1] - tail[0]))
print("first val_nll %.4f -> final %.4f" % (h[0]["nll"], h[-1]["nll"]))
PY

echo
echo "=== E1-E4 need the section split, which this job does NOT produce. Chain:"
echo "      sbatch --dependency=afterany:\$SLURM_ARRAY_JOB_ID \\"
echo "          scripts/slurm/sudoku2_dq3_split_2026-11-22.sbatch"
echo "    Use scripts/stagedq_probe.py / stagedq_fit.py pointed at"
echo "    sudoku_dq3_2026-11-22 -- NEVER scripts/stagensplit_fit.py, whose"
echo "    FLOOR dict is keyed on N alone and applies a q=4 val_nll floor to a"
echo "    q=3 cell (and KeyErrors outright at N=120)."
