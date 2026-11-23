#!/bin/bash
#SBATCH --job-name=sud2DQ4
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --array=0-4%3
#SBATCH --output=%x_%A_%a.out
#
# STAGE DQ4 -- the seed distribution at ONE size. Five new seeds, 3..7, all at
# N=120, d=4, q=3. ONE knob moves against stages DQ (seeds 0,1) and DQ3
# (seed 2): the seed. Same ensemble, same architecture, same data budget, same
# eval grid, same 400 instances per bin. With the three existing cells this
# gives n=8 at a single size.
#
#   sbatch scripts/slurm/sudoku2_dq4_2026-11-22.sbatch
#
# PREDICTIONS F1-F5 AND A FALSIFIER FOR FINDINGS 119/120 ARE REGISTERED IN
# runs/stagedq4_2026-11-22/REGISTERED.md BEFORE this job was submitted. Read
# that file, not this header.
#
# WHY. Stage DQ3 measured a third seed and found the spread between seeds at
# fixed N larger than the effect of N, with the size ordering REVERSED inside
# seed 2:
#     N=120 s0  reason_color 1.0987 (chance)   a_eff < 0.94202 censored
#     N=120 s1  reason_color 0.0001 (exact)    a_eff < 0.94202 censored
#     N=120 s2  reason_color 0.0002 (exact)    a_eff   0.98542
# Best-to-best the N=96 -> N=120 penalty is 0.99191 -> 0.98542 = 0.0065, while
# two of three cells at the SAME size cannot be measured at all. Every
# comparison this project has drawn on this ensemble is a two- or three-cell
# difference, so none of them is readable until the spread is known. This stage
# measures the spread; it is not an attempt to win a number, and F5 (the
# standing 0.99191 bar) is registered expecting a MISS.
#
# SEED COUNT IS NOT A CLOSED KNOB -- the closed list is width, cosine decay,
# more data, best-val selection, role count, schedule length, training budget,
# and this run re-tests none of them. It is also the knob STATUS.md named as
# the honest next move.
#
# WALLTIME IS MEASURED. sacct -X on 53243550 (N=120): 13:15:05 and 13:11:43;
# 53313361 task 1 (N=120, seed 2) agreed. --time=18:00:00 is ~36% headroom.
#
# THROTTLED TO 3 CONCURRENT ON PURPOSE. At submit, sganguli's only undrained
# GPU node (sh04-03n01) had all 8 GPUs held by outside `owners` jobs, which
# preempt as soon as this group asks. %3 takes three of them rather than five
# and finishes in two waves (~26 h) instead of one (~13 h); the science does
# not need it tonight.
#
# TRAPS CARRIED FORWARD
#  1. Chance is ln 3 = 1.098612 and chance accuracy is 1/3. NOTHING FROM q=4
#     IS QUOTED: not the free line 0.740, s* 0.1484/0.119, the val_nll floors
#     1.371/1.625/1.861, (5/16) log N, nor ln 4 itself.
#  2. Scoring is PER CELL. Pooling is what produced stage DQ's misleading D2,
#     so any pooled number is reported beside its parts. With 8 cells the
#     summary statistic is the spread, and the parts are still listed.
#  3. IF A BIN SOLVES 0 OF 400, a_eff = 0 IS NOT REPORTED. Zero successes
#     bounds the rate: p < 1 - 0.05^(1/400) = 0.00746136, a_eff < p^(1/n_blank).
#     At N=120, rho=0.32, n_blank=82 the bound is < 0.94202 and is IDENTICAL
#     for every censored seed, so it is drawn ONCE in grey, never per seed.
#     The bound RISES with T purely because n_blank grows -- an artefact, and
#     it must be labelled as one wherever it is drawn.
#  4. A censored cell counts toward F2 (reason_color exists for every cell) but
#     not toward F3 (a_eff exists only where something was solved). Report the
#     censored count beside the spread.
#  5. val_nll CANNOT RANK THESE CELLS. In DQ3, 0.019 nats of aggregate loss sat
#     on top of a 47x difference in whole-instance solving. Health check only.
#  6. CHECK FOR THE ARTEFACT, NOT THE EXIT CODE. DQ3's split job exited 0:0
#     with no verdict because a `|| true` masked a scorer crash. Nothing in the
#     DQ4 chain wraps a scorer in `|| true`.
#
set -euo pipefail

cd /home/users/hshunt/spin_gnn

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 \
    py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312

# task -> seed. N is 120 on EVERY cell; the seed is the only thing that moves,
# both within this array and against DQ/DQ3.
case "$SLURM_ARRAY_TASK_ID" in
    0) SEED=3 ;;
    1) SEED=4 ;;
    2) SEED=5 ;;
    3) SEED=6 ;;
    4) SEED=7 ;;
    *) echo "bad array index"; exit 1 ;;
esac
N=120
DEG=4
NCOL=3
RHO0=0.14

# A NEW DATED DIRECTORY: nothing is written next to DQ's or DQ3's cells, so
# every already-scored artefact stays exactly as it was scored.
SCAN=$SCRATCH/spin_gnn/sudoku_dq4_2026-11-22
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
echo "=== F3/F5: the transition curve, identical grid to stages DQ and DQ3 ==="
# --seed-start left at its default, as in both parents, so every seed is
# evaluated on the SAME 400 instances per bin and the eight cells are paired
# point for point. Changing the grid would break the comparison this stage
# exists to make.
python3 -u -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.60 0.50 0.40 0.32 0.27 0.22 0.18 0.15 \
    --n-per-bin 400 \
    --out-csv "$SCAN/rho_curve_n${N}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== trainability (health check only -- see trap 5) ==="
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
echo "=== F1/F2/F4 need the section split, which this job does NOT produce. Chain:"
echo "      sbatch --dependency=afterany:\$SLURM_ARRAY_JOB_ID \\"
echo "          scripts/slurm/sudoku2_dq4_split_2026-11-22.sbatch"
echo "    Use scripts/stagedq_probe.py pointed at sudoku_dq4_2026-11-22, then"
echo "    scripts/stagedq4_fit.py -- NEVER scripts/stagensplit_fit.py, whose"
echo "    FLOOR dict is keyed on N alone and applies a q=4 val_nll floor to a"
echo "    q=3 cell (and KeyErrors outright at N=120)."
