#!/bin/bash
#SBATCH --job-name=sud2DQ
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --array=0-3%4
#SBATCH --output=%x_%A_%a.out
#
# STAGE DQ -- the (degree, n_colors) depth lever. hshunt chose this on
# 2026-11-21 over the trace-format change (f) and a fourth size rung.
#
#   sbatch scripts/slurm/sudoku2_dq_2026-11-21.sbatch
#
# PREDICTIONS ARE REGISTERED IN runs/stagedq_2026-11-21/REGISTERED.md,
# commit 496c766, BEFORE this job was submitted. D1 is the discriminator and
# it registers BOTH arms with a numeric bar and registers how to read the
# band between them, so a partial result cannot be rounded afterwards to
# whichever hypothesis is convenient. Read that file, not this header, for
# what counts as a hit.
#
# THE QUESTION. Finding 116: at d=5 q=4 the model's reason_color sits at
# ln 4 = 1.3863 nats with accuracy 1/q at N=192, every density, both seeds,
# while at N<=96 it is near-perfect (0.0036, 0.9991). Two causes fit every
# number we have, and a SIZE LADDER CANNOT SEPARATE THEM because N and DD
# depth move together along it:
#   H-N  it is N -- vocabulary, graph-section length, node identities;
#   H-T  it is the DD depth that comes with larger N.
# (d=4, q=3) separates them by buying depth from the LOCAL parameters at
# fixed N. Measured in runs/stagedq_ensemble_2026-11-21/, deepest decile:
#   d4q3 N=96   T=32.8  len 687  vocab 105
#   d4q3 N=120  T=35.0  len 856  vocab 129
#   d5q4 N=96   T=22.5  len 781  vocab 106   <- control, already on disk
#   d5q4 N=192  T~35.4  len ~1560             <- the cell that failed
# So N=96 here is DEEPER than the q=4 cell at the same N while being
# SHORTER, and N=120 matches the depth of the failed cell at 1.8x fewer
# tokens -- both at an N where the colouring rule is known learnable.
#
# NOTHING FROM q=4 CARRIES OVER AND NOTHING FROM q=4 IS QUOTED. rho_c moves
# ~0.465 -> ~0.25 (measured where T/log N stops collapsing across N=48..384;
# the same diagnostic recovers the published 0.465 on the d5q4 arm, so it is
# calibrated). rho_min moves ~0.314 -> ~0.150. The chance level moves to
# ln 3 = 1.0986. BANNED from the verdict and the deck: the free line 0.740,
# s* 0.1484/0.119, the val_nll floors 1.371/1.625/1.861, (5/16) log N, and
# ln 4 itself.
#
# FINDING 10 NEEDS AMENDING AND THIS STAGE DOES NOT LEAN ON IT. Its "~6x
# cheaper" came from sweep 49509101, whose scorer never passes `fall`.
# Reproduced on that ensemble: 5.00x in N, 5.48x in tokens. On fall=True,
# which every training cell uses: 1.58x and 1.77x. The fall constraint helps
# the q=4 reference far more than the lever (at N=192 it moves rho_min
# 0.4729 -> 0.3134 against 0.1944 -> 0.1437). No prediction depends on the
# factor; the reason to run this is the depth-vs-N separation above.
#
# PREFLIGHT PASSED BEFORE SUBMIT. scripts/stagedq_datacheck.py on CPU, 40
# instances per cell: F1 every deduction forced by its prior-round
# neighbours, F2 deduced colour == planted, F3 forcing context already
# emitted, F4 node tokens round-trip under the moved node_offset (10 -> 9)
# -- 0 bad on all four, on d4q3 N=96, d4q3 N=120 and the d5q4 N=96 control.
# Measured marginal on reason_color 1.0985 against ln 3 = 1.098612, so a
# trained cell landing there is finding 116 repeating in this ensemble.
# GPU timing probe 53242607 (COMPLETED 00:15:32, exit 0:0) confirmed q=3
# trains at all, which had never been shown on GPU:
#   N=96  d4q3  max_len  919  9,766,761 trainable  3 ep x 4800 in 364 s
#   N=120 d4q3  max_len 1139  9,835,393 trainable  3 ep x 4800 in 555 s
# val_nll fell monotonically on both (3.9933 -> 3.7564 and 4.1854 -> 4.0136).
#
# WALLTIME IS SIZED CONSERVATIVELY AND THE BASIS IS STATED, because unlike
# the N=192 rung this probe could NOT isolate s/epoch: the sweep does not
# timestamp epochs and history.json carries no wall time, so the spans above
# are gen + 3 epochs + a 14x20 eval and are dominated by the one-off terms.
# Two independent handles, neither of them a clean per-epoch measurement:
#   (a) token-count scaling off the MEASURED d5q4 N=96 full run (13.6 h at
#       240k x 50). Mean encoded length 687 vs 781 -> ~12 h at N=96 d4q3,
#       and 856 -> ~15 h at N=120.
#   (b) the probe's own N=120/N=96 span ratio, 555/364 = 1.52, which is
#       STEEPER than the 1.25x token ratio -- consistent with the attention
#       term plus per-instance generation cost. Taking N=120 as 1.52x of
#       (a)'s 12 h gives ~18 h.
# --time=30:00:00 carries ~67% headroom over the larger of the two
# estimates. 4 cells is ~60 GPU-h, against the N=192 rung's 66.
# IF A CELL HITS THE WALL ANYWAY that is a sizing failure, not a result:
# requeue once with a measured time, and do not read a truncated history.
#
# TRAPS CARRIED FORWARD
#  1. Chance is ln 3 = 1.0986. A cell AT 1.3863 here is WORSE than chance.
#  2. Identical eval grid VALUES do not license pooling across ensembles.
#     The cross-ensemble axis is dd_mean_T.
#  3. Whole-instance solve rate is not comparable across ensembles: n_blank
#     is ~82 at d4q3 N=96 rho=0.15 against ~65 at d5q4 N=96 rho=0.32.
#     Normalise to a_eff = solve_rate^(1/n_blank) first.
#  4. IF A BIN SOLVES 0 OF 400, a_eff = 0 IS NOT REPORTED. Zero successes
#     bounds the rate, it does not pin it: p < 1 - 0.05^(1/400) = 0.00746
#     and a_eff < p^(1/n_blank). This is what kept N1 honest at N=192.
#  5. No closed knob may be re-opened if this nulls. Not width, cosine
#     decay, more data, best-val selection, role count, schedule length or
#     training budget. A null points at (f), which is hshunt's call.
#
set -euo pipefail

cd /home/users/hshunt/spin_gnn

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 \
    py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312

# task -> (N, seed). d and q are fixed at 4 and 3 for every cell: ONE knob
# changes relative to the N=96 d5q4 control on disk, and that knob is (d,q).
case "$SLURM_ARRAY_TASK_ID" in
    0) N=96;  SEED=0 ;;
    1) N=96;  SEED=1 ;;
    2) N=120; SEED=0 ;;
    3) N=120; SEED=1 ;;
    *) echo "bad array index"; exit 1 ;;
esac
DEG=4
NCOL=3
RHO0=0.14

SCAN=$SCRATCH/spin_gnn/sudoku_dq_2026-11-21
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
# Legality at q=3, d=4 reduces to 3 | N, against 12 | N at q=4 d=5. Assert it
# rather than trusting the table above: an illegal cell raises deep inside
# planted_regular_graph after the job has already queued.
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

# Locate the checkpoint by globbing rather than reconstructing the tag
# string: the q=4 tag convention is not guaranteed to be ensemble
# independent, and a wrong guessed path fails AFTER the training spend.
CKPT=$(dirname "$(find "$CELL" -name model.pt | head -1)")
[ -f "${CKPT}/model.pt" ] || { echo "no checkpoint under $CELL"; ls -R "$CELL"; exit 1; }
echo "checkpoint: $CKPT"

echo
echo "=== D4: the transition curve, straddling this ensemble's rho_c ~ 0.25 ==="
# Four bins BELOW rho_c and bottoming at the measured rho_min ~ 0.150, so the
# slow phase is actually sampled -- the q=4 grid stopped at 0.32 and would
# see none of it here. --seed-start left at its default so both seeds get the
# SAME 400 instances per bin and are paired point for point.
python3 -u -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.60 0.50 0.40 0.32 0.27 0.22 0.18 0.15 \
    --n-per-bin 400 \
    --out-csv "$SCAN/rho_curve_n${N}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== D5: trainability ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
best = min(h, key=lambda r: r["nll"])
print("epochs %d  final train_nll %.4f  final val_nll %.4f  best val_nll %.4f at epoch %d"
      % (len(h), h[-1]["train_nll"], h[-1]["nll"], best["nll"], best["epoch"]))
print("overfit gap (final - best) = %+.4f   [D5 falsifier: > 0]"
      % (h[-1]["nll"] - best["nll"]))
tail = [r["nll"] for r in h[-5:]]
print("last 5 val_nll:", " ".join("%.4f" % x for x in tail),
      "  drop over them %+.4f" % (tail[-1] - tail[0]))
print("first val_nll %.4f -> final %.4f" % (h[0]["nll"], h[-1]["nll"]))
# D5's breaking point: flat at the initial plateau for all 50 epochs means
# not trainable at this budget. That IS the result and is not retried larger.
PY

echo
echo "=== D1/D2/D3 need the section split; scored after the array by"
echo "    scripts/stagensplit_fit.py pointed at $SCAN (chance = ln 3)."
