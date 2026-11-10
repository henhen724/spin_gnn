#!/bin/bash
# =============================================================================
# STAGE P -- is the depth a COMPOSITION of 16 maps, or ONE map iterated 16 times?
#
# THE BRIEF CHANGED ON 2026-11-10. The goal is a conceptually simple model that
# might be theoretically ANALYZABLE, explicitly NOT a model with the fewest
# parameters. Stage P is the first cut registered under that objective, and it
# is chosen for what it does to the OBJECT rather than to the count.
#
# WHAT THE OBJECT BECOMES
#
#   untied (J2, the standing cell)      tied (P1, this stage)
#     h <- Emb(x) + Pos                   h <- Emb(x) + Pos
#     for i in 1..16:                     for i in 1..16:
#         h <- Block_i(h)                     h <- Block(h)        <- SHARED
#     out <- Readout(h)                   out <- Readout(h)
#
#     a composition of 16 distinct       a fixed-point iteration
#     maps; "layer 9" is its own           h_{t+1} = F(h_t);
#     object with its own weights          "layer 9" is the 9th application
#                                          of ONE operator
#
# Direct deduction proceeds in SYNCHRONOUS ROUNDS: every forced node is emitted,
# then the next round reads the result. A tied stack is the same shape of
# computation -- one update rule applied until nothing more is forced -- so if
# the tied cell learns DD then "one layer = one deduction round" stops being an
# analogy about a trained network and becomes a structural property of the
# architecture. That is the whole reason this cut leads, and the parameter drop
# is a side effect, not the point.
#
# WHAT CHANGES: EXACTLY ONE KNOB
#
#     J2  no_mlp      L=16 d192   2,504,890 trainable   rho_50 0.580   3/3  FREE
#     P1  tied_no_mlp T=16 d192     270,010 trainable   ?                   this
#
# Same depth, same width, same nhead 4, same ff 1024, same warmup 0, same
# 240k x 50ep budget, same `--shuffle-adj --fall` ensemble, same 14-point grid
# to 0.33 at 400 per bin, same seeds 0/1/2. The ONLY difference is that the 16
# blocks are one block. 9.28x fewer trainable parameters falls out of that.
#
# THE CONFOUND, NAMED IN ADVANCE. 270,010 is BELOW H1 (L=12 d192... no: L=12
# d128, 879,418) which CENSORED at 1/3 seeds. So a censor here is consistent
# with two different stories -- tying costs the deduction, or 270k parameters
# is simply too few -- and this stage cannot separate them on its own. P1 in
# scripts/stagep_score.py names the follow-up that does, registered now rather
# than chosen after the numbers land: a tied cell at d320 (613,818) or d448
# (1,088,698), which holds the sharing fixed and buys the capacity back.
#
# WHY NOT A WIDER TIED CELL IN THIS SAME ARRAY. Only three GPUs are free on
# sganguli (stage M holds two, stage O's surviving arm holds three), and the
# programme's rule is one change at a time. Matching J2's shape exactly is what
# makes this a statement about SHARING; a second width in the same stage would
# make every reading a two-variable one.
#
# WHAT WAS CHECKED BEFORE THIS WAS SUBMITTED (scripts/_tied_dry_check.py, 7/7):
# the untied path still counts 2,504,890 exactly, so nothing the ladder rests on
# regressed; the tied cell counts 270,010 = (630d+58) + (4d^2+8d) exactly;
# encoder.layers has ONE entry while n_iters is 16; the tied forward differs
# from the same weights applied once by 2.81 in max-abs logit, so the iteration
# is not silently collapsing; the feedforward writes identically zero; gradient
# reaches the shared block and the frozen MLP gets none; and a saved tied
# checkpoint reloads with num_layers=16 and reproduces its logits bit for bit.
#
# NON-NEGOTIABLES CARRIED FORWARD: `--shuffle-adj` on every cell (findings
# 16/17/23), 3 seeds with the spread quoted raw AND conditioned on the plateau
# criterion (findings 41/42/47/51/56), validation token accuracy read BEFORE
# rho_50 (findings 4+6; a rung that falls OFF acc_vs_solve's curve is the
# interesting one), and the scorer registered before the first cell lands --
# this array is submitted `--dependency=afterok` on the registration job.
#
# N=48 ONLY. Finding 55: the N=48 minimum does not transfer to N=96, so every
# claim from this stage is an N=48 claim and is labelled as one.
# =============================================================================

#SBATCH --job-name=sud2N48TiedP
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -euo pipefail

# d_model:num_layers:seed -- one rung, three seeds. num_layers is the number of
# TIMES THE SHARED BLOCK IS APPLIED, which is what joins this cell to the depth
# ladder; it no longer costs parameters.
SPECS=(
  "192:16:0"  "192:16:1"  "192:16:2"
)

TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r DMODEL LAYERS SEED <<< "${SPECS[$TASK]}"
DMODEL=$((10#$DMODEL)); LAYERS=$((10#$LAYERS)); SEED=$((10#$SEED))

FF=1024
NHEAD=4
WARMUP=0
ARCH=tied_no_mlp

LL=$(printf '%02d' "$LAYERS")
HH=$(printf '%02d' "$NHEAD")
TT=$(printf '%02d' "$TASK")
DD=$(printf '%04d' "$DMODEL")
FFS=$(printf '%05d' "$FF")

case "$ARCH" in
  full)        ATAG="" ;;
  linear)      ATAG="_lin" ;;
  rf_mlp)      ATAG="_rfm" ;;
  rf_all)      ATAG="_rfa" ;;
  no_mlp)      ATAG="_noff" ;;
  tied_no_mlp) ATAG="_noff_tied" ;;
  *) echo "unknown arch_mode: $ARCH"; exit 1 ;;
esac

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"

echo "=== smoke check ==="
python3 - <<'PY'
import torch, numpy, numba
import spin_gnn.sudoku_hparam_sweep  # noqa: F401
import scripts.rho_transition_eval   # noqa: F401
from spin_gnn.sat_transformer2 import ARCH_MODES
from spin_gnn.sudoku_hparam_sweep import ARCH_TAG
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
print("arch modes", ARCH_MODES)
assert "tied_no_mlp" in ARCH_MODES, ("the tied mode is not in this checkout -- "
                                     "the array outran the registration commit", ARCH_MODES)
assert ARCH_TAG["tied_no_mlp"] == "_noff_tied", ARCH_TAG
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_tied_P_2026-11-10"
cell="${OUT}/t${TT}_d${DD}_ff${FFS}_h${HH}_${ARCH}_L${LL}_s${SEED}"
mkdir -p "$cell"

echo "task=${TASK}  layers=${LAYERS} d_model=${DMODEL} ff=${FF} nhead=${NHEAD} arch=${ARCH} seed=${SEED}"
echo "cell -> ${cell}"
date +'cell start %F %T'

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model "$DMODEL" --nhead "$NHEAD" --dim-feedforward "$FF" \
    --num-layers "$LAYERS" --arch-mode "$ARCH" --seed "$SEED" \
    --warmup-epochs "$WARMUP" \
    --n-train-instances 240000 --epochs 50 \
    --n-val-instances 300 --batch-size 32 --n-per-bin 100 \
    --out-dir "$cell"

date +'cell end   %F %T'

TAG="d${DD}_h${HH}_L${LL}_ff${FFS}_lr1e-03_pre_shuf_fall${ATAG}"
if [ "$WARMUP" -ne 0 ]; then
    TAG="${TAG}_w$(printf '%02d' "$WARMUP")"
fi
if [ "$SEED" -ne 0 ]; then
    TAG="${TAG}_s$(printf '%02d' "$SEED")"
fi
CKPT="${cell}/${TAG}"
[ -f "${CKPT}/model.pt" ] || { echo "expected checkpoint missing: ${CKPT}"; ls -1 "$cell"; exit 1; }

echo
echo "=== W: what the checkpoint says it is (read back, not trusted) ==="
python3 - "${CKPT}/config.json" "$ARCH" "$LAYERS" "$NHEAD" "$DMODEL" "$FF" "$WARMUP" "$SEED" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
want_arch, want_L, want_h = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
want_d, want_ff, want_w = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
want_seed = int(sys.argv[8])
assert c["shuffle_adj"] is True, "LEAK: this cell is not shuffle_adj"
assert c["fall"] is True, "wrong ensemble: fall is not set"
assert c["arch_mode"] == want_arch, (c["arch_mode"], want_arch)
assert c["num_layers"] == want_L and c["nhead"] == want_h
assert c["d_model"] == want_d, (c["d_model"], want_d)
assert c["d_model"] % c["nhead"] == 0, ("d_model must divide by nhead",
                                        c["d_model"], c["nhead"])
assert c["dim_feedforward"] == want_ff, (c["dim_feedforward"], want_ff)
assert int(c.get("warmup_epochs", 0) or 0) == want_w, (c.get("warmup_epochs"), want_w)
assert c["max_len_margin"] == 2 * c["n_nodes"] + 2, c["max_len_margin"]
assert c["n_nodes"] == 48, ("stage P is N=48 only (finding 55)", c["n_nodes"])
assert int(c["seed"]) == want_seed, ("wrong seed; the triple would not pool",
                                     c["seed"], want_seed)
assert want_seed in (0, 1, 2), ("stage P is the first triple of a new rung", want_seed)
d, L = c["d_model"], c["num_layers"]
# 630*d + 58 is the N=48 fixed term (vocab 58, max_len 512, final LayerNorm
# included); asserted above that this IS N=48. A TIED cell pays the per-block
# term ONCE however many times the block is applied -- that identity is the
# whole stage, so it is asserted rather than described.
BLOCK = 4 * d * d + 8 * d
pred_tied = (630 * d + 58) + BLOCK
pred_untied = (630 * d + 58) + BLOCK * L
got = c.get("n_trainable_params", -1)
print("arch_mode=%s T=%d nhead=%d d_model=%d ff=%d warmup=%d seed=%d  trainable %s "
      "params  max_len=%d (margin %d)"
      % (c["arch_mode"], L, c["nhead"], d, c["dim_feedforward"],
         int(c.get("warmup_epochs", 0) or 0), int(c["seed"]), format(got, ","),
         c["max_len"], c["max_len_margin"]))
assert got == pred_tied, ("this is not a TIED cell: the count matches neither the "
                          "registered tied prediction nor anything else",
                          got, pred_tied, pred_untied)
assert got != pred_untied, "tied and untied counts coincide; the comparison is void"
RUNGS = {270010: "P1  tied_no_mlp T=16 d192"}
assert got in RUNGS, ("not the registered rung", got, sorted(RUNGS))
TWIN = 2504890      # J2, no_mlp L=16 d192, the untied twin at the SAME shape
print("%s -- same shape as J2, %.2fx fewer trainable parameters (%s vs %s), and "
      "the depth %d is now applications of ONE block rather than %d blocks"
      % (RUNGS[got], TWIN / got, format(got, ","), format(TWIN, ","), L, L))
FLOOR = 879418      # H1, no_mlp L=12 d128, CENSORED at 1/3 seeds
print("NOTE for P1: %s is BELOW the untied censoring floor %s (H1), so a censor "
      "here does not by itself separate 'tying costs the deduction' from '270k is "
      "too few'. The registered follow-up is a tied cell at d320 (613,818) or "
      "d448 (1,088,698)." % (format(got, ","), format(FLOOR, ",")))
PY

echo
echo "=== the transition curve for this cell (P0, P1) ==="
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== W: plateau escape (P2) ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
PLATEAU = 1.64          # finding 44, an N=48 number
h = json.load(open(sys.argv[1]))
best = min(h, key=lambda r: r["nll"])
final_train = h[-1]["train_nll"]
tail = [r["train_nll"] for r in h[-5:]]
slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
first = next((r["epoch"] for r in h if r["train_nll"] < PLATEAU), None)
print("epochs %d  final train_nll %.4f  final val_nll %.4f  best val_nll %.4f at epoch %d"
      % (len(h), final_train, h[-1]["nll"], best["nll"], best["epoch"]))
# history.json's key is "accuracy"; the TRAINING LOG prints it as val_accuracy=.
print("final val TOKEN ACCURACY %.4f  (P3 reads this BEFORE rho_50; it never censors)"
      % h[-1]["accuracy"])
print("tail slope over the last 5 epochs: %+.5f nll/epoch" % slope)
if first is None:
    print("PLATEAU NOT ESCAPED: train_nll never went below %.2f (min %.4f). For a "
          "TIED cell this is the predicted failure mode -- one block must serve 16 "
          "roles -- and P2 is registered to be read BEFORE any level."
          % (PLATEAU, min(r["train_nll"] for r in h)))
else:
    print("plateau escaped at epoch %d (train_nll < %.2f)" % (first, PLATEAU))
print("steps bound: %d epochs completed of the registered 50" % len(h))
assert len(h) == 50, ("the run did not reach the registered 50 epochs; a "
                      "truncated cell does not compare with its matched rung",
                      len(h))
PY

echo
echo "=== done ==="
date +'%F %T'
echo "score with: python3 -u -m scripts.stagep_score"
