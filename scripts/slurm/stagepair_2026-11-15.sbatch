#!/bin/bash
# =============================================================================
# STAGE PAIR -- SUPERVISE THE CONTRAST, NOT THE RANKING. Zero new parameters.
#
# WHERE THIS STARTS, and it starts from a measurement, not a hunch.
#
#     S1       tied_no_mlp T=16 d320 k=1   613,818   rho_50 0.754/0.757/0.765
#              val token accuracy 0.4830/0.4830/0.4833     free line 0.740
#     stage X  the same cell with a determinacy BCE on the READOUT's node
#              logits. Verdict: representation_without_use. AUROC 0.947,
#              guard rate still 1.0, rho_50 0.866/0.864/0.855.
#     stage Y  the BCE moved to its own channel g = <h*s, E_v>, +320 params,
#              emission z + log sigma(g) on the node block. rho_50
#              0.772/0.758/censored -- no better than the control.
#     stage Z  WHY, and this is the finding this stage is built on (78,
#              partial_predicate_not_causally_wired). Stage Y's g DOES hold a
#              partial predicate: WITHIN a stratum of fixed revealed-neighbour
#              count it ranks nodes by determinacy at AUROC 0.8059/0.8021/
#              0.8019. And on a MATCHED PAIR -- one prompt, and the same prompt
#              with v's forcing destroyed by recolouring a single hint -- it
#              orders the two at 0.4457/0.5714/0.4543, against a placebo's
#              0.4800/0.4686/0.4714. Chance, both. Median |delta g| under the
#              corruption is 0.0103 where the placebo's is 0.0102, against g's
#              own spread of 3.5, and the emitted node flips in 0 of 236/228/217
#              cases where the clean prompt named v.
#
# THE DIAGNOSIS. Stage X and stage Y both supervised a RANKING OVER NODES ON
# CLEAN INPUTS. Nothing in either objective ever asked g to MOVE when the
# property it names stops being true. A ranker that reads "how many distinct
# colours do v's revealed neighbours show" gets AUROC 0.80 within a stratum and
# needs no counterfactual sensitivity whatsoever -- which is exactly the
# model-free count's 0.8857 against the model's 0.9493, and exactly what the
# matched pair measures as chance.
#
# THE ONE THING THIS CHANGES, and it is an objective, not an architecture:
#
#     L_pair = max(0, m - (g[p, v] - g_broken[p, v]))      m = 1.0, lambda = 1.0
#
# with p the position whose next token is v's round-0 node token, v the node the
# edit un-forces, and g_broken the SAME channel on the SAME prompt with one hint
# colour token rewritten. The corrupted forward feeds the margin and nothing
# else: after the edit the planted trace is no longer a correct continuation, so
# its readout is computed and deliberately unread.
#
# PRICE: ZERO PARAMETERS. 614,138 trainable, stage Y's exactly, and the state
# dict is byte-compatible with stage Y's -- which is why the config read-back
# below asserts pair_weight, since nothing else in the artifact distinguishes
# the two objectives.
#
# WALL TIME, measured rather than guessed (52415691, runs/2026-11-15_stagepair_
# price): the scan yields 0.8034 of instances (16,068/20,000) at 0.04 ms each,
# free against build_instances' own 0.94 ms; the step goes 76.55 -> 151.43 ms,
# 1.978x, and peak memory 4.55 -> 8.99 GiB. That projects 22.0 min/epoch and
# 18.3 h for 50 epochs. 36:00:00 is requested against that; sganguli's limit is
# 7 days, so the budget is not the constraint.
#
# HELD FIXED, so this is ONE knob: the same tied cell (tied_no_mlp T=16 d320
# ff1024 h4 k=1), the same channel, the same determinacy BCE at the same
# lambda = 1.0 on the same targets, the same 240k instances, the same 50 epochs,
# the same seeds 0/1/2, the same --deterministic kernels, the same 14-point rho
# grid at 400 per point. The ONLY difference from stage Y is the extra term.
#
# REGISTERED PREDICTIONS -- scripts/stagepair_score.py, whose docstring IS the
# registration, committed at 9f4edad BEFORE this launcher was written and
# before one cell trained. The bars are IMPORTED below, never retyped.
#
#   P1  THE POINT. Matched-pair AUROC >= 0.65 in >= 2 of 3 seeds, against
#       stage Z's measured 0.4457/0.5714/0.4543. Predicted 0.85 +- 0.10.
#   P2  THE CONFOUND, AND IT VOIDS P1. The corrupted prompt differs by exactly
#       ONE TOKEN, so L_pair is satisfiable by a detector for THE EDIT rather
#       than for the predicate -- a strictly simpler function that a
#       gradient-following model has every reason to find first. The placebo
#       separates them: a hint recolour that is NOT v's neighbour, leaving v
#       forced, and NEVER TRAINED ON. Placebo AUROC must stay in [0.40, 0.60].
#       Outside it, the verdict is edit_detector and P1 means nothing.
#   P3  THE RANKING MUST SURVIVE. Pooled within-stratum AUROC >= 0.90, i.e. the
#       contrast was not bought by destroying what stage Y already had.
#   P4  BEHAVIOUR. The conditional guard rate, finding 68's probe unchanged.
#       Predicted NOT to fire -- a representation that tracks determinacy under
#       intervention is still not a schedule that uses it.
#
# WHAT A NULL WOULD MEAN. P1 refuted with P2 and P3 passing says the contrast is
# not learnable from this coupling at all, which would be the strongest evidence
# yet that the tied cell's DEPTH, not its objective, is the binding constraint.
# P1 with P2 refuted says the model learned the edit and not the predicate, and
# is a negative result about the training signal's identifiability rather than
# about the architecture.
#
# WHAT THIS CANNOT SHOW: the margin is trained on ONE case per instance at ONE
# position, so a positive says the channel can be made counterfactually
# sensitive where it is supervised and does not establish that it generalizes to
# deductions of other kinds. The placebo is the only out-of-sample check here
# and it is a negative control, not a generalization test.
#
# SMOKE-TESTED BEFORE TRAINING, IN THE JOB: scripts/stagepair_smoke.py, seven
# assertions. Three of them are about MAGNITUDE, which is the lesson of finding
# 76 -- stage Y's first array was cancelled at 2h29 because five algebraically
# correct assertions never looked at the scale of what they asserted.
#
# NON-NEGOTIABLES CARRIED FORWARD: `--shuffle-adj` on every cell (16/17/23), 3
# seeds with the spread quoted (41/42/47/51/56), validation token accuracy read
# BEFORE rho_50 (4+6), ONE knob at a time, N=48 only (finding 55).
# =============================================================================

#SBATCH --job-name=sud2N48Pair
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -euo pipefail

# pair_weight:seed -- ONE weight, three seeds. Everything else is stage Y's
# cell, unchanged.
SPECS=(
  "1.0:0"  "1.0:1"  "1.0:2"
)

TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r PAIRW SEED <<< "${SPECS[$TASK]}"
SEED=$((10#$SEED))

LAYERS=16
DMODEL=320
KBLOCKS=1
FF=1024
NHEAD=4
WARMUP=0
EPOCHS=50
ARCH=tied_no_mlp
DETW=1.0
PAIRM=1.0

LL=$(printf '%02d' "$LAYERS")
HH=$(printf '%02d' "$NHEAD")
TT=$(printf '%02d' "$TASK")
DD=$(printf '%04d' "$DMODEL")
FFS=$(printf '%05d' "$FF")

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
# Read ONCE at cuBLAS init, so it must be exported before python starts.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${SLURM_SUBMIT_DIR}"

echo "=== smoke check ==="
python3 - "$DMODEL" "$NHEAD" "$KBLOCKS" "$LAYERS" "$EPOCHS" "$DETW" "$PAIRW" "$PAIRM" <<'PY'
import sys
import torch, numpy, numba
import spin_gnn.sudoku_hparam_sweep  # noqa: F401
import scripts.rho_transition_eval   # noqa: F401
import scripts.stagepair_score as sp
from spin_gnn.sat_transformer2 import ARCH_MODES, SATReasoningTransformer
from spin_gnn.sudoku_pair import build_pair_index  # noqa: F401
from spin_gnn.sat_dataset2 import make_token_batch_pair  # noqa: F401
d, nhead, k, L, ep = (int(a) for a in sys.argv[1:6])
detw, pairw, pairm = (float(a) for a in sys.argv[6:9])
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
assert "tied_no_mlp" in ARCH_MODES, ("the tied mode is not in this checkout", ARCH_MODES)
assert d == 320 and nhead == 4 and k == 1, (
    "stage PAIR is stage Y's cell with ONE term added and nothing else moved",
    d, nhead, k)
assert L == 16 and ep == 50, ("stage PAIR holds the application count and the "
                              "budget at stage S's", L, ep)
assert detw == 1.0, ("the determinacy BCE is held at stage Y's lambda", detw)
assert pairw == sp.LAMBDA_PAIR == 1.0, ("this is not the registered pair weight",
                                        pairw, sp.LAMBDA_PAIR)
assert pairm == sp.MARGIN == 1.0, ("this is not the registered margin",
                                   pairm, sp.MARGIN)

m = SATReasoningTransformer(vocab_size=58, d_model=d, num_layers=L, nhead=nhead,
                            max_len=512, dim_feedforward=1024, norm_first=True,
                            arch_mode="tied_no_mlp", n_blocks=k,
                            determinacy_channel=True, node_offset=2, n_nodes=48)
# THE PRICE OF THIS STAGE, asserted rather than claimed: nothing. The term reads
# the channel stage Y already paid 320 parameters for.
want = (630 * d + 58) + (4 * d * d + 8 * d) * k + d
got = m.trainable_parameter_count()
assert got == want == sp.N_TRAINABLE == 614138, (got, want, sp.N_TRAINABLE)
print("%s trainable, stage Y's exactly -- the margin adds no parameter"
      % format(got, ","))
print("registered bars: pair>=%.2f  placebo in [%.2f,%.2f]  pooled>=%.2f  "
      "guard<%.2f  in >=%d of 3" % (sp.PAIR_BAR, sp.PLACEBO_BAND[0],
      sp.PLACEBO_BAND[1], sp.POOLED_BAR, sp.GUARD_BAR, sp.MIN_SEEDS))
print("predicted: pair %.2f+-%.2f  placebo %.2f+-%.2f  guard fires %s  "
      "rho_50 worse than stage Y %s"
      % (sp.PRED_PAIR, sp.PRED_PAIR_TOL, sp.PRED_PLACEBO, sp.PRED_PLACEBO_TOL,
         sp.PRED_GUARD_FIRES, sp.PRED_RHO50_WORSE_THAN_Y))
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

echo
echo "=== the eight assertions of the margin, in the job ==="
# Three of these are about MAGNITUDE and that is the point of running them at
# all: finding 76 is a stage whose first array died because five algebraically
# correct assertions never looked at the scale of the term they asserted. At
# init det_scale = 0, so g is identically 0 on both prompts, L_pair is EXACTLY
# the margin, and its gradient reaches only the 320-vector -- each of which is
# checkable before a GPU-hour is spent and invisible afterwards.
python3 scripts/stagepair_smoke.py

echo
echo "=== the determinacy target is pinned against DD itself, not assumed ==="
python3 - <<'PY'
import importlib
m = importlib.import_module("tests.test_sudoku_determinacy")
names = sorted(n for n in dir(m) if n.startswith("test_"))
for n in names:
    getattr(m, n)()
    print("PASS", n)
print("%d determinacy tests pass" % len(names))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_pair_2026-11-15"
cell="${OUT}/t${TT}_d${DD}_ff${FFS}_h${HH}_${ARCH}_L${LL}_dw${DETW}_chan_pw${PAIRW}_s${SEED}"
mkdir -p "$cell"

echo
echo "task=${TASK}  pair_weight=${PAIRW} margin=${PAIRM} determinacy_weight=${DETW} CHANNEL=on applications=${LAYERS} d_model=${DMODEL} ff=${FF} nhead=${NHEAD} arch=${ARCH} k=${KBLOCKS} epochs=${EPOCHS} seed=${SEED}"
echo "cell -> ${cell}"
date +'cell start %F %T'

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model "$DMODEL" --nhead "$NHEAD" --dim-feedforward "$FF" \
    --num-layers "$LAYERS" --arch-mode "$ARCH" --n-blocks "$KBLOCKS" \
    --seed "$SEED" --warmup-epochs "$WARMUP" --deterministic \
    --determinacy-weight "$DETW" --determinacy-channel \
    --pair-weight "$PAIRW" --pair-margin "$PAIRM" \
    --n-train-instances 240000 --epochs "$EPOCHS" \
    --n-val-instances 300 --batch-size 32 --n-per-bin 100 \
    --out-dir "$cell"

date +'cell end   %F %T'

TAG="d${DD}_h${HH}_L${LL}_ff${FFS}_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_chan_pw1"
if [ "$SEED" -ne 0 ]; then
    TAG="${TAG}_s$(printf '%02d' "$SEED")"
fi
CKPT="${cell}/${TAG}"
[ -f "${CKPT}/model.pt" ] || { echo "expected checkpoint missing: ${CKPT}"; ls -1 "$cell"; exit 1; }

echo
echo "=== what the checkpoint says it is (read back, not trusted) ==="
python3 - "${CKPT}/config.json" "$ARCH" "$LAYERS" "$NHEAD" "$DMODEL" "$FF" "$WARMUP" "$SEED" "$KBLOCKS" "$EPOCHS" "$DETW" "$PAIRW" "$PAIRM" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
want_arch, want_L, want_h = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
want_d, want_ff, want_w = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
want_seed, want_k, want_ep = int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10])
want_dw, want_pw, want_pm = (float(a) for a in sys.argv[11:14])
assert c["shuffle_adj"] is True, "LEAK: this cell is not shuffle_adj"
assert c["fall"] is True, "wrong ensemble: fall is not set"
assert c["arch_mode"] == want_arch, (c["arch_mode"], want_arch)
assert c["nhead"] == want_h and c["d_model"] == want_d
assert c["dim_feedforward"] == want_ff, (c["dim_feedforward"], want_ff)
assert int(c.get("warmup_epochs", 0) or 0) == want_w
assert c["n_nodes"] == 48, ("stage PAIR is N=48 only (finding 55)", c["n_nodes"])
assert int(c["seed"]) == want_seed, ("wrong seed; the triple would not pool",
                                     c["seed"], want_seed)
assert int(c.get("n_blocks", 1) or 1) == want_k, ("k moved; stage PAIR holds k=1",
                                                  c.get("n_blocks"))
assert c["num_layers"] == want_L, ("the application count moved; that is stage "
                                   "V's knob and PAIR holds it at 16", c["num_layers"])
assert int(c["epochs"]) == want_ep, ("the budget moved; that is stage W's knob",
                                     c["epochs"])
assert int(c.get("n_train_instances", 240000)) == 240000, (
    "the data size moved, and n_train_instances is NOT in rung_key -- this cell "
    "would pool into stage S's rung and destroy its own control",
    c.get("n_train_instances"))
assert c.get("deterministic") is True, "the pinned-kernel flag did not round-trip"
assert float(c.get("determinacy_weight", 0.0)) == want_dw, (
    "determinacy_weight did not round-trip", c.get("determinacy_weight"), want_dw)
assert c.get("determinacy_channel") is True, (
    "determinacy_channel did not round-trip -- this cell has no channel to "
    "contrast", c.get("determinacy_channel"))
# THE KNOB OF THIS STAGE, and the ONLY place in the artifact it appears. Unlike
# determinacy_channel, pair_weight does not change the state dict: a stage-PAIR
# checkpoint and a stage-Y checkpoint hold the same 614,138 tensors and load
# into each other without a murmur. A trainer that dropped the term would have
# silently rerun stage Y under a stage-PAIR directory name -- the exact failure
# this line exists to prevent, and one no shape error would catch.
assert float(c.get("pair_weight", 0.0)) == want_pw, (
    "pair_weight did not round-trip -- this cell is stage Y again, not stage "
    "PAIR", c.get("pair_weight"), want_pw)
assert float(c.get("pair_margin", -1.0)) == want_pm, (
    "pair_margin did not round-trip", c.get("pair_margin"), want_pm)
assert c["max_len_margin"] == 2 * c["n_nodes"] + 2, c["max_len_margin"]
assert c["max_len"] == 512, (
    "max_len is %s, not stage S's 512 -- this cell has a different positional "
    "table and its count will not match the control" % c["max_len"])
d, L = c["d_model"], c["num_layers"]
pred = (630 * d + 58) + (4 * d * d + 8 * d) + d   # k=1, plus stage Y's s
got = c.get("n_trainable_params", -1)
print("arch_mode=%s lambda=%g channel=%s pair_weight=%g margin=%g applications=%d "
      "nhead=%d d_model=%d ff=%d seed=%d  trainable %s params"
      % (c["arch_mode"], float(c["determinacy_weight"]), c["determinacy_channel"],
         float(c["pair_weight"]), float(c["pair_margin"]), L, c["nhead"], d,
         c["dim_feedforward"], int(c["seed"]), format(got, ",")))
assert got == pred == 614138, (
    "this is not the registered cell: stage PAIR costs NOTHING on top of stage "
    "Y's 614,138", got, pred)
print("614,138 trainable = stage Y's exactly. The object is still "
      "h_{t+1} = F(h_t), iterated 16 times.")
PY

echo
echo "=== did the margin actually train? (the history's own columns) ==="
python3 - "${CKPT}/history.json" "$EPOCHS" <<'PY'
import json, sys
sys.path.insert(0, ".")
import scripts.stagepair_score as sp
h = json.load(open(sys.argv[1]))
want_ep = int(sys.argv[2])
first, last = h[0], h[-1]
# The margin starts at EXACTLY the margin, because det_scale = 0 makes g
# identically zero on both prompts (stagepair_smoke check 3). A first epoch that
# does not start near 1.0 means the term is not the registered one.
print("pair_hinge  %.4f -> %.4f    pair_gap %+.4f -> %+.4f    active %.3f -> %.3f"
      % (first["train_pair_hinge"], last["train_pair_hinge"],
         first["train_pair_gap"], last["train_pair_gap"],
         first["train_pair_active"], last["train_pair_active"]))
print("pair applied to %.4f of rows (the scan's yield, priced at 0.8034)"
      % last["train_pair_frac"])
assert first["train_pair_hinge"] > 0.5, (
    "epoch 0's hinge is %.4f; at det_scale = 0 the average over the epoch must "
    "start near the margin %.3f" % (first["train_pair_hinge"], sp.MARGIN))
print("train_nll %.4f -> %.4f   det_bce %.4f -> %.4f   det AUROC %.4f -> %.4f"
      % (first["train_nll"], last["train_nll"],
         first["train_determinacy_bce"], last["train_determinacy_bce"],
         first["determinacy_auroc"], last["determinacy_auroc"]))
acc = last["accuracy"]
print("final val TOKEN ACCURACY %.4f against stage Y's control 0.4830 and the "
      "free line's cell 0.4830/0.4830/0.4833" % acc)
first_ep = next((r["epoch"] for r in h if r["train_nll"] < 1.64), None)
print("plateau %s" % ("escaped at epoch %d" % first_ep if first_ep is not None
                      else "NOT ESCAPED (min %.4f)" % min(r["train_nll"] for r in h)))
# 1.561 is the basin line every earlier stage is quoted against. A literal here
# and not sp.something: the registration committed at 9f4edad does not define it
# and is not being edited after the fact to make a launcher tidier.
low = next((r["epoch"] for r in h if r["train_nll"] < 1.561), None)
print("basin line 1.561 %s" % ("crossed at epoch %d" % low
                               if low is not None else "not reached"))
print("converged (final train_nll <= %.3f + %.3f): %s"
      % (sp.FLOOR, sp.STUCK_TOL, last["train_nll"] <= sp.FLOOR + sp.STUCK_TOL))
assert len(h) == want_ep, ("the run did not reach the registered %d epochs; a "
                           "truncated cell does not compare with its matched "
                           "rung" % want_ep, len(h))
PY

echo
echo "=== rho_50: the transition curve for this cell, the SAME 14-point grid ==="
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== done ==="
date +'%F %T'
echo "P1-P4 need the matched-pair probe and finding 68's guard probe, chained"
echo "with --dependency=afterok on this array so nothing is polled."
