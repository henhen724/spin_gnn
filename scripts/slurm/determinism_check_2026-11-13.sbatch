#!/bin/bash
# =============================================================================
# DETERMINISM CHECK -- is the measuring instrument repeatable at all?
#
# This is not a stage and not an arm. It moves no scientific knob and it cannot
# change any published number. It asks whether a RERUN of one cell reproduces
# that cell, which every comparison in stages S through W silently assumed.
#
# WHY NOW. Finding 66b, from stage W's deck: the 100-epoch cells and their
# 50-epoch "controls" are independent RUNS of one configuration -- identical
# config but for the epoch count, identical data (the stored baseline accuracy
# agrees to 16 digits) -- and at the control's last epoch they already differ by
#
#     seed 0   -0.0113 train NLL   +0.0018 val accuracy
#     seed 1   +0.0078             -0.0027
#     seed 2   +0.0602             -0.0246
#
# against the 0.0011 of token accuracy that stage S's remaining 0.017 of rho_50
# is worth. The noise between two runs of ONE configuration is 2x to 22x the
# effect every stage since S has been trying to read. That is why S, T, U, V and
# W all came back "within noise": with 3 pairs, sd of the paired accuracy
# difference is about 0.0144 (0.0032 dropping seed 2), so resolving 0.0011 needs
# of order 66 to 5,100 paired replicates. Neither number is affordable. The way
# out is not more replicates, it is a smaller noise floor.
#
# WHAT IS ALREADY PINNED, so that what is left is identified and not guessed:
# train_sudoku_model seeds the INSTANCES (build_instances(cfg, n, cfg.seed)),
# the WEIGHT INIT (torch.manual_seed(cfg.seed)) and the BATCH ORDER
# (np.random.default_rng(cfg.seed) passed into iterate_token_batches). The
# learning rate is constant 1.00e-03 at every epoch of both arms with
# warmup_epochs=0, so nothing in the schedule depends on the epoch count. What
# is NOT pinned is the order a GPU kernel reduces a sum in, and floating-point
# addition is not associative, so two runs take different rounding from step 1.
# The measured divergence is largest at epoch 0 (2.5167 vs 2.3656, a gap of
# 0.151), falls to 0.0168 by epoch 1 and 0.0003 by epoch 3 -- a steep descent
# amplifying rounding and then contracting -- and reopens late, where basins
# separate. Seed 2's control stepped at epoch 22 and its twin never did, which
# is most of why W-a read 1/3.
#
# HYPOTHESIS, STATED BEFORE THE JOB RUNS
#
#   H1  With --deterministic, two runs of this cell at seed 0 produce train NLL
#       that agrees to every printed digit at every epoch (|difference| = 0.0000
#       at epochs 0, 1 and 2, and bit-identical in history.json).
#   H2  Without it, two runs at seed 0 differ at epoch 0 on the order of 0.1,
#       the way the existing 50/100-epoch pair does (2.5167 vs 2.3656).
#   H3  Determinism costs wall clock. Unknown sign of the cost in advance;
#       this job measures it as min/epoch, which is what decides whether a
#       future stage can afford to run this way.
#
# H2 FAILING IS THE INTERESTING OUTCOME and must be reported as such: if two
# unpinned runs also agree to all digits, then kernel nondeterminism is NOT what
# separated the 50- and 100-epoch cells, finding 66b has another cause, and
# pinning determinism fixes nothing. The free arm is here to be able to say that.
#
# DESIGN. Four cells, run sequentially in one job (scheduler etiquette: each is
# ~35-65 min, well past the 10-minute floor, and they share one dataset build
# each). Seed 0 throughout -- replicates of ONE seed, which is the question;
# three seeds would answer a different one. The cell is S1 exactly, the control
# every stage since S is read against: tied_no_mlp, T=16 applications of one
# shared block, d320 ff1024 h4, 613,818 trainable. Only the epoch count is cut,
# to 3, because H1 and H2 are both decided at epoch 0 and the third epoch only
# shows whether agreement survives accumulation. n_val_instances 32 and
# n_per_bin 2 cut the end-of-run evals, which answer nothing here; the TRAIN NLL
# is the quantity under test and it is computed on the full 240,000-instance
# training set, unchanged from S and W, so epoch 0 here is directly comparable
# to those runs' epoch 0.
#
# The deterministic arm runs FIRST, so a preemption or a time-out still leaves
# H1 answered. CUBLAS_WORKSPACE_CONFIG is set per-command rather than for the
# whole job on purpose: cuBLAS reads it once at init and it can make GEMMs
# repeatable on its own, so exporting it job-wide would quietly pin the free arm
# too and H2 would be unfalsifiable.
#
# NOTHING PUBLISHED CHANGES. --deterministic defaults to off, so every number in
# stages S-W is reproduced by the code as it stands; a cell run with the flag
# gets a _det tag segment and its own directory, because pinned and unpinned
# kernels end at different weights from the same init.
# =============================================================================
#SBATCH --job-name=sud2N48Det
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out
#
# 5 hours from the measured cost of this cell: the 50- and 100-epoch arms took
# 474 and 942 min, so an epoch is 9.4 min and the dataset build plus the evals
# together are about 6 min. Three epochs is ~34 min unpinned; the pinned arm is
# allowed to be twice as slow at ~65 min; four cells is ~3.3 h with margin for a
# worse determinism penalty.

set -uo pipefail
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
date +'job start %F %T'

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

OUT="${SCRATCH}/spin_gnn/runs/2026-11-13_determinism"
mkdir -p "$OUT"

DMODEL=320; NHEAD=4; FF=1024; LAYERS=16; ARCH=tied_no_mlp; KBLOCKS=1
SEED=0; EPOCHS=3
TAG="d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied"

echo
echo "=== 0. preflight ================================================="
# Three things, all cheap, all of which would otherwise be discovered after a
# dataset build or four hours in:
#   * the flag is OFF by default, which is what keeps stages S-W reproducible;
#   * the cell really is S1 (613,818 trainable, one block applied 16 times);
#   * use_deterministic_algorithms(True) does not RAISE on the ops this model
#     actually runs. That is the live risk: PyTorch refuses rather than silently
#     approximating when an op has no deterministic kernel, and the error names
#     the op. One forward/backward/step on a random batch exercises the
#     embedding, the tied encoder stack, the masked NLL and Adam -- which is the
#     whole set -- in seconds.
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 - <<'PY'
import os, sys, torch
sys.path.insert(0, "/scratch/users/hshunt/spin_gnn")
from spin_gnn.config import SudokuReasoningTrainConfig
from spin_gnn.sat_transformer2 import SATReasoningTransformer, sat2_nll_loss

assert SudokuReasoningTrainConfig().deterministic is False, (
    "the flag must default OFF -- every run up to stage W was unpinned and "
    "their numbers must stay reproducible")
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print("deterministic algorithms accepted at import")

d, L, nhead, k = 320, 16, 4, 1
m = SATReasoningTransformer(vocab_size=58, d_model=d, num_layers=L, nhead=nhead,
                            max_len=512, dim_feedforward=1024, norm_first=True,
                            arch_mode="tied_no_mlp", n_blocks=k).cuda()
want = (630 * d + 58) + (4 * d * d + 8 * d) * k
got = m.trainable_parameter_count()
assert got == want == 613818, ("this is meant to be S1 exactly, unchanged", got, want)
assert len(m.encoder.layers) == 1 and m.n_iters == L, (len(m.encoder.layers), m.n_iters)
print("%s trainable, one block applied %d times -- S1" % (format(got, ","), L))

# A real step, so an op without a deterministic kernel raises HERE.
torch.manual_seed(0)
x = torch.randint(0, 58, (4, 64), device="cuda")
y = torch.randint(0, 58, (4, 64), device="cuda")
mask = torch.ones(4, 64, dtype=torch.bool, device="cuda")
opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
loss = sat2_nll_loss(m(x, valid_mask=mask), y, mask)
opt.zero_grad(); loss.backward(); opt.step()
print("one deterministic forward/backward/step: OK (loss %.4f)" % loss.item())
PY
if [ $? -ne 0 ]; then
    echo "PREFLIGHT FAILED -- no cell submitted. If the message above names an"
    echo "op with no deterministic implementation, that op is the finding."
    exit 1
fi

run_cell () {                      # $1 = cell name, $2 = "det" or "free"
    local name="$1" mode="$2"
    local cell="${OUT}/${name}"
    mkdir -p "$cell"
    echo
    echo "=== cell ${name} (${mode}) ======================================"
    date +'cell start %F %T'
    local t0 t1
    t0=$(date +%s)
    if [ "$mode" = "det" ]; then
        CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -m spin_gnn.sudoku_hparam_sweep \
            --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
            --d-model "$DMODEL" --nhead "$NHEAD" --dim-feedforward "$FF" \
            --num-layers "$LAYERS" --arch-mode "$ARCH" --n-blocks "$KBLOCKS" \
            --seed "$SEED" --warmup-epochs 0 --deterministic \
            --n-train-instances 240000 --epochs "$EPOCHS" \
            --n-val-instances 32 --batch-size 32 --n-per-bin 2 \
            --out-dir "$cell"
    else
        python3 -m spin_gnn.sudoku_hparam_sweep \
            --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
            --d-model "$DMODEL" --nhead "$NHEAD" --dim-feedforward "$FF" \
            --num-layers "$LAYERS" --arch-mode "$ARCH" --n-blocks "$KBLOCKS" \
            --seed "$SEED" --warmup-epochs 0 \
            --n-train-instances 240000 --epochs "$EPOCHS" \
            --n-val-instances 32 --batch-size 32 --n-per-bin 2 \
            --out-dir "$cell"
    fi
    local rc=$?
    t1=$(date +%s)
    date +'cell end   %F %T'
    echo "cell ${name} rc=${rc} wall=$(( (t1 - t0) / 60 ))m$(( (t1 - t0) % 60 ))s"
    return $rc
}

# The pinned arm first: it answers H1, and H1 is what a later stage would act on.
# Only the FIRST cell aborts the job on failure -- if the pinned path is broken
# the other three would break identically and there is nothing to learn from
# spending three more GPU-hours on it. A later cell failing still leaves a pair
# that answers one of the two hypotheses, so those run on and the report says
# which half landed.
run_cell det_a  det || { echo "det_a failed -- stopping, the remaining cells fail the same way"; exit 1; }
run_cell det_b  det
run_cell free_a free
run_cell free_b free

echo
echo "=== report ======================================================="
python3 - <<'PY'
import json, os
OUT = os.path.join(os.environ["SCRATCH"], "spin_gnn/runs/2026-11-13_determinism")
TAG = "d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied"

def hist(name, det):
    p = os.path.join(OUT, name, TAG + ("_det" if det else ""), "history.json")
    if not os.path.exists(p):
        print("MISSING %s" % p)
        return None
    with open(p) as f:
        return json.load(f)

pairs = [("deterministic", hist("det_a", True), hist("det_b", True)),
         ("unpinned", hist("free_a", False), hist("free_b", False))]
for label, a, b in pairs:
    print()
    if a is None or b is None:
        print("%s: incomplete, no verdict" % label)
        continue
    print("%s pair, seed 0, replicate a vs replicate b" % label)
    print("  epoch |      a train_nll |      b train_nll |   b - a    | a acc    b acc")
    identical = True
    for ra, rb in zip(a, b):
        d = rb["train_nll"] - ra["train_nll"]
        identical &= (rb["train_nll"] == ra["train_nll"]
                      and rb["accuracy"] == ra["accuracy"])
        print("  %5d | %16.12f | %16.12f | %+10.2e | %.4f  %.4f"
              % (ra["epoch"], ra["train_nll"], rb["train_nll"], d,
                 ra["accuracy"], rb["accuracy"]))
    print("  bit-identical train_nll AND val accuracy at every epoch: %s"
          % ("YES" if identical else "NO"))

# Epoch 0 against the two runs that raised the question. Same 240,000-instance
# training set and the same init, so these are comparable draws of one number.
print()
print("epoch-0 train NLL, this job against stage W's seed-0 pair")
print("  stage W 100-epoch cell   2.5167")
print("  stage S 50-epoch control 2.3656   (the 0.151 that started this)")
for label, name, det in (("det_a", "det_a", True), ("det_b", "det_b", True),
                         ("free_a", "free_a", False), ("free_b", "free_b", False)):
    h = hist(name, det)
    if h:
        print("  %-24s %.4f" % (label, h[0]["train_nll"]))
PY

date +'job end %F %T'
