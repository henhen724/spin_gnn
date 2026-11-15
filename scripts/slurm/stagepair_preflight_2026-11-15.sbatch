#!/bin/bash
#
# PREFLIGHT for stage PAIR's 3-seed array, which the array is chained behind
# with --dependency=afterok. NOTHING HERE IS A RESULT: no bar is read, no
# verdict is made, and the 2-epoch model it trains is thrown away.
#
# WHY A SEPARATE JOB AND NOT JUST THE ARRAY'S OWN PREAMBLE. The array requests
# 36 hours on a GPU, three times. Its preamble does run the smoke test, and
# `set -e` would stop a cell before it trains -- but a bug in the new code would
# only surface after three GPU allocations, and Rule 10 says not to resubmit
# a failed job without finding the cause first. 15 minutes on one GPU here buys
# the whole array a gate.
#
# WHAT THE SMOKE TEST CANNOT SEE, which is the other half of this job's point.
# stagepair_smoke.py asserts the margin's algebra and its SCALE at init, on a
# batch it builds itself. It never runs the trainer. So it cannot see a wrong
# `batch[5]`, a collate that is not selected, a history key that is never
# written, or a config.json that does not round-trip pair_weight -- and the last
# of those is the failure that would silently rerun stage Y under a stage-PAIR
# name, since the two state dicts are identical. A 2-epoch run on 320 instances
# exercises exactly that path, end to end, including save_checkpoint.
#
#SBATCH --job-name=stagepairPre
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -euo pipefail
cd /home/users/hshunt/spin_gnn

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo
echo "=== 1. compile ==================================================="
python3 -m py_compile spin_gnn/sudoku_pair.py spin_gnn/sat_dataset2.py \
    spin_gnn/train_sudoku_transformer.py spin_gnn/sudoku_hparam_sweep.py \
    spin_gnn/config.py scripts/stagepair_smoke.py

echo
echo "=== 2. the seven assertions, three of them about magnitude ======="
python3 scripts/stagepair_smoke.py

echo
echo "=== 3. the determinacy targets still agree with DD ==============="
python3 - <<'PY'
import importlib
m = importlib.import_module("tests.test_sudoku_determinacy")
names = sorted(n for n in dir(m) if n.startswith("test_"))
for n in names:
    getattr(m, n)()
print("%d determinacy tests pass" % len(names))
PY

TMP="${SCRATCH}/spin_gnn/runs/2026-11-15_stagepair_preflight"
mkdir -p "$TMP"

echo
echo "=== 4. two epochs end to end, 320 instances, thrown away ========="
# Everything is the real path except the sizes: the same CLI, the same cell, the
# same flags, the same save_checkpoint. --n-per-bin 2 keeps the generation eval
# to seconds; it is not a measurement and is not read.
#
# --n-val-instances must be at least --batch-size. iterate_token_batches drops
# the last PARTIAL batch, so 20 validation instances at batch 32 yields zero
# batches and evaluate() divides by zero -- which is what 52422951 did, in the
# trainer's own code and not in anything stage PAIR added. The real array runs
# 300 and never meets this.
python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model 320 --nhead 4 --dim-feedforward 1024 \
    --num-layers 16 --arch-mode tied_no_mlp --n-blocks 1 \
    --seed 0 --warmup-epochs 0 --deterministic \
    --determinacy-weight 1.0 --determinacy-channel \
    --pair-weight 1.0 --pair-margin 1.0 \
    --n-train-instances 320 --epochs 2 \
    --n-val-instances 64 --batch-size 32 --n-per-bin 2 \
    --out-dir "$TMP"

echo
echo "=== 5. what the artifact says it is ============================="
python3 - "$TMP" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
cks = sorted(glob.glob(os.path.join(root, "*", "config.json")))
assert len(cks) == 1, ("expected one cell, found %d" % len(cks), cks)
cell = os.path.dirname(cks[0])
print("cell dir:", os.path.basename(cell))
assert "_pw1" in os.path.basename(cell), (
    "the tag does not carry the pair weight, so a stage-PAIR cell and its "
    "stage-Y control collide on disk", os.path.basename(cell))
c = json.load(open(cks[0]))
assert float(c["pair_weight"]) == 1.0, ("pair_weight did not round-trip", c.get("pair_weight"))
assert float(c["pair_margin"]) == 1.0, ("pair_margin did not round-trip", c.get("pair_margin"))
assert c["determinacy_channel"] is True
# NOT 614,138 here, and the difference is the preflight's own, not the term's:
# max_len is sized to the longest sequence in the TRAINING set plus a margin, the
# positional table is trainable, and 320 instances do not contain the 408-token
# trace that 240,000 do. So the count is asserted against the formula at this
# cell's own max_len -- which still pins the channel's +320 and the absence of
# any other new parameter, the only thing this check is for. The array asserts
# the literal 614,138, where max_len is 512 and must be.
d, ml = c["d_model"], c["max_len"]
want = (630 * d + 58) + (4 * d * d + 8 * d) + d - (512 - ml) * d
assert c["n_trainable_params"] == want, (c["n_trainable_params"], want, ml)
print("trainable %s at max_len=%d (= 614,138 - %d unreached positions x %d)"
      % (format(c["n_trainable_params"], ","), ml, 512 - ml, d))
h = json.load(open(os.path.join(cell, "history.json")))
need = ("train_pair_hinge", "train_pair_gap", "train_pair_active", "train_pair_frac")
missing = [k for k in need if k not in h[0]]
assert not missing, ("the history does not record the term: %s" % missing)
print("epoch 0: hinge %.4f  gap %+.4f  active %.3f  applied to %.3f of rows"
      % (h[0]["train_pair_hinge"], h[0]["train_pair_gap"],
         h[0]["train_pair_active"], h[0]["train_pair_frac"]))
print("epoch 1: hinge %.4f  gap %+.4f  active %.3f"
      % (h[1]["train_pair_hinge"], h[1]["train_pair_gap"], h[1]["train_pair_active"]))
# At det_scale = 0 the gap is identically zero, so epoch 0's AVERAGE hinge is
# the margin minus however much the first steps already bought. It cannot be
# above the margin and it must not be near zero.
assert 0.3 < h[0]["train_pair_hinge"] <= 1.0 + 1e-6, (
    "epoch 0's hinge is %.4f, which is not where a margin of 1.0 starting from "
    "g == 0 can be" % h[0]["train_pair_hinge"])
assert 0.6 < h[0]["train_pair_frac"] < 0.95, (
    "the term reached %.4f of rows; the scan was priced at 0.8034"
    % h[0]["train_pair_frac"])
print("preflight OK -- the plumbing carries the term and the artifact records it")
PY

echo
echo "end $(date '+%F %T')"
