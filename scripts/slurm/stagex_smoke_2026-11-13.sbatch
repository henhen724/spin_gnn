#!/bin/bash
# =============================================================================
# STAGE X SMOKE -- the three things a CPU smoke cannot test, before 24 GPU-h.
#
# The CPU end-to-end already ran both branches (weight 0 reproduces the old
# history keys exactly; weight 1 printed det_bce 0.6467 -> 0.6363 and val
# AUROC 0.5666 -> 0.6079 over two epochs on a toy N=24 cell). What it did NOT
# exercise:
#
#   1. GATHER BACKWARD UNDER PINNED KERNELS. determinacy_loss reads the node
#      logits with logits.gather(1, ...), whose backward is a scatter_add, and
#      torch.use_deterministic_algorithms(True) RAISES on several CUDA
#      scatter/index backward kernels rather than falling back. Stage X runs
#      --deterministic, so if this throws it throws at the first step of a
#      20-hour cell. Checked here in isolation first, so the failure is 30
#      seconds old instead of an hour.
#   2. The int8 target crossing to CUDA and being cast against a half/float
#      logit tensor.
#   3. Throughput and memory with the aux collate at N=48, where a batch
#      carries an extra (32, ~13, 48) int8 tensor.
#
# This is a THROWAWAY cell: 2000 instances, 2 epochs, n_per_bin 10, written to
# its own _smoke directory so it can never pool with a stage-X cell or with
# stage S. Nothing it produces is scored and nothing is read from it except
# "did it run".
# =============================================================================

#SBATCH --job-name=stagexSmoke
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -euo pipefail

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
# Torch reads this ONCE at cuBLAS init, so it has to be in the environment
# before python starts -- the trainer asserts on it rather than running
# non-deterministically while claiming otherwise.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${SLURM_SUBMIT_DIR}"

echo "=== 1. the isolated risk: gather backward with pinned kernels on CUDA ==="
python3 - <<'PY'
import torch
import torch.nn.functional as F
from spin_gnn.sudoku_determinacy import determinacy_loss, MASKED

assert torch.cuda.is_available(), "no GPU visible to torch"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
print("cuda", torch.cuda.get_device_name(0), "torch", torch.__version__)

# The shape stage X actually runs: batch 32, sequence 391, vocab 58, 48 nodes,
# 13 scored positions, node tokens a contiguous block at node_offset.
B, T, V, N, P, OFF = 32, 391, 58, 48, 13, 10
logits = torch.randn(B, T, V, device="cuda", requires_grad=True)
pos = torch.randint(0, T, (B, P), device="cuda")
tgt = torch.randint(-1, 2, (B, P, N), dtype=torch.int8, device="cuda")
loss = determinacy_loss(logits, pos, tgt, OFF, N)
assert loss is not None
loss.backward()
g = logits.grad
assert torch.isfinite(g).all(), "non-finite gradient out of the determinacy term"
# The gradient must be ZERO everywhere outside the node block, and zero at
# every position the targets do not name. If gather/scatter mis-indexed, this
# is where it shows.
touched = torch.zeros(B, T, dtype=torch.bool, device="cuda")
touched.scatter_(1, pos, True)
assert g[~touched].abs().max() == 0, "the term wrote gradient at an unscored position"
blk = torch.zeros(V, dtype=torch.bool, device="cuda")
blk[OFF:OFF + N] = True
assert g[..., ~blk].abs().max() == 0, "the term wrote gradient outside the node block"
assert g[touched][:, OFF:OFF + N].abs().sum() > 0, "the term wrote no gradient at all"
print(f"gather backward OK under use_deterministic_algorithms(True); "
      f"loss {loss.item():.4f}, gradient confined to {int(touched.sum())} "
      f"positions x {N} node logits")

# An all-masked batch must return None rather than a nan, since the padded
# rows of a short instance are exactly that.
none_tgt = torch.full((B, P, N), MASKED, dtype=torch.int8, device="cuda")
assert determinacy_loss(logits, pos, none_tgt, OFF, N) is None
print("an all-masked batch returns None, not nan")
PY

echo
echo "=== 2. end to end at N=48: 2000 instances, 2 epochs, pinned, dw=1 ==="
OUT="${SCRATCH}/spin_gnn/stagex_smoke_2026-11-13"
mkdir -p "$OUT"
date +'smoke start %F %T'

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model 320 --nhead 4 --dim-feedforward 1024 \
    --num-layers 16 --arch-mode tied_no_mlp --n-blocks 1 \
    --seed 0 --warmup-epochs 0 --deterministic \
    --determinacy-weight 1.0 \
    --n-train-instances 2000 --epochs 2 \
    --n-val-instances 300 --batch-size 32 --n-per-bin 10 \
    --out-dir "$OUT"

date +'smoke end   %F %T'

echo
echo "=== 3. read back what the throwaway checkpoint says it is ==="
CKPT="${OUT}/d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1"
[ -f "${CKPT}/model.pt" ] || { echo "the tag is not what the launcher predicts:"; ls -1 "$OUT"; exit 1; }
python3 - "${CKPT}/config.json" "${CKPT}/history.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
h = json.load(open(sys.argv[2]))
assert c["shuffle_adj"] is True and c["fall"] is True
assert c["arch_mode"] == "tied_no_mlp" and c["n_nodes"] == 48
assert c["deterministic"] is True, "the pinned-kernel flag did not round-trip"
# The headline of the whole stage: the objective moved and the OBJECT did not.
assert float(c["determinacy_weight"]) == 1.0, (
    "determinacy_weight did not round-trip; a stage-X cell would be "
    "indistinguishable from its stage-S control in the checkpoint",
    c.get("determinacy_weight"))
got = c.get("n_trainable_params", -1)
# NOT 613,818 here, and that is not a defect. The positional embedding is
# trainable, so max_len is a term in the count; 2000 instances give a shorter
# longest sequence than 240k and max_len resolves to 511 instead of 512,
# which is exactly one position = d_model = 320 parameters fewer. The
# invariant the smoke can check is the one that matters -- the term itself
# costs nothing -- so the count is predicted FROM max_len rather than
# hard-coded, and the real array asserts max_len == 512 and 613,818.
d = c["d_model"]
pred = (c["max_len"] + 2 * 58 + 2) * d + 58 + (4 * d * d + 8 * d)
assert got == pred, ("the determinacy term cost parameters -- it must cost "
                     "ZERO, the loss attaches to the EXISTING readout",
                     got, pred, c["max_len"])
print("%s trainable at max_len=%d; the formula with ZERO terms for the "
      "objective predicts %s -- the term is free, as designed"
      % (format(got, ","), c["max_len"], format(pred, ",")))
for r in h:
    print("epoch %d  train_nll %.4f  val_nll %.4f  acc %.4f  det_bce %.4f  auroc %.4f"
          % (r["epoch"], r["train_nll"], r["nll"], r["accuracy"],
             r["train_determinacy_bce"], r["determinacy_auroc"]))
assert all("determinacy_auroc" in r for r in h), "the AUROC was never recorded"
print("tag, checkpoint and history all carry the term. The numbers above are "
      "from 2000 instances and mean nothing except that it ran.")
PY

echo
echo "=== done ==="
date +'%F %T'
