#!/bin/bash
#SBATCH --job-name=sud2N48ChainR
#SBATCH --partition=sganguli
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagerChain_%j.out
#
# STAGE R -- does the ATTENTION-ONLY cell CHAIN, or only apply the rule once?
#
# The registration is the docstring of scripts/stager_score.py and it is
# committed by STEP 1 below, before any arm runs. Read it, not this header, for
# the hypotheses, the falsifiers and the reading order.
#
# WHAT THIS ADDS TO STAGE Q. Q edited a HINT and measured round-0 forcing: flip
# 1.000/0.997/1.000 vs chance 0.250, clean 1.000 everywhere. Q therefore settled
# that the elimination RULE is applied over GIVEN facts. sudoku_chain edits a
# previously EMITTED DEDUCTION instead, which is the difference between applying
# an inference rule once and running the algorithm -- and the 0.980 chaining
# number in findings 27/28 was, like the rest of 27/28, only ever measured on a
# FULL-architecture N=24 checkpoint.
#
# WHAT "SAME INSTANCES" MEANS, EXACTLY, SO R4 IS NOT OVERCLAIMED. --seed-start
# is passed explicitly and equals the value stage Q used, so both stages draw
# the same graph/colouring/hint ensemble and Q's 1200 instances are a PREFIX of
# R's 2500. The CASES are necessarily disjoint -- round-0 and round>=1
# deductions cannot be the same node -- so R4 compares flip rates over the same
# instance distribution, not over the same nodes. That is the comparison the
# provenance question needs, and it is the only cross-stage one licensed.
#
# WHY 2500 INSTANCES where Q used 1200: a chained case needs a round>=1 node
# whose forcing set covers q-1 colours with the binding colour held by exactly
# one neighbour that was itself DEDUCED. That is rarer than a round-0 case, and
# sudoku_chain exits non-zero rather than scoring a short set, so the margin is
# deliberate. If an arm still finds fewer than 350 it scores what it found and
# prints n; the scorer reads n_cases from the summary.
#
# WHY NO GPU. Two forward passes per counterfactual, no patching and no
# ablation. Stage O's OA arm holds GPUs with stage P's array running on them,
# and putting the gate behind the runs it interprets would be backwards.
#
# ARM ORDER: the three attention-only seeds first, the control last. R0 needs
# all three seeds; the control is context and is shape-confounded (R2/Q2). If
# the limit truncates anything it must be the control, never the gate.
#
# NO VENV. $SCRATCH/spin_gnn_venv was purged; sudoku_chain imports only numpy,
# torch and its own modules. Never pipe `module load`, never set PYTHONPATH=.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

SW="${SCRATCH}/spin_gnn"
OUT="${SW}/sudoku_n48_chain_R_2026-11-10"
MSG=/scratch/users/hshunt/commit_msg_1110_stager.txt
SEED_START=600000000          # identical to stage Q's default; see the header

J2_S0="${SW}/sudoku_n48_minarch_J_2026-11-05/t01_d0192_ff01024_h04_no_mlp_L16_s0/d0192_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff"
J2_S1="${SW}/sudoku_n48_minarch_J_2026-11-05/t03_d0192_ff01024_h04_no_mlp_L16_s1/d0192_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_s01"
J2_S2="${SW}/sudoku_n48_minarch_J_2026-11-05/t05_d0192_ff01024_h04_no_mlp_L16_s2/d0192_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_s02"
CTRL="${SW}/sudoku_n48_minarch_2026-11-04/t00_L12_h04_full_s0/d0256_h04_L12_ff01024_lr1e-03_pre_shuf_fall"

mkdir -p "$OUT"
echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
echo "commit $(git rev-parse --short HEAD)"

echo
echo "=== STEP 1/3  commit the registration BEFORE any arm runs ================"
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged, nothing committed,"
    echo "and the lock is left alone -- it belongs to whichever process holds it."
    ls -l .git/index.lock
    exit 1
fi
git add scripts/stager_score.py scripts/slurm/sudoku2_n48_chain_R_2026-11-10.sbatch \
    notebooks/make_stagem_deck.py STATUS.md
echo "add rc=$?"
git diff --cached --name-only
if git diff --cached --quiet; then
    echo "nothing staged -- already committed"
else
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -F "$MSG" -q
    echo "commit rc=$?"
fi
module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$? (non-fatal: the commit is the record)"

echo
echo "=== STEP 2/3  the four checkpoints are the cells the registration names ==="
echo "    (the SAME four as stage Q -- that is what makes R4 a within-arm delta)"
python3 -u - "$J2_S0" "$J2_S1" "$J2_S2" "$CTRL" << 'PY'
import json, os, sys
# Guard: an arm that is silently a different cell makes every verdict void.
# arch_mode is absent in configs written before the field existed; those are
# 'full' by construction, so a missing value is read as 'full' and NOT as a
# mismatch -- but a present-and-wrong value fails here.
want = [("no_mlp", 16, 192, 0), ("no_mlp", 16, 192, 1),
        ("no_mlp", 16, 192, 2), ("full", 12, 256, 0)]
bad = 0
for p, w in zip(sys.argv[1:], want):
    if not os.path.exists(os.path.join(p, "model.pt")):
        print("MISSING model.pt: %s" % p); bad += 1; continue
    c = json.load(open(os.path.join(p, "config.json")))
    got = (c.get("arch_mode") or "full", c["num_layers"], c["d_model"], c["seed"])
    ok = got == w and c["n_nodes"] == 48 and c["shuffle_adj"] and c["fall"]
    print("  %-7s L=%-2d d=%-3d s%d  n_nodes=%d shuffle_adj=%s fall=%s  %s"
          % (got[0], got[1], got[2], got[3], c["n_nodes"], c["shuffle_adj"], c["fall"],
             "ok" if ok else "MISMATCH, registered %r" % (w,)))
    bad += 0 if ok else 1
if bad:
    print("FAILING CLOSED: %d arm(s) are not the cells the registration names." % bad)
    sys.exit(1)
print("all four arms match the registration; every arm is leak-free and FALL")
PY
if [ "$?" -ne 0 ]; then
    echo "no arm runs against an unverified checkpoint."
    exit 1
fi

echo
echo "=== STEP 3/3  the four arms, gate first, control last ==================="
run_arm () {
    local tag="$1" ckpt="$2"
    echo
    echo "--- ARM $tag ---"
    python3 -u -m spin_gnn.sudoku_chain \
        --checkpoint "$ckpt" \
        --n-instances 2500 \
        --n-cases 350 \
        --seed-start "$SEED_START" \
        --device cpu \
        --out-dir "${OUT}/${tag}"
    echo "arm $tag rc=$?  $(date '+%T')"
}

run_arm R_noff_s0 "$J2_S0"
run_arm R_noff_s1 "$J2_S1"
run_arm R_noff_s2 "$J2_S2"
run_arm R_full_ctrl "$CTRL"

echo
echo "=== the registered score ================================================"
python3 -u -m scripts.stager_score
echo "scorer rc=$?"
echo
echo "READ R3 FIRST, THEN R1, AND ONLY THEN R0. A low flip rate is ambiguous"
echo "between 'does not chain' and 'detected the edit' -- reverts_to_original"
echo "is the column that tells them apart."
echo "end $(date '+%F %T')"
