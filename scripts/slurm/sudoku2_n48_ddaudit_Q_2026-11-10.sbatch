#!/bin/bash
#SBATCH --job-name=sud2N48DDauditQ
#SBATCH --partition=sganguli
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stageqAudit_%j.out
#
# STAGE Q -- THE GATE: does the ATTENTION-ONLY cell actually do direct deduction?
#
# The registration is the docstring of scripts/stageq_score.py and it is
# committed by STEP 1 below, before any arm runs. Read it, not this header, for
# the hypotheses and falsifiers.
#
# WHY NO GPU. Behaviour only -- two forward passes per counterfactual, no
# patching, no ablation. That is ~2,800 forwards total at d192/L16 and
# d256/L12, which fits CPU comfortably, and all three sganguli GPUs are held by
# stage O's OA arm with stage P's array queued behind it. Asking for a GPU here
# would put the GATE behind the very runs it is meant to interpret.
#
# WHY --stages behavior. sudoku_circuits always runs the behaviour arm and gates
# the three localization stages on membership in {patching,heads,sublayer}; a
# stage name outside that set therefore runs behaviour and nothing else.
# Localization asks WHERE the mechanism is, which only matters once we know
# there IS one -- and it is where the GPU cost lives.
#
# ARM ORDER IS DELIBERATE: the three attention-only seeds first, the control
# last. Q0 (the gate) and Q4 (the seed story) need all three seeds; the control
# is context only and is explicitly shape-confounded (Q2). If the 4 h limit
# truncates anything it must be the control, never the gate. Each arm writes its
# own out-dir, so a partial run is complete for whichever arms finished and the
# scorer prints PENDING for the rest.
#
# NO VENV. $SCRATCH/spin_gnn_venv was purged; sudoku_circuits imports only
# numpy, torch and its own modules, so the module stack is sufficient. Never
# pipe `module load`, never set PYTHONPATH= inline.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

SW="${SCRATCH}/spin_gnn"
OUT="${SW}/sudoku_n48_ddaudit_Q_2026-11-10"
MSG=/scratch/users/hshunt/commit_msg_1110_stageq.txt

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
git add scripts/stageq_score.py scripts/slurm/sudoku2_n48_ddaudit_Q_2026-11-10.sbatch STATUS.md
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
    python3 -u -m spin_gnn.sudoku_circuits \
        --checkpoint "$ckpt" \
        --n-instances 1200 \
        --n-cases 350 \
        --device cpu \
        --stages behavior \
        --out-dir "${OUT}/${tag}"
    echo "arm $tag rc=$?  $(date '+%T')"
}

run_arm Q_noff_s0 "$J2_S0"
run_arm Q_noff_s1 "$J2_S1"
run_arm Q_noff_s2 "$J2_S2"
run_arm Q_full_ctrl "$CTRL"

echo
echo "=== the registered score ================================================"
python3 -u -m scripts.stageq_score
echo "scorer rc=$?"
echo
echo "end $(date '+%F %T')"
