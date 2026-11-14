#!/bin/bash
# =============================================================================
# STAGE X, X2 AND X3 -- the two measurements that need the trained cells.
#
# Chained with --dependency=afterok on the training array, so it fires when
# all three cells land and nothing has to be polled. If any cell fails this
# never runs, which is correct: a 2-of-3 guard bar cannot be read off 2 cells.
#
# X2, THE MECHANISM. Finding 68's guard probe, UNCHANGED, on the stage-X
# checkpoints. Same module (spin_gnn.sudoku_skip), same n_instances 800, same
# n_cases 350, same seed_start 600000000 -- so the case list is byte-identical
# to the one that measured S1 at 1.0000 / 1.0000 / 1.0000 and J2 at 0.0000 /
# 0.0000 / 0.8444. Nothing about the probe is re-tuned for the new cells;
# that is the entire point of reusing it.
#
#   BAR: conditional guard rate below 0.90 in at least 2 of 3 seeds. Lower is
#   better -- the rate is how often the model still names a node whose
#   determinacy was destroyed, CONDITIONAL on it naming that node on the clean
#   prompt. The control measured 1.0000 three times with spread 0.0000, so
#   0.90 is ten times its entire spread.
#
#   THE DENOMINATOR TRAP, which this project has walked into twice with two
#   different scripts: ddguard_score.ratio() computes both the conditional and
#   the unconditional rate. The conditional one is the published number and
#   the scorer labels it IN WORDS. Do not read a guard rate off a column
#   index.
#
# X3, DID THE TERM TAKE. determinacy_eval on all six cells -- three stage-X
# and the three stage-S controls -- in ONE invocation, so every number comes
# off the same fixed 600-instance held-out set in the same process. The
# stage-S side is finding 73 (AUROC 0.7610 / 0.7635 / 0.7774) and re-measuring
# it here costs seconds and removes any doubt that the two groups saw the same
# instances.
#
#   BAR: every stage-X seed above every stage-S seed. NOT "above 0.5" -- the
#   main loss already lifts the next node's logit and the next node is always
#   deducible, which is why the control sits at 0.77 and not at chance.
#
# X1 and X4 are already measured by the training array itself (the rho grid
# and the history), so they are not repeated here.
# =============================================================================

#SBATCH --job-name=stagexGuard
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -euo pipefail

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"

X="${SCRATCH}/spin_gnn/sudoku_n48_determinacy_X_2026-11-13"
S="${SCRATCH}/spin_gnn/sudoku_n48_tiedwidth_S_2026-11-10"
RUN="${SCRATCH}/spin_gnn/runs/2026-11-13_stagex_guard"
mkdir -p "$RUN"

# Finding 68's constants, copied so the case list is the SAME one. If any of
# these three changes the comparison against 1.0000 x3 is void.
SEED_START=600000000
NINST=800
NCASES=350

XC0="${X}/t00_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_s0/d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1"
XC1="${X}/t01_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_s1/d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_s01"
XC2="${X}/t02_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_s2/d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_s02"
SC0="${S}/t00_d0320_ff01024_h04_tied_no_mlp_L16_s0/d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied"
SC1="${S}/t01_d0320_ff01024_h04_tied_no_mlp_L16_s1/d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_s01"
SC2="${S}/t02_d0320_ff01024_h04_tied_no_mlp_L16_s2/d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_s02"

for c in "$XC0" "$XC1" "$XC2" "$SC0" "$SC1" "$SC2"; do
    [ -f "${c}/model.pt" ] || { echo "missing checkpoint: ${c}"; exit 1; }
done

echo "=== every stage-X cell is the registered one, checked before measuring ==="
python3 - "$XC0" "$XC1" "$XC2" <<'PY'
import json, sys
import scripts.stagex_score as sx
for p in sys.argv[1:]:
    c = json.load(open("%s/config.json" % p))
    dw = float(c.get("determinacy_weight", 0.0) or 0.0)
    assert dw == sx.LAMBDA, ("not the registered lambda", p, dw)
    assert c["deterministic"] is True and c["n_nodes"] == 48
    assert c["arch_mode"] == "tied_no_mlp" and c["num_layers"] == 16
    assert c["n_trainable_params"] == 613818, c["n_trainable_params"]
    assert len(json.load(open("%s/history.json" % p))) == 50
    print("  seed %s  lambda %g  613,818 trainable  50 epochs" % (c["seed"], dw))
PY

date +'start %F %T'

echo
echo "=== X2: finding 68's guard probe, unchanged, on the stage-X cells ==="
echo "    n_instances=${NINST} n_cases=${NCASES} seed_start=${SEED_START} -- "
echo "    identical to the run that measured S1 at 1.0000 x3"
i=0
for CKPT in "$XC0" "$XC1" "$XC2"; do
    echo
    echo "--- guard stagex s${i} ---"
    python3 -u -m spin_gnn.sudoku_skip --checkpoint "$CKPT" \
        --n-instances "$NINST" --n-cases "$NCASES" --seed-start "$SEED_START" \
        --out-dir "${RUN}/guard/stagex_s${i}"
    i=$((i + 1))
done

echo
echo "=== X3: all six cells on ONE fixed held-out set, one process ==="
python3 -u -m scripts.determinacy_eval \
    --checkpoint "$XC0" "$XC1" "$XC2" "$SC0" "$SC1" "$SC2" \
    --n-instances 600 \
    --out-csv "${RUN}/determinacy_both_arms.csv"

echo
echo "=== the guard rates, CONDITIONAL, labelled in words ==="
python3 - "${RUN}/guard" <<'PY'
import json, os, sys
import scripts.stagex_score as sx
root = sys.argv[1]
rates = []
for name in sorted(os.listdir(root)):
    path = os.path.join(root, name, "summary.json")
    if not os.path.exists(path):
        print("  %-14s no summary.json -- read the directory by hand" % name)
        continue
    s = json.load(open(path))
    print("  %-14s %s" % (name, json.dumps(s, sort_keys=True)[:300]))
print()
print("CONDITIONAL on clean_emits_v is the published number. The control is")
print("S1 at %s, spread 0.0000. The bar is below %.2f in at least %d of 3."
      % ("/".join("%.4f" % v for v in sx.S1_GUARD), sx.GUARD_BAR, sx.GUARD_MIN_SEEDS))
print("Compute it with ddguard_score.ratio() and label it in words -- the")
print("denominator mistake has been made twice by two different scripts.")
PY

echo
echo "=== done. X1 and X4 come from the training array's own output. ==="
echo "score with: python3 scripts/stagex_score.py <summary.json>"
date +'end %F %T'
