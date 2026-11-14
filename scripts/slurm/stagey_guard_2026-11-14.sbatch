#!/bin/bash
# =============================================================================
# STAGE Y, Y2 AND Y3 -- the two measurements that need the trained cells, plus
# the registered verdict for the whole stage.
#
# Chained with --dependency=afterok on the training array, so it fires when all
# three cells land and nothing is polled. If any cell fails this never runs,
# which is correct: a 2-of-3 bar cannot be read off 2 cells.
#
# Y2, THE MECHANISM. Finding 68's guard probe, UNCHANGED, on the stage-Y
# checkpoints. Same module (spin_gnn.sudoku_skip), same n_instances 800, same
# n_cases 350, same seed_start 600000000 -- byte-identical to the case list
# that measured S1 at 1.0000/1.0000/1.0000, J2 at 0.0000/0.0000/0.8444 and
# stage X at 1.0000/1.0000/0.9568.
#
#   BAR: conditional guard rate below 0.90 in >= 2 of 3 seeds, lower better.
#
#   THE DENOMINATOR TRAP, walked into three times now by three different
#   scripts. The published rate is CONDITIONAL on clean_emits_v and its
#   denominator is how many of the 350 cases the model named cleanly -- 232/
#   237/233 for the control, 194/175/185 for stage X, NOT 350. Worse, job
#   52285735's own log prose said "of the 156 skips", which is just 350-194 and
#   not a guard rate at all; reading it would have scored X2 backwards. So this
#   job computes every rate with ddguard_score.ratio(), prints the denominator
#   beside it, and never prints a bare count that could be mistaken for one.
#
# Y3, THE SHAPE, and it is new in stage Y. Among the cases where the guard
# FIRES, what does the model emit instead? guard_shape.shape() on the same
# CSVs: node_token >= 0.90 and still_forced >= 0.60 in >= 2 of 3 seeds. The
# untied model is the reference at 0.966 and 0.903 (finding 74). A guard that
# passes Y2 and fails Y3 is stage X's stop-shaped block suppression rebuilt at
# 320 parameters' cost, and the scorer returns "design refuted" rather than
# crediting it.
#
# The AUROC of the channel is NOT a bar here. Stage X reached 0.947 on the
# readout and bought zero counterfactual responsiveness (finding 75), which is
# the whole reason stage Y exists; a second high AUROC would say nothing new.
# It is measured and printed for the record, off the same fixed held-out set
# as the control, and it is scored by no bar.
#
# Y1 and Y4 are measured by the training array itself. They are re-read here
# from its artifacts so that ONE job prints the registered verdict for all four
# bars, which is the only thing allowed to say whether the stage passed.
# =============================================================================

#SBATCH --job-name=stageyGuard
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

Y="${SCRATCH}/spin_gnn/sudoku_n48_determinacy_Y_2026-11-14"
S="${SCRATCH}/spin_gnn/sudoku_n48_tiedwidth_S_2026-11-10"
RUN="${SCRATCH}/spin_gnn/runs/2026-11-14_stagey_guard"
mkdir -p "$RUN"

# Finding 68's constants, copied so the case list is the SAME one. If any of
# these three changes the comparison against 1.0000 x3 is void.
SEED_START=600000000
NINST=800
NCASES=350

TAG="d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_chan"
YC0="${Y}/t00_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s0/${TAG}"
YC1="${Y}/t01_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s1/${TAG}_s01"
YC2="${Y}/t02_d0320_ff01024_h04_tied_no_mlp_L16_dw1.0_chan_s2/${TAG}_s02"
STAG="d0320_h04_L16_ff01024_lr1e-03_pre_shuf_fall_noff_tied"
SC0="${S}/t00_d0320_ff01024_h04_tied_no_mlp_L16_s0/${STAG}"
SC1="${S}/t01_d0320_ff01024_h04_tied_no_mlp_L16_s1/${STAG}_s01"
SC2="${S}/t02_d0320_ff01024_h04_tied_no_mlp_L16_s2/${STAG}_s02"

for c in "$YC0" "$YC1" "$YC2" "$SC0" "$SC1" "$SC2"; do
    [ -f "${c}/model.pt" ] || { echo "missing checkpoint: ${c}"; exit 1; }
done

echo "=== every stage-Y cell is the registered one, checked before measuring ==="
python3 - "$YC0" "$YC1" "$YC2" <<'PY'
import json, sys
import scripts.stagey_score as sy
for p in sys.argv[1:]:
    c = json.load(open("%s/config.json" % p))
    dw = float(c.get("determinacy_weight", 0.0) or 0.0)
    assert dw == sy.LAMBDA, ("not the registered lambda", p, dw)
    assert c.get("determinacy_channel") is True, (
        "this checkpoint has no determinacy channel -- it is a stage-X cell "
        "in a stage-Y directory", p)
    assert c["deterministic"] is True and c["n_nodes"] == 48
    assert c["arch_mode"] == "tied_no_mlp" and c["num_layers"] == 16
    assert c["n_trainable_params"] == 614138, c["n_trainable_params"]
    assert len(json.load(open("%s/history.json" % p))) == 50
    print("  seed %s  lambda %g  channel on  614,138 trainable  50 epochs"
          % (c["seed"], dw))
PY

date +'start %F %T'

echo
echo "=== Y2: finding 68's guard probe, unchanged, on the stage-Y cells ==="
echo "    n_instances=${NINST} n_cases=${NCASES} seed_start=${SEED_START} -- "
echo "    identical to the run that measured S1 at 1.0000 x3 and X at 1/1/0.9568"
i=0
for CKPT in "$YC0" "$YC1" "$YC2"; do
    echo
    echo "--- guard stagey s${i} ---"
    python3 -u -m spin_gnn.sudoku_skip --checkpoint "$CKPT" \
        --n-instances "$NINST" --n-cases "$NCASES" --seed-start "$SEED_START" \
        --out-dir "${RUN}/guard/stagey_s${i}"
    i=$((i + 1))
done

echo
echo "=== the channel's AUROC, for the record, on ONE fixed set, one process ==="
echo "    scored by no bar: stage X reached 0.947 and bought nothing (f.75)"
python3 -u -m scripts.determinacy_eval \
    --checkpoint "$YC0" "$YC1" "$YC2" "$SC0" "$SC1" "$SC2" \
    --n-instances 600 \
    --out-csv "${RUN}/determinacy_both_arms.csv"

echo
echo "=== Y2, Y3 and the registered verdict ==="
python3 - "${RUN}/guard" "$Y" "$YC0" "$YC1" "$YC2" <<'PY'
import csv, json, os, sys
sys.path.insert(0, ".")
import scripts.stagey_score as sy
from scripts.ddguard_score import ratio
from scripts.guard_shape import shape
from scripts.n96_fallscan_analysis import rho_50

root, ydir, ckpts = sys.argv[1], sys.argv[2], sys.argv[3:6]

# Y2 and Y3 off the same rows of the same CSV, so the rate and the shape can
# never be computed from different case sets.
guard, shapes = [], []
for i in range(3):
    with open(os.path.join(root, "stagey_s%d" % i, "skip_behavior.csv")) as f:
        rows = list(csv.DictReader(f))
    rate, den = ratio(rows, "clean_emits_v", "corrupt_emits_v")
    _, fired, s = shape(rows)
    guard.append(rate)
    shapes.append(s)
    print("  seed %d  guard rate CONDITIONAL on naming it cleanly %.4f "
          "over %d cases  (the 350 offered is NOT the denominator)"
          % (i, rate, den))
    if s is None:
        print("           the guard never fired, so there is no shape to read "
              "and this seed cannot pass Y3")
    else:
        print("           of %d fired: node token %.3f | still forced %.3f | "
              "names v again %.3f   (untied reference 0.966 / 0.903 / 0.000)"
              % (fired, s["node_token"], s["still_forced"], s["names_v_again"]))

# Y1 from the array's own rho grids, Y4 from its own histories.
rho = []
for i in range(3):
    hits = [p for p in os.listdir(ydir)
            if p.startswith("rho_curve_t%02d_" % i) and p.endswith(".csv")]
    assert len(hits) == 1, ("expected exactly one rho curve for seed %d" % i, hits)
    with open(os.path.join(ydir, hits[0])) as f:
        curve = [(float(r["realized_rho"]), float(r["solve_rate"]))
                 for r in csv.DictReader(f)]
    v, flag = rho_50(curve)
    assert flag == "ok", ("censored rho_50, reported with its flag and not "
                          "silently used", i, v, flag)
    rho.append(v)
acc = [json.load(open("%s/history.json" % p))[-1]["accuracy"] for p in ckpts]

print()
print("Y4 val token accuracy %s   against the control %s"
      % ("/".join("%.4f" % a for a in acc),
         "/".join("%.4f" % a for a in sy.CONTROL_ACC)))
print("Y1 rho_50             %s   against the control %s"
      % ("/".join("%.3f" % r for r in rho),
         "/".join("%.3f" % r for r in sy.CONTROL_RHO50)))
print()
out = sy.score(acc, rho, guard, shapes)
print(json.dumps(out, indent=2))
with open(os.path.join(root, os.pardir, "stagey_verdict.json"), "w") as f:
    json.dump(out, f, indent=2)
print()
print("VERDICT: %s" % out["verdict"])
print("free line %.3f cleared by every seed: %s  (reported separately from "
      "Y1, which asks only about separation from the control)"
      % (sy.FREE_LINE, out["Y1"]["clears_free_line_separately"]))
PY

echo
echo "=== done ==="
date +'end %F %T'
