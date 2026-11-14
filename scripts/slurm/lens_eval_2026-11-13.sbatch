#!/bin/bash
#SBATCH --job-name=lensEval
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/lensEval_%j.out
#
# FINDING 72: where in the iteration is the decision made?
#
# Nothing is trained here. Six checkpoints that already exist (S1 s0-2 tied,
# J2 s0-2 untied), finding 68's byte-identical round-0 case list, two
# instruments per cell:
#   - a LENS, one forward per (case, variant), reading the model's own norm
#     and readout off the state after every application;
#   - a PATCH, splicing the clean state into the broken run at one
#     (application, site) and asking when repair stops working.
#
# The registration is scripts/lens_score.py and it is COMMITTED BEFORE THIS
# JOB RUNS, so no bar in it can be chosen after the cells land. The headline
# L3 -- recovery drops from 0.80 to 0.20 across 2 applications, in 2 of 3
# untied seeds -- has a registered falsifier that is itself a result: a
# gradual decline means the guard is DISTRIBUTED, and the scorer prints that
# verdict rather than a failure.
#
# Two gates fail the whole run closed, both recomputed by the scorer rather
# than trusted from the module: L1, the last-application lens must reproduce
# the model's own output exactly (it is the same computation, so anything else
# is a bug in how the readout is applied); and A2, splicing the clean state
# into the CLEAN run must change nothing, which is what stops a mis-indexed
# hook from manufacturing a recovery curve.
#
# Steps 1-3 are cheap and ordered so a mistake is caught before the six cells:
# the cell table is checked for existence first, the census prints how many
# cases the stream yields (finding 68 got 350), and a small smoke cell runs
# both instruments end to end and asserts both gates on it. Only then do the
# real cells run. Aggregated into one job because each cell is minutes, not
# hours -- six jobs of four minutes would be six scheduler round trips.

set -uo pipefail
cd /scratch/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date +'start %F %T'

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

RUN=/scratch/users/hshunt/spin_gnn/runs/2026-11-13_lens
SMOKE=$RUN/_smoke
mkdir -p "$RUN"

echo
echo "=== 1. cell table ================================================="
# Emitted from ddguard_score.CKPT rather than retyped, so the six cells here
# are the same six checkpoints findings 68-71 scored.
python3 - "$RUN/cells.txt" <<'PY' || exit 1
import os, sys
sys.path.insert(0, "/scratch/users/hshunt/spin_gnn")
from scripts.ddguard_score import ARMS, SEEDS, CKPT
missing = []
with open(sys.argv[1], "w") as fh:
    for arm in ARMS:
        for s in SEEDS:
            p = CKPT[arm][s]
            ok = os.path.isdir(p)
            print("  %-6s %s %s" % ("%s_s%d" % (arm, s), "OK " if ok else "MISSING", p))
            if not ok:
                missing.append(p)
            fh.write("%s_s%d\t%s\n" % (arm, s, p))
if missing:
    raise SystemExit("%d checkpoints missing -- nothing run" % len(missing))
PY

echo
echo "=== 2. census (how many round-0 cases does the stream yield?) ====="
HEAD=$(head -n 1 "$RUN/cells.txt" | cut -f2)
python3 -m spin_gnn.sudoku_lens --checkpoint "$HEAD" --cases-only || exit 1

echo
echo "=== 3. smoke cell, both instruments, both gates ==================="
rm -f "$SMOKE"/lens_*.csv "$SMOKE"/lens_meta.json
python3 -m spin_gnn.sudoku_lens --checkpoint "$HEAD" --out-dir "$SMOKE" \
    --n-cases 8 --n-patch 4 --n-noop 2 || { echo "SMOKE RUN FAILED"; exit 1; }
python3 - "$SMOKE" <<'PY' || { echo "SMOKE GATES FAILED -- six cells not run"; exit 1; }
import json, sys
sys.path.insert(0, "/scratch/users/hshunt/spin_gnn")
from scripts.lens_score import LENS_GAP, _num
from scripts.ddguard_score import rows_of, truthy
d = sys.argv[1]
meta = json.load(open(d + "/lens_meta.json"))
traj, patch, noop = (rows_of("%s/lens_%s.csv" % (d, n))
                     for n in ("traj", "patch", "noop"))
last = [r for r in traj if int(r["t"]) == meta["depth"] - 1]
agree = sum(_num(r["emits_v"]) == _num(r["final_emits_v"]) for r in last) / len(last)
gap = max(abs(_num(r["p_freed"]) - _num(r["final_p_freed"])) for r in last)
same = sum(truthy(r["same_emits_v"]) for r in noop) / len(noop)
drift = max(_num(r["abs_freed_drift"]) for r in noop)
print("  depth=%d tied=%s  rows traj/patch/noop = %d/%d/%d"
      % (meta["depth"], meta["tied"], len(traj), len(patch), len(noop)))
print("  L1 agree %.4f gap %.2e   A2 same %.4f drift %.2e" % (agree, gap, same, drift))
assert agree >= 1.0 and gap <= LENS_GAP, "L1: the lens is not the model at t=depth-1"
assert same >= 1.0 and drift <= LENS_GAP, "A2: the splice is not a no-op on its own source"
print("  both gates pass on the smoke cell")
PY

echo
echo "=== 4. six cells =================================================="
while IFS=$'\t' read -r NAME CKPT; do
    echo
    echo "--- $NAME ---"
    date +'  %T'
    python3 -m spin_gnn.sudoku_lens --checkpoint "$CKPT" --out-dir "$RUN/$NAME" \
        --n-cases 350 --n-patch 150 --n-noop 10 \
        || { echo "CELL $NAME FAILED"; exit 1; }
done < "$RUN/cells.txt"

echo
echo "=== 5. score ======================================================"
date +'  %T'
python3 scripts/lens_score.py || { echo "SCORER FAILED"; exit 1; }

date +'end %F %T'
