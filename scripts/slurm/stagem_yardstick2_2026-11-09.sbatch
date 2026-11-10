#!/bin/bash
#SBATCH --job-name=stagemYard2
#SBATCH --partition=sganguli
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagemYard2_%j.out
#
# The OTHER TWO yardsticks stage M needs, measured before M's registration is
# written. Read-only; nothing is written to the repo or to any run directory.
#
# 51743995 printed the rho_50 yardstick (s*_96 re-estimates to 0.117 over four
# multi-seed N=96 rungs: 0.138 / 0.115 / 0.105 / 0.111). But rho_50 CENSORS, and
# stage F's worst stripped seed landed at 0.937 with the grid's top bin at 0.95 --
# one bin of headroom. Any stage-M corner that is worse than the stripped arm can
# censor, and then its level is not quotable at all.
#
# So M registers two never-censoring metrics alongside, and a bar for each has to
# be frozen from the SAME four rungs rather than invented after the fact:
#   * mean_solve -- the mean solve rate over the 14-point grid.
#   * final-epoch validation TOKEN ACCURACY -- the standing non-negotiable says to
#     score this first; it never censors.
# Both bars are the mean |max-min| across multi-seed N=96 rungs, which is exactly
# how s*_96 was built, so the three bars are the same kind of object.
#
# NOT DONE HERE, deliberately: acc_vs_solve.py's fitted accuracy->solve curve is
# an N=48 object (it is fitted on SCAN_DIRS). It is not evaluated against any
# N=96 cell, here or in M's scorer -- a different problem size has a different
# accuracy-to-solve mapping, and borrowing the N=48 fit would be the same mistake
# as reusing s* = 0.1484 at N=96.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
    py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
    py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export MPLCONFIGDIR="${SCRATCH}/mplcache"
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
echo "commit $(git rev-parse --short HEAD)"

python3 -u - <<'PY'
from collections import defaultdict

from scripts.acc_vs_solve import val_accuracy
from scripts.minarch_analysis import cell_metrics, load_all, spread
from scripts.n96_control import SCAN_N96_COSINE, SCAN_N96_FALL, label, n96_key
from scripts.stagef_score import SCAN_F

DIRS = [SCAN_F, SCAN_N96_FALL, SCAN_N96_COSINE]

by = defaultdict(lambda: defaultdict(list))
for r in load_all(DIRS):
    if int(r["n_nodes"]) != 96:
        continue
    by[n96_key(r)][int(r["seed"])].append(r)

print("every leak-free N=96 rung on disk, per-seed. boot=200 only because "
      "cell_metrics divides by n_boot; no CI is read below.")
ms_parts, acc_parts = [], []
for key in sorted(by):
    seeds = sorted(by[key])
    cells = {s: cell_metrics(by[key][s], 200) for s in seeds}
    accs = {s: val_accuracy(cells[s]["path"]) for s in seeds}
    print("\n  %s   [%d seed(s)]" % (label(key), len(seeds)))
    for s in seeds:
        print("    s%d  mean_solve %.4f   val_token_acc %s   n_params %s"
              % (s, cells[s]["mean_solve"],
                 ("%.4f" % accs[s]) if accs[s] == accs[s] else "   n/a",
                 format(cells[s]["n_params"], ",")))
    if len(seeds) > 1:
        v = spread([cells[s]["mean_solve"] for s in seeds])
        print("    spread(mean_solve) %.4f" % v)
        ms_parts.append((label(key), v, len(seeds)))
        a = [accs[s] for s in seeds if accs[s] == accs[s]]
        if len(a) > 1:
            w = max(a) - min(a)
            print("    spread(val_token_acc) %.4f  (%d seeds)" % (w, len(a)))
            acc_parts.append((label(key), w, len(a)))

for name, parts in (("mean_solve", ms_parts), ("val token accuracy", acc_parts)):
    print("\n=== CANDIDATE N=96 BAR FOR %s ===" % name.upper())
    if not parts:
        print("  nothing multi-seed to estimate from.")
        continue
    for lab, v, n in parts:
        print("    %-52s %.4f  (%d seeds)" % (lab, v, n))
    print("  mean over %d rung(s) = %.4f" % (len(parts),
                                             sum(v for _, v, _ in parts) / len(parts)))
PY
echo "rc=$?"
echo "end $(date '+%F %T')"
