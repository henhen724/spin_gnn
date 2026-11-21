"""Round-width statistics across an N ladder from 48 to 1020.

For slide 23 of the round-width deck, which currently shows the width/density
confound at N=48 and N=96 only.  hshunt asked for the width plotted over a
range "48-1024".

WHAT THIS IS, AND WHAT IT IS NOT.  Round width is a property of DD running on
the INSTANCE -- the number of cells naked-single elimination forces in one
pass -- so it needs no trained model and costs nothing to measure at any N.
Everything here is therefore instance combinatorics.  It says nothing about
model accuracy at N > 192, because no model has been trained above 192; the
accuracy-vs-width curves in the rest of the deck stay at N=48/96 and this
figure must be labelled so a reader cannot read model performance into it.

N must satisfy planted_regular_graph's constraints: n_colors | n_nodes and
n_colors*(n_colors-1) | degree*n_nodes, i.e. 4 | N and 12 | 5N.  Since
gcd(5,12)=1 that means 12 | N.  **1024 is NOT a legal size** (1024/12 is not an
integer); 1020 is the legal size nearest hshunt's requested top end.

Diagnostics rather than predictions: this is a descriptive measurement made to
draw a figure, not a hypothesis test, so there is nothing to pre-register.  The
two checks that can actually fail are
  DW1  the round widths of an instance sum to its blank-cell count
  DW2  the achieved hint density is never BELOW the target (by > 0.02)
DW2 is one-sided on purpose.  `build_instances` stops along a removal path, and
if that path's own sparsest DD-recoverable point is denser than the target it
falls back to that point -- documented behaviour, not a fault.  So the achieved
density can overshoot upward and routinely does at small N and low rho (at
N=48, rho=0.35 is below rho_c ~ 0.46 and is often unreachable).  A density
*below* target would mean a broken instance.  The overshoot is real and is
reported per cell as `mean_achieved_rho` and `max_rho_gap`, so the figure can
be labelled with what was actually sampled rather than what was asked for.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spin_gnn.config import SudokuReasoningConfig
from spin_gnn.train_sudoku_transformer import build_instances

SCRATCH = os.environ.get("SCRATCH", "/scratch/users/hshunt")
OUT = os.path.join(SCRATCH, "spin_gnn/runs/widthladder_2026-11-20")

# All divisible by 12.  48 and 96 are the two sizes with trained models, 192 is
# the ladder rung currently training, the rest are instance-only.
N_LADDER = [48, 96, 192, 288, 384, 576, 768, 1020]
RHOS = [0.90, 0.75, 0.60, 0.45, 0.35]
SEED_START = 80_000_000_000     # disjoint from the eval grid (60e9) and SCALE (70e9)


def cell(n_nodes, rho, n_inst, rho_hi=None, label=None):
    """One (N, density) cell.  `rho_hi=None` pins the density (rho_max = rho);
    giving rho_hi draws it uniformly in [rho, rho_hi], which is the mix the
    training and eval sets actually use and the mix slide 23's confound figure
    was built from."""
    hi = float(rho if rho_hi is None else rho_hi)
    cfg = SudokuReasoningConfig(n_nodes=n_nodes, degree=5, n_colors=4,
                                fall=True, shuffle_adj=True,
                                rho=float(rho), rho_max=hi)
    inst = build_instances(cfg, n_inst, SEED_START)
    widths, Ts, rhos, bad_sum, bad_rho = [], [], [], 0, 0
    hist = Counter()
    # Sum of the instance density over every round of each width, so the deck
    # can rebuild slide 23's "mean rho per width bin" exactly after pooling the
    # five density cells.  Within a bin every round contributes the same token
    # count w, so round-weighted and token-weighted means coincide.
    rho_by_w = Counter()
    for _adj, _colors, _hints, tr in inst:
        w = [len(e) for e in tr["rounds"]]
        n_blank = n_nodes - len(tr["hints"])
        if sum(w) != n_blank:                       # DW1
            bad_sum += 1
        ach = len(tr["hints"]) / n_nodes
        if ach < rho - 0.02:                        # DW2, one-sided
            bad_rho += 1
        widths.extend(w)
        hist.update(w)
        for wi in w:
            rho_by_w[wi] += ach
        Ts.append(len(w))
        rhos.append(ach)
    widths.sort()
    n = len(widths)
    mean = sum(widths) / n
    var = sum((x - mean) ** 2 for x in widths) / n
    return {
        "n_nodes": n_nodes, "rho": label or ("%.2f" % rho),
        "rho_lo": "%.2f" % rho, "rho_hi": "%.2f" % hi,
        "n_instances": len(inst),
        "n_rounds_total": n,
        "mean_width": "%.6f" % mean,
        "sd_width": "%.6f" % (var ** 0.5),
        "median_width": "%.1f" % widths[n // 2],
        "p90_width": "%.1f" % widths[int(0.90 * (n - 1))],
        "max_width": max(widths),
        "mean_T": "%.4f" % (sum(Ts) / len(Ts)),
        "mean_achieved_rho": "%.4f" % (sum(rhos) / len(rhos)),
        # overshoot above the TOP of the requested interval (0 for a cell whose
        # densities all landed inside it); the fallback can only push upward
        "max_rho_gap": "%.4f" % max(max(r - hi, 0.0) for r in rhos),
        "mean_n_blank": "%.3f" % (sum(sum(len(e) for e in tr["rounds"])
                                      for *_x, tr in inst) / len(inst)),
        "dw1_bad": bad_sum, "dw2_bad": bad_rho,
        "hist": json.dumps(sorted(hist.items())),
        "rho_by_w": json.dumps([[w, round(s, 6)]
                                for w, s in sorted(rho_by_w.items())]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=300)
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows, t_all = [], time.time()
    for N in N_LADDER:
        # the five pinned densities, then one cell drawn from the full training
        # mix [0.33, 0.95] -- the latter is what the confound figure needs
        jobs = [(r, None, None) for r in RHOS] + [(0.33, 0.95, "pool")]
        for rho, hi, lab in jobs:
            t0 = time.time()
            r = cell(N, rho, a.n_per_cell, rho_hi=hi, label=lab)
            r["seconds"] = "%.1f" % (time.time() - t0)
            rows.append(r)
            print("N=%4d rho=%-4s (ach %.3f)  mean w %7.3f  sd %6.3f  max %3d"
                  "  T %6.2f  DW1 %d DW2 %d  (%s s)"
                  % (N, r["rho"], float(r["mean_achieved_rho"]),
                     float(r["mean_width"]), float(r["sd_width"]),
                     r["max_width"], float(r["mean_T"]),
                     r["dw1_bad"], r["dw2_bad"], r["seconds"]), flush=True)
    path = os.path.join(a.out_dir, "width_ladder.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tot1 = sum(r["dw1_bad"] for r in rows)
    tot2 = sum(r["dw2_bad"] for r in rows)
    print("\nDW1 widths sum to blank count   : %d violation(s)  %s"
          % (tot1, "PASS" if tot1 == 0 else "FAIL"))
    print("DW2 achieved rho never under    : %d violation(s)  %s"
          % (tot2, "PASS" if tot2 == 0 else "FAIL"))
    worst = max(rows, key=lambda r: float(r["max_rho_gap"]))
    print("largest upward rho overshoot    : %+.4f at N=%s target rho=%s "
          "(removal-path fallback, expected)"
          % (float(worst["max_rho_gap"]), worst["n_nodes"], worst["rho"]))
    print("wrote %s  (%d rows, %.1f s total)"
          % (path, len(rows), time.time() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
