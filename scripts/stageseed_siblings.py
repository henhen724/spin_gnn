#!/usr/bin/env python3
"""Finding 135 -- which single-seed cells in the 79 have unevaluated sibling seeds.

Read-only. Nothing is trained, nothing is submitted, nothing is written outside
stdout. This is the audit stage SEED (134) left behind: d192's seeds 1 and 2 had
been on disk since 2026-11-10 and were unmeasured only because the population
gate dropped them, so the obvious question is how many other cells are in that
position -- and the answer costs a directory walk, not a GPU.

TWO FACTS ABOUT THE LAYOUT, BOTH OF WHICH COST ME A WRONG GLOB:

  * checkpoints are NOT under `runs/`. They sit at the repo root as
    `<scan dir>/<task dir>/<tag dir>/`, e.g.
    `sudoku_n48_tied_P_2026-11-10/t01_d0192_..._s1/d0192_h04_L16_..._tied_s01/`,
    and `runs/` holds only the scorers' outputs.
  * the tag directory encodes width BEFORE arch_mode (`d0192_..._tied_no_mlp_L16`),
    so a glob written as `*tied_no_mlp_L16*d0192*` silently matches nothing.

Keying is therefore on the CONFIG, not on any path pattern: read every
`config.json` under the root and compare `(arch_mode, num_layers, d_model)`.

The gate being audited is `stagepos_cells.successful_runs()`:

    if len(at50) == 1 and float(at50[0]["solve_rate"]) > 0:

i.e. a cell enters the 79 only if it solved at least one instance of 400 at
rho = 0.50. That is a gate on SOLVING, and 134 showed it is not neutral on the
rule arm, so every single-seed row in the ladder is read against this table.

  module purge --force; module load devel math python/3.12.1
  python3 scripts/stageseed_siblings.py
"""

import collections
import csv
import glob
import json
import os
import sys

ROOT = "/scratch/users/hshunt/spin_gnn"
LOGIC = os.path.join(ROOT, "runs", "stagelogic_2026-11-17")

# The gate's own threshold, quoted from stagepos_cells.successful_runs().
GATE_RHO = 0.50

# Siblings stage SEED (134) ALREADY measured, as r80/r81. They are still dropped
# by the gate, so the audit finds them like any other; subtracting them here is
# what keeps this script's count equal to the one STATUS.md quotes.
MEASURED_BY_134 = {
    "sudoku_n48_tied_P_2026-11-10/t01_d0192_ff01024_h04_tied_no_mlp_L16_s1",
    "sudoku_n48_tied_P_2026-11-10/t02_d0192_ff01024_h04_tied_no_mlp_L16_s2",
}


def num(x, default=float("nan")):
    """rho_curve rows carry blank val_nll for some scans; never let that raise."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def population():
    """run_id -> record, from the stage LOGIC summary shards (the published 79)."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(LOGIC, "LF_summary_all_*of10.csv"))):
        for r in csv.DictReader(open(f)):
            runs.setdefault(r["run_id"], r)
    return runs


def single_seed_cells(runs):
    """Cells with exactly one run inside the 79, smallest first."""
    cells = collections.defaultdict(list)
    for rid, r in runs.items():
        if r["in79"] == "True":
            cells[r["cell"]].append(r)
    singles = [(c, v[0]) for c, v in cells.items() if len(v) == 1]
    singles.sort(key=lambda t: int(t[1]["n_trainable_params"]))
    return cells, singles


def checkpoints(keys):
    """(arch_mode, num_layers, d_model) -> [(seed, relpath, epochs)] with a model.pt.

    Three glob depths because the scans nest inconsistently; the config decides
    membership, so an extra depth costs a stat and never a false positive.
    """
    hits = collections.defaultdict(list)
    for pat in ("/*/*/config.json", "/*/*/*/config.json", "/*/*/*/*/config.json"):
        for cfgp in glob.glob(ROOT + pat):
            try:
                c = json.load(open(cfgp))
            except Exception:
                continue
            k = (c.get("arch_mode", "full"), c.get("num_layers"), c.get("d_model"))
            if k not in keys:
                continue
            d = os.path.dirname(cfgp)
            if not os.path.exists(os.path.join(d, "model.pt")):
                continue
            hits[k].append((c.get("seed"), os.path.relpath(d, ROOT), c.get("epochs")))
    return hits


def solve_at_gate(rel):
    """solve_rate at rho=0.50 from the scan's OWN rho_curve, keyed on the task prefix.

    Read from the scan rather than from any scorer's output, so this audit cannot
    inherit a filtering decision from the thing it is auditing.
    """
    parts = rel.split(os.sep)
    scan, task = parts[0], parts[1]
    out = []
    pat = os.path.join(ROOT, scan, "rho_curve_%s_*.csv" % task.split("_")[0])
    for fp in sorted(glob.glob(pat)):
        for r in csv.DictReader(open(fp)):
            if abs(num(r.get("rho"), -1.0) - GATE_RHO) < 1e-9:
                out.append((num(r.get("solve_rate")),
                            int(num(r.get("n_instances"), 0))))
    return out


def main():
    runs = population()
    cells, singles = single_seed_cells(runs)
    print("stage LOGIC shards: %d runs, %d in the 79, %d cells"
          % (len(runs), sum(1 for r in runs.values() if r["in79"] == "True"), len(cells)))
    print("seeds per cell: %s"
          % dict(sorted(collections.Counter(
              len([x for x in v]) for v in cells.values()).items())))
    print("single-seed cells: %d of %d\n" % (len(singles), len(cells)))

    keys = {(r["arch_mode"], int(r["num_layers"]), int(r["d_model"])) for _, r in singles}
    hits = checkpoints(keys)

    pending = 0
    done = 0
    for cell, r in singles:
        k = (r["arch_mode"], int(r["num_layers"]), int(r["d_model"]))
        admitted_ep = None
        rows = []
        for seed, rel, ep in sorted(set(hits.get(k, [])), key=lambda x: (x[0], x[2] or 0)):
            for sr, ni in solve_at_gate(rel) or [(float("nan"), 0)]:
                rows.append((seed, ep, sr, ni, rel))
                if sr > 0 and int(r["seed"]) == (seed if seed is not None else -1):
                    admitted_ep = ep
        print("-- %-26s %s  %s params, admitted seed %s"
              % (cell, r["run_id"], format(int(r["n_trainable_params"]), ","), r["seed"]))
        for seed, ep, sr, ni, rel in rows:
            if ni == 0:
                print("     seed %-3s ep=%-4s  no rho_curve row at %.2f   %s"
                      % (seed, ep, GATE_RHO, rel.split(os.sep)[1]))
                continue
            verdict = "IN THE 79" if sr > 0 else "DROPPED BY THE GATE"
            # Only a sibling at the SAME epoch budget as the admitted seed is a
            # like-for-like replicate; a different budget is a different cell.
            matched = (admitted_ep is not None and ep == admitted_ep and sr <= 0)
            already = "/".join(rel.split(os.sep)[:2]) in MEASURED_BY_134
            if matched and already:
                done += 1
                note = "   <- matched, MEASURED by 134"
            elif matched:
                pending += 1
                note = "   <- matched, unmeasured"
            else:
                note = ""
            print("     seed %-3s ep=%-4s  solve@%.2f %.4f (%d/%d)  %-19s %s%s"
                  % (seed, ep, GATE_RHO, sr, round(sr * ni), ni, verdict,
                     rel.split(os.sep)[1], note))
        print("")

    print("MATCHED-BUDGET SIBLINGS THE GATE DROPPED: %d  (134 measured %d, %d remain)"
          % (pending + done, done, pending))
    print("Measuring any of them is a NEW EVAL and hshunt's call. Route anything so")
    print("measured through POS_EXTRA, which flags in79=False and leaves the default")
    print("population bit-identical; never pool it into the 79.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
