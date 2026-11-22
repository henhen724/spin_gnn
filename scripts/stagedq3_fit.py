#!/usr/bin/env python3
"""Score stage DQ3's E1-E5 against runs/stagedq3_2026-11-22/REGISTERED.md.

The readers, the Clopper-Pearson helpers and the token-weighted pooling come
straight from `scripts/stagedq_fit.py` -- including its two guards, which are
the reason to import rather than re-type: `load_curves` REFUSES a curve whose
(degree, n_colors) is not (4, 3), and `load_sections` REFUSES a row whose
vocabulary is not N + 6 + 3. A q=4 checkpoint therefore cannot be scored
against ln 3 by accident. Both loaders take an explicit glob, so nothing in
that module needs rebinding here.

The SCORING is new, because E1-E5 are not D1-D5. The differences that matter:

  * every E is scored PER CELL. Stage DQ's pooled D2 averaged a cell at the
    prior with a cell holding the rule exactly and read as a depth trend; any
    pooled number printed below is printed beside its parts.
  * E1 can MISS, and a MISS is informative: it says the colour head's state is
    continuous, and "bimodal" is the wrong description.
  * E2 carries an explicit falsifier, not just a target.

    python3 -m scripts.stagedq3_fit [--out <dir>]
"""
import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.stagedq_fit import (CHANCE, N_COLORS, a_eff_or_bound, fmt_a,
                                 history, load_curves, load_sections, pooled)

SCRATCH = os.environ.get("SCRATCH", "/scratch/users/hshunt")
SCAN_DQ3 = os.path.join(SCRATCH, "spin_gnn/sudoku_dq3_2026-11-22")
SPLIT_DQ3 = os.path.join(SCRATCH, "spin_gnn/runs/stagedq3_2026-11-22")
SCAN_DQ = os.path.join(SCRATCH, "spin_gnn/sudoku_dq_2026-11-21")
SPLIT_DQ = os.path.join(SCRATCH, "spin_gnn/runs/stagedq_2026-11-21")

# --- registered bars, transcribed from REGISTERED.md, changed by nobody ----
E1_RULE = 0.10          # below this, the colouring rule is learned
E1_PRIOR = 0.99         # above this, the head is at the prior (ln 3 = 1.0986)
E4_BAR = 0.01           # the zero-floor bar D3 used, applied to round_mark
E5_STANDING = 0.99191   # N=96 s1 at rho=0.32, the standing best on this eval
E5_RHO = 0.32
CELLS = [(96, 2), (120, 2)]
N_PER_BIN = 400


def state(v):
    """E1's three-way classification, with the registered band in the middle."""
    if v is None:
        return "missing"
    if v < E1_RULE:
        return "rule exact"
    if v > E1_PRIOR:
        return "at the prior"
    return "INTERMEDIATE"


def sections(split_dir):
    return load_sections(os.path.join(split_dir, "dqsplit_*.csv"))


def curves(scan_dir):
    return load_curves(os.path.join(scan_dir, "rho_curve_n*_s*_*.csv"))


def line(out, s=""):
    print(s, flush=True)
    out.append(s)


def table(out, sec, cells, title):
    line(out, title)
    line(out, "   %-11s %10s %8s   %10s %8s   %10s %8s"
         % ("cell", "rea_color", "acc", "rea_node", "acc", "round_mk", "acc"))
    for n, s in cells:
        rows = sec.get((n, s))
        if not rows:
            line(out, "   N=%-4d s%d  %s" % (n, s, "NO SECTION DATA"))
            continue
        vals = []
        for name in ("reason_color", "reason_node", "round_mark"):
            v, _k, a = pooled(rows, name)
            vals += ["%10.4f" % v if v is not None else "       n/a",
                     "%8.4f" % a if a is not None else "     n/a"]
        line(out, "   N=%-4d s%d  %s %s   %s %s   %s %s"
             % (n, s, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]))


def score_e1(sec, out):
    line(out, "E1 -- is reason_color bimodal? (per cell, pooled over density)")
    states = []
    for n, s in CELLS:
        rows = sec.get((n, s))
        v = pooled(rows, "reason_color")[0] if rows else None
        states.append(state(v))
        line(out, "   N=%-4d s%d  reason_color %s  -> %s"
             % (n, s, "%.4f" % v if v is not None else "   n/a", states[-1]))
    if "missing" in states:
        line(out, "E1 UNSCOREABLE: a cell is missing. Not read as a result.")
    elif "INTERMEDIATE" in states:
        line(out, "E1 MISS: a cell landed inside the %.2f-%.2f band, so the "
                  "state is CONTINUOUS and 'bimodal' is the wrong word. Stage "
                  "DQ's N=96 s0 at 0.4468 was already in that band -- this "
                  "makes two, and the bimodal reading is dropped."
             % (E1_RULE, E1_PRIOR))
    else:
        line(out, "E1 HIT: every cell is at an extreme, consistent with the "
                  "colour head having two basins rather than a continuum.")
    return states


def score_e2(sec, out):
    line(out, "E2 -- is N the driver? (falsified if N=120 s2 is at the prior)")
    rows = sec.get((120, 2))
    v = pooled(rows, "reason_color")[0] if rows else None
    if v is None:
        line(out, "   N=120 s2 missing -- E2 UNSCOREABLE")
        return
    line(out, "   N=120 s2  reason_color %.4f   chance ln %d = %.6f"
         % (v, N_COLORS, CHANCE))
    if v > E1_PRIOR:
        line(out, "E2 FALSIFIED: 2 of 3 seeds at N=120 sit at the prior, so N "
                  "IS doing the work and stage DQ's N=120 s1 was the outlier.")
    elif v < E1_RULE:
        line(out, "E2 HIT: 2 of 3 seeds at N=120 hold the colouring rule, so "
                  "the collapse is not a function of size at this N.")
    else:
        line(out, "E2 INDETERMINATE: between the bars. Reported as such.")


def score_e3_e4(sec, out):
    for name, tag, desc in (
            ("reason_node", "E3", "does search degrade with N regardless?"),
            ("round_mark", "E4", "is round_mark the N-sensitive zero section?")):
        line(out, "%s -- %s" % (tag, desc))
        vals = {}
        for n, s in CELLS:
            rows = sec.get((n, s))
            v, _k, a = pooled(rows, name) if rows else (None, 0, None)
            vals[n] = v
            line(out, "   N=%-4d s%d  %-12s %s  acc %s"
                 % (n, s, name, "%.4f" % v if v is not None else "   n/a",
                    "%.4f" % a if a is not None else "  n/a"))
        if vals.get(96) is None or vals.get(120) is None:
            line(out, "%s UNSCOREABLE: a cell is missing." % tag)
            continue
        d = vals[120] - vals[96]
        line(out, "   N=120 - N=96 = %+.4f" % d)
        if tag == "E3":
            line(out, "E3 %s: %s" % (
                "HIT" if d > 0 else "MISS",
                "search is worse at the larger size, as on every stage DQ cell"
                if d > 0 else "search did NOT worsen with N on this seed, "
                              "which contradicts the one trend that held "
                              "across all four stage DQ cells"))
        else:
            above = vals[96] > E4_BAR and vals[120] > E4_BAR
            line(out, "E4 %s: ordering %s, both above the %.2f bar %s"
                 % ("HIT" if (d > 0 and above) else "MISS",
                    "holds" if d > 0 else "fails", E4_BAR,
                    "yes" if above else "NO"))
            line(out, "   D3 as registered never gated this section even "
                      "though its data entropy is exactly zero.")


def score_e5(cur, out, ref=None):
    line(out, "E5 -- a_eff at rho=%.2f, the standing metric" % E5_RHO)
    line(out, "   (standing best on this eval: %.5f, N=96 s1 of stage DQ)"
         % E5_STANDING)
    best = None
    for key in sorted(set(list(cur) + (list(ref) if ref else []))):
        rows = cur.get(key) or (ref or {}).get(key)
        hit = [r for r in rows if abs(r["rho"] - E5_RHO) < 1e-9]
        if not hit:
            continue
        r = hit[0]
        nb = r["n_nodes"] - round(r["n_nodes"] * r["realized_rho"])
        pair = a_eff_or_bound(r["solve_rate"], nb, r["n_instances"])
        tag = "DQ3" if key in cur else "DQ "
        line(out, "   %s N=%-4d s%d  T %6.2f  n_blank %3d  solve %.4f  "
                  "a_eff %s" % (tag, r["n_nodes"], r["seed"], r["dd_mean_T"],
                                nb, r["solve_rate"], fmt_a(pair)))
        if key in cur and not pair[1] and (best is None or pair[0] > best[0]):
            best = (pair[0], key)
    if best is None:
        line(out, "E5 MISS: no DQ3 cell produced a MEASURED a_eff at this "
                  "depth -- every one solved 0 of %d, which bounds the rate "
                  "from above and does not pin it. a_eff = 0 is NOT reported."
             % N_PER_BIN)
    elif best[0] > E5_STANDING:
        line(out, "E5 HIT: N=%d s%d reaches %.5f, above the standing %.5f."
             % (best[1][0], best[1][1], best[0], E5_STANDING))
    else:
        line(out, "E5 MISS: best DQ3 cell is %.5f at N=%d s%d, below the "
                  "standing %.5f. No improvement is claimed."
             % (best[0], best[1][0], best[1][1], E5_STANDING))


def score_trainability(out):
    line(out, "trainability (same readout as stage DQ's D5, not a prediction)")
    for n, s in CELLS:
        root = os.path.join(SCAN_DQ3, "n%d_d4_q%d_L12_s%d" % (n, N_COLORS, s))
        hits = sorted(glob.glob(os.path.join(root, "*", "history.json")))
        if len(hits) != 1:
            line(out, "   N=%-4d s%d  no single history.json under %s"
                 % (n, s, root))
            continue
        h = history(os.path.dirname(hits[0]))
        b = min(h, key=lambda r: r["nll"])
        line(out, "   N=%-4d s%d  %d ep  first %.4f -> final %.4f  best %.4f "
                  "@ep%d  gap %+.4f" % (n, s, len(h), h[0]["nll"],
                                        h[-1]["nll"], b["nll"], b["epoch"],
                                        h[-1]["nll"] - b["nll"]))
    line(out, "   A gap of +0.0002 is an ORDERING in a noisy tail, not the "
              "registered 'flat at its initial plateau'. Stage DQ's N=120 "
              "cells flagged on exactly that and fell 3.06 -> 1.74.")


def main():
    ap = argparse.ArgumentParser()
    # A FILE path, matching stagedq_fit.py's convention exactly. The two are
    # invoked from sibling sbatch scripts and a silent disagreement about
    # whether --out is a file or a directory would put the verdict somewhere
    # nobody looks.
    ap.add_argument("--out", default=os.path.join(SPLIT_DQ3, "VERDICT.txt"))
    args = ap.parse_args()

    out = []
    line(out, "STAGE DQ3 -- scored against runs/stagedq3_2026-11-22/"
              "REGISTERED.md")
    line(out, "ensemble d=4 q=%d   chance = ln %d = %.6f nats"
         % (N_COLORS, N_COLORS, CHANCE))
    line(out, "one knob moves against stage DQ: the seed, 0/1 -> 2")
    line(out)

    sec3 = sections(SPLIT_DQ3)
    cur3 = curves(SCAN_DQ3)
    if not sec3 and not cur3:
        raise SystemExit("no DQ3 artefacts under %s or %s yet"
                         % (SPLIT_DQ3, SCAN_DQ3))
    missing = os.path.join(SPLIT_DQ3, "MISSING.txt")
    if os.path.exists(missing):
        line(out, "!!! MISSING.txt exists -- a cell did not land:")
        line(out, "    " + open(missing).read().strip().replace("\n",
                                                                "\n    "))
        line(out)

    # Stage DQ's own cells are loaded for CONTEXT and labelled as such. They
    # are a different run in a different directory and are never pooled with
    # the new seed; printing them here is what makes a per-cell comparison
    # possible without anyone re-deriving stage DQ's numbers by hand.
    try:
        secD, curD = sections(SPLIT_DQ), curves(SCAN_DQ)
    except SystemExit:
        secD, curD = {}, {}

    table(out, dict(secD, **sec3),
          sorted(set(list(secD) + list(sec3))),
          "all cells, both stages (DQ seeds 0/1 for context, DQ3 seed 2 new)")
    line(out)
    score_e1(sec3, out)
    line(out)
    score_e2(sec3, out)
    line(out)
    score_e3_e4(sec3, out)
    line(out)
    score_e5(cur3, out, ref=curD)
    line(out)
    score_trainability(out)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\nwrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
