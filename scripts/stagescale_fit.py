"""Scorer for stage SCALE.  Committed BEFORE the probe ran; see the sbatch
header for the registered predictions C1-C6 and diagnostics D1-D3.

Stdlib only, so it runs in any python3 on this cluster.
"""
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

SCAN = "/scratch/users/hshunt/spin_gnn/runs/stagescale_2026-11-20"

C1_LO, C1_HI = 0.03, 0.20          # drop in reason_node acc, N=48 -> N=96
C1_FALSIFIER_LO, C1_FALSIFIER_HI = 0.01, 0.30
C2_BAR, C2_FALSIFIER = 0.05, 0.15  # the same drop AT MATCHED DEPTH
C2_MIN_TOK = 200                   # per (N, band); below this the band is skipped
C4_LO, C4_HI = 0.40, 0.75          # rise in graph_nbr_novel, log 2 = 0.6931
C4_FALSIFIER_LO, C4_FALSIFIER_HI = 0.25, 0.95
C5_BAR, C5_FALSIFIER = 0.99, 0.95
C6_BAR, C6_FALSIFIER = 0.15, 0.35  # change in graph_nbr_copy
RHO_MAIN = 0.60
PAIRS = ["L08_s0", "L12_s0", "L08_s1"]


def load():
    """argv[1] overrides the scan dir, so the same scorer reads the smoke dir
    and, later, the N=192 rerun."""
    rows = []
    scan = sys.argv[1] if len(sys.argv) > 1 else SCAN
    globals()["SCAN"] = scan
    for path in sorted(glob.glob(os.path.join(scan, "scale_*of*.csv"))):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                r["rho"] = float(r["rho"])
                r["n_nodes"] = int(r["n_nodes"])
                for k in ("nll_sum", "total_nll", "total_accuracy", "mean_T"):
                    r[k] = float(r[k])
                for k in ("nll", "accuracy"):
                    r[k] = float(r[k]) if r[k] != "" else None
                for k in ("n_tokens", "n_correct", "d1_bad", "d3_bad", "n_skip"):
                    r[k] = int(r[k])
                rows.append(r)
    return rows


def grouped(rows):
    g = defaultdict(dict)
    for r in rows:
        g[(r["run_id"], r["rho"])][r["section"]] = r
    return g


def median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def pair_runs(rows):
    """pair_key -> {n_nodes: run_id}, for the cells that exist at both N."""
    out = defaultdict(dict)
    for r in rows:
        if r["pair_key"]:
            out[r["pair_key"]][r["n_nodes"]] = r["run_id"]
    return out


def depth_table(rows):
    """(n_nodes, T) -> [n_correct, n_total] on reason_node, pooled over the
    matched pairs and over every density.  Depth is the physical variable; rho
    is only the knob that sets it, and it sets a DIFFERENT depth at each N."""
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["section"] != "reason_node" or not r["pair_key"]:
            continue
        for t, (c, n) in json.loads(r["by_T"]).items():
            cell = out[(r["n_nodes"], int(t))]
            cell[0] += c
            cell[1] += n
    return out


def main():
    rows = load()
    if not rows:
        print("no CSVs in %s" % SCAN)
        return 1
    g = grouped(rows)
    pr = pair_runs(rows)
    rhos = sorted({r["rho"] for r in rows})
    print("%d rows, %d runs, rhos %s"
          % (len(rows), len({r["run_id"] for r in rows}), rhos))

    # ---------------- diagnostics first: every C is VOID if one fails -------
    d1 = sum(r["d1_bad"] for r in rows if r["section"] == "term")
    d3 = sum(r["d3_bad"] for r in rows if r["section"] == "term")
    skip = sum(r["n_skip"] for r in rows if r["section"] == "term")
    print("\nD1 label walk vs encode_instance: %d bad  %s"
          % (d1, "PASS" if d1 == 0 else "FAIL"))
    print("D3 length == N(degree+3)+T+4: %d bad  %s  (max_len skips %d)"
          % (d3, "PASS" if d3 == 0 else "FAIL", skip))
    worst = 0.0
    for (run, rho), sec in g.items():
        tot = sum(v["nll_sum"] for v in sec.values())
        ntok = sum(v["n_tokens"] for v in sec.values())
        pub = list(sec.values())[0]["total_nll"]
        worst = max(worst, abs(tot / ntok - pub))
    print("D2 sections reconstruct the total: max %.3e over %d cells  %s"
          % (worst, len(g), "PASS" if worst < 1e-6 else "FAIL"))

    # ---------------- per-section table at the main density -----------------
    for n_nodes in (48, 96):
        runs = sorted({r["run_id"] for r in rows
                       if r["n_nodes"] == n_nodes and r["pair_key"]})
        print("\n-- N=%d, rho %.2f, median over the %d matched cells"
              % (n_nodes, RHO_MAIN, len(runs)))
        print("   %-18s %8s %9s %9s %8s" % ("section", "ntok", "nll", "acc",
                                            "share"))
        for s in ["hint_node", "hint_color", "graph_lead", "graph_nbr_novel",
                  "graph_nbr_copy", "round_mark", "reason_node",
                  "reason_color", "term", "begin"]:
            cells = [g[(r, RHO_MAIN)][s] for r in runs if (r, RHO_MAIN) in g]
            if not cells:
                continue
            tots = [sum(v["nll_sum"] for v in g[(r, RHO_MAIN)].values())
                    for r in runs if (r, RHO_MAIN) in g]
            print("   %-18s %8d %9.4f %9.4f %8.4f"
                  % (s, median([c["n_tokens"] for c in cells]),
                     median([c["nll"] for c in cells if c["nll"] is not None]),
                     median([c["accuracy"] for c in cells
                             if c["accuracy"] is not None]),
                     median([c["nll_sum"] for c in cells]) / median(tots)))

    # ---------------- C1: matched rho ---------------------------------------
    print("\nC1 reason_node accuracy drop, N=48 -> N=96, at rho %.2f"
          % RHO_MAIN)
    drops = []
    for key in PAIRS:
        m = pr.get(key, {})
        if 48 not in m or 96 not in m:
            print("   %-8s MISSING" % key)
            continue
        a48 = g[(m[48], RHO_MAIN)]["reason_node"]["accuracy"]
        a96 = g[(m[96], RHO_MAIN)]["reason_node"]["accuracy"]
        drops.append(a48 - a96)
        print("   %-8s %.4f -> %.4f   drop %+.4f" % (key, a48, a96, a48 - a96))
    if not drops:
        # A nan median must never print as "MISS": an absent pair is a missing
        # measurement, not a negative result.
        print("   NO matched pair present -- C1 UNSCOREABLE")
    else:
        med = median(drops)
        inband = sum(1 for d in drops if C1_LO <= d <= C1_HI)
        fals = sum(1 for d in drops if d < C1_FALSIFIER_LO or d > C1_FALSIFIER_HI)
        print("   median drop %.4f, in [%.2f, %.2f] on %d of %d  %s"
              % (med, C1_LO, C1_HI, inband, len(drops),
                 "FALSIFIED" if fals >= 2 else
                 ("HIT" if C1_LO <= med <= C1_HI else "MISS")))

    # ---------------- C2: matched DEPTH, the live one -----------------------
    print("\nC2 the same drop AT MATCHED DD DEPTH (pooled over the matched "
          "pairs and every density)")
    dt = depth_table(rows)
    ts = sorted({t for (_n, t) in dt})
    print("   %6s %10s %10s %10s %9s" % ("T", "N=48 acc", "N=96 acc", "gap",
                                         "min ntok"))
    gaps = []
    for t in ts:
        a = dt.get((48, t))
        b = dt.get((96, t))
        if not a or not b:
            continue
        if min(a[1], b[1]) < C2_MIN_TOK:
            continue
        ga = a[0] / a[1] - b[0] / b[1]
        gaps.append((t, ga, min(a[1], b[1])))
        print("   %6d %10.4f %10.4f %+10.4f %9d"
              % (t, a[0] / a[1], b[0] / b[1], ga, min(a[1], b[1])))
    if not gaps:
        print("   NO OVERLAPPING DEPTH with >= %d tokens at both N -- C2 is "
              "UNSCOREABLE and says so rather than being read off a thin band."
              % C2_MIN_TOK)
    else:
        deepest = max(gaps, key=lambda x: x[0])
        print("   deepest common band T=%d: gap %+.4f  %s"
              % (deepest[0], deepest[1],
                 "FALSIFIED (N carries a per-step penalty beyond depth)"
                 if abs(deepest[1]) > C2_FALSIFIER else
                 ("HIT (depth explains it)" if abs(deepest[1]) <= C2_BAR
                  else "MISS")))
        print("   median gap over %d common bands %+.4f"
              % (len(gaps), median([x[1] for x in gaps])))

    # ---------------- C3: deduction share falls with N ----------------------
    print("\nC3 deduction share of total NLL at rho %.2f" % RHO_MAIN)
    ok = seen = 0
    for key in PAIRS:
        m = pr.get(key, {})
        if 48 not in m or 96 not in m:
            continue
        seen += 1
        sh = {}
        for n in (48, 96):
            sec = g[(m[n], RHO_MAIN)]
            tot = sum(v["nll_sum"] for v in sec.values())
            sh[n] = (sec["reason_node"]["nll_sum"]
                     + sec["reason_color"]["nll_sum"]) / tot
        ok += sh[96] < sh[48]
        print("   %-8s %.4f -> %.4f  %s" % (key, sh[48], sh[96],
                                            "lower" if sh[96] < sh[48] else "HIGHER"))
    if not seen:
        print("   NO matched pair present -- C3 UNSCOREABLE")
    else:
        print("   lower at N=96 on %d of %d present  %s"
              % (ok, seen, "HIT" if ok == seen == len(PAIRS) else "MISS"))

    # ---------------- C4/C6: the two graph slots scale differently ----------
    for name, lo, hi, flo, fhi, sec_key, sign in (
            ("C4 graph_nbr_novel RISE (log 2 = %.4f)" % math.log(2),
             C4_LO, C4_HI, C4_FALSIFIER_LO, C4_FALSIFIER_HI,
             "graph_nbr_novel", +1),
            ("C6 graph_nbr_copy CHANGE (expected N-invariant)",
             -C6_BAR, C6_BAR, -C6_FALSIFIER, C6_FALSIFIER,
             "graph_nbr_copy", +1)):
        print("\n%s, N=48 -> N=96 at rho %.2f" % (name, RHO_MAIN))
        vals = []
        for key in PAIRS:
            m = pr.get(key, {})
            if 48 not in m or 96 not in m:
                continue
            v48 = g[(m[48], RHO_MAIN)][sec_key]["nll"]
            v96 = g[(m[96], RHO_MAIN)][sec_key]["nll"]
            vals.append(sign * (v96 - v48))
            print("   %-8s %.4f -> %.4f   change %+.4f" % (key, v48, v96,
                                                           v96 - v48))
        if vals:
            med = median(vals)
            inb = sum(1 for v in vals if lo <= v <= hi)
            fl = sum(1 for v in vals if v < flo or v > fhi)
            print("   median %+.4f, in [%.2f, %.2f] on %d of %d  %s"
                  % (med, lo, hi, inb, len(vals),
                     "FALSIFIED" if fl >= 1 else
                     ("HIT" if inb == len(vals) else "MISS")))

    # ---------------- C5: the colour step is local and should not care ------
    print("\nC5 reason_color accuracy at N=96, rho >= %.2f" % RHO_MAIN)
    bad = worstv = 0
    worstv = 1.0
    for key in PAIRS:
        m = pr.get(key, {})
        if 96 not in m:
            continue
        for rho in [r for r in rhos if r >= RHO_MAIN]:
            a = g[(m[96], rho)]["reason_color"]["accuracy"]
            worstv = min(worstv, a)
            bad += a < C5_BAR
            print("   %-8s rho %.2f  %.4f" % (key, rho, a))
    print("   worst %.4f  %s" % (worstv,
                                 "FALSIFIED" if worstv < C5_FALSIFIER else
                                 ("HIT" if bad == 0 else "MISS")))

    # ---------------- context: every run, not only the pairs ----------------
    print("\nAll runs at rho %.2f (the unmatched cells are CONTEXT and score "
          "nothing)" % RHO_MAIN)
    print("   %-12s %5s %9s %10s %10s %8s"
          % ("run", "N", "total", "reas_node", "novel", "meanT"))
    for run in sorted({r["run_id"] for r in rows}):
        if (run, RHO_MAIN) not in g:
            continue
        sec = g[(run, RHO_MAIN)]
        print("   %-12s %5d %9.4f %10.4f %10.4f %8.2f"
              % (run, sec["reason_node"]["n_nodes"],
                 sec["reason_node"]["total_nll"],
                 sec["reason_node"]["accuracy"],
                 sec["graph_nbr_novel"]["nll"],
                 sec["reason_node"]["mean_T"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
