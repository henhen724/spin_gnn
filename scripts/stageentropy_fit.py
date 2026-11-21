"""Scorer for stage ENTROPY.  Committed while 53151109 was still running, so
the verdicts are read off a rule written before the numbers existed.

Scores E1-E7 and DE1-DE2 against the bars in `stageentropy_probe.py`'s
docstring.  DE3 is scored inside the probe itself.

Stdlib only.  argv[1] overrides the scan dir.
"""
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

SCAN = "/scratch/users/hshunt/spin_gnn/runs/stageentropy_2026-11-20"
SCALE = "/scratch/users/hshunt/spin_gnn/runs/stagescale_2026-11-20"

DETERMINISTIC = ("reason_node", "reason_color", "round_mark", "term")
E1_BAR, E1_FALSIFIER = 0.05, 0.30
E2_BAR, E2_FALSIFIER = 0.15, 0.60
E3_BAR, E3_FALSIFIER = 0.70, 0.30
E4_BAR, E4_FALSIFIER = 1.00, 0.50
E5_BAR, E5_FALSIFIER = -0.70, -0.30
E6_MAX_FAILS = 2
H_EDGES = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
H3_EDGE = 0.10                      # E3 counts tokens with H < 0.10 nats


def load(scan, pat):
    rows = []
    for path in sorted(glob.glob(os.path.join(scan, pat))):
        with open(path) as fh:
            rows.extend(list(csv.DictReader(fh)))
    return rows


def fnum(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", "None") else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main():
    scan = sys.argv[1] if len(sys.argv) > 1 else SCAN
    rows = load(scan, "entropy_*of*.csv")
    if not rows:
        print("no CSVs in %s" % scan)
        return 1
    cells = sorted({r["run_id"] for r in rows})
    print("%d rows, %d cells, %d densities"
          % (len(rows), len(cells), len({r["rho"] for r in rows})))

    # index: (run_id, rho, section) -> row
    ix = {(r["run_id"], r["rho"], r["section"]): r for r in rows}
    rhos = sorted({r["rho"] for r in rows}, key=float, reverse=True)

    # ---------------- DE1: token counts agree with stage SCALE --------------
    sc = {}
    for r in load(SCALE, "scale_*of*.csv"):
        sc[(r["run_id"], r["rho"], r["section"])] = int(r["n_tokens"])
    # SCALE used all 200 cached instances, this probe used --limit 100, so the
    # counts cannot match in absolute terms.  What must match is the RATIO of
    # section sizes within a cell, which is a property of the encoding.
    worst, n_cmp = 0.0, 0
    for (rid, rho, sec), r in ix.items():
        k = (rid, rho, sec)
        if k not in sc or sec == "begin":
            continue
        tot_mine = sum(int(ix[(rid, rho, s)]["n_tokens"]) for s in
                       {kk[2] for kk in ix if kk[0] == rid and kk[1] == rho})
        tot_sc = sum(sc[(rid, rho, s)] for s in
                     {kk[2] for kk in sc if kk[0] == rid and kk[1] == rho})
        if not tot_mine or not tot_sc:
            continue
        worst = max(worst, abs(int(r["n_tokens"]) / tot_mine
                               - sc[k] / tot_sc))
        n_cmp += 1
    print("\nDE1 section SHARE agrees with stage SCALE: max %.3e over %d cells"
          "  %s" % (worst, n_cmp, "PASS" if worst < 2e-2 else "FAIL"))

    # ---------------- DE2: 0 <= H <= log V ----------------------------------
    bad = 0
    for r in rows:
        h, lv = fnum(r, "mean_H"), fnum(r, "log_vocab")
        mx = fnum(r, "max_H_seen")
        if h == h and (h < 0 or h > lv + 1e-6):
            bad += 1
        if mx == mx and mx > lv + 1e-6:
            bad += 1
    print("DE2 0 <= H <= log(vocab) everywhere: %d violation(s)  %s"
          % (bad, "PASS" if bad == 0 else "FAIL"))

    # ---------------- the best cell, by reason_node accuracy ----------------
    acc_by_cell = {}
    for c in cells:
        num = sum(fnum(ix[(c, r, "reason_node")], "accuracy")
                  * int(ix[(c, r, "reason_node")]["n_tokens"])
                  for r in rhos if (c, r, "reason_node") in ix)
        den = sum(int(ix[(c, r, "reason_node")]["n_tokens"])
                  for r in rhos if (c, r, "reason_node") in ix)
        acc_by_cell[c] = num / den if den else float("nan")
    best = max(cells, key=lambda c: acc_by_cell[c])
    print("\nbest cell by pooled reason_node accuracy: %s (%.4f)"
          % (best, acc_by_cell[best]))

    def pooled_H(cell, section, only_rho=None):
        num = den = 0.0
        for r in rhos:
            if only_rho and r != only_rho:
                continue
            k = (cell, r, section)
            if k not in ix:
                continue
            n = int(ix[k]["n_tokens"])
            num += fnum(ix[k], "mean_H") * n
            den += n
        return num / den if den else float("nan")

    # ---------------- E1 / E2 ------------------------------------------------
    print("\nPER-DENSITY ENTROPY AT THE BEST CELL (nats)")
    print("  %6s %14s %15s %17s" % ("rho", "H reason_node", "H reason_color",
                                    "H graph_novel"))
    for r in rhos:
        print("  %6s %14.4f %15.4f %17.4f"
              % (r, fnum(ix[(best, r, "reason_node")], "mean_H"),
                 fnum(ix[(best, r, "reason_color")], "mean_H"),
                 fnum(ix[(best, r, "graph_nbr_novel")], "mean_H")))
    h_rc, h_rn = pooled_H(best, "reason_color"), pooled_H(best, "reason_node")
    print("\nE1 mean H reason_color at best cell: %.4f  (bar < %.2f)  %s"
          % (h_rc, E1_BAR, "FALSIFIED" if h_rc > E1_FALSIFIER else
             ("HIT" if h_rc < E1_BAR else "MISS")))
    print("E2 mean H reason_node  at best cell: %.4f  (bar < %.2f)  %s"
          % (h_rn, E2_BAR, "FALSIFIED" if h_rn > E2_FALSIFIER else
             ("HIT" if h_rn < E2_BAR else "MISS")))

    # ---------------- E3: the concentration measure -------------------------
    agg = [0] * (len(H_EDGES) + 1)
    for r in rhos:
        k = (best, r, "reason_node")
        if k in ix and ix[k]["h_bins"]:
            for i, v in enumerate(json.loads(ix[k]["h_bins"])):
                agg[i] += v
    tot = sum(agg)
    cum, frac_below = 0, {}
    for i, e in enumerate(H_EDGES):
        cum += agg[i]
        frac_below[e] = cum / tot if tot else float("nan")
    print("\nE3 DISTRIBUTION of per-token H on reason_node, best cell "
          "(%d tokens)" % tot)
    for e in H_EDGES:
        print("     H < %-5.2f : %6.2f%%" % (e, 100 * frac_below[e]))
    f3 = frac_below[H3_EDGE]
    print("  fraction with H < %.2f = %.4f  (bar >= %.2f)  %s"
          % (H3_EDGE, f3, E3_BAR,
             "FALSIFIED" if f3 < E3_FALSIFIER else
             ("HIT" if f3 >= E3_BAR else "MISS")))

    # ---------------- E4: the random floor does not collapse ----------------
    print("\nE4 graph_nbr_novel entropy per cell (must stay > %.2f)" % E4_BAR)
    worst_cell, worst_h = None, float("inf")
    for c in cells:
        h = pooled_H(c, "graph_nbr_novel")
        if h < worst_h:
            worst_cell, worst_h = c, h
    print("  lowest is %s at %.4f nats  %s"
          % (worst_cell, worst_h,
             "FALSIFIED" if worst_h < E4_FALSIFIER else
             ("HIT" if worst_h > E4_BAR else "MISS")))

    # ---------------- E5: entropy tracks competence -------------------------
    xs, ys = [], []
    print("\nE5 reasoning entropy vs reason_node accuracy, over %d cells"
          % len(cells))
    print("  %-14s %8s %12s" % ("cell", "acc", "H(reasoning)"))
    for c in cells:
        h = sum(pooled_H(c, s) * 1 for s in ("reason_node", "reason_color"))/2
        xs.append(acc_by_cell[c])
        ys.append(h)
        print("  %-14s %8.4f %12.4f" % (c, acc_by_cell[c], h))
    rho_s = spearman(xs, ys)
    print("  Spearman = %+.4f  (bar <= %.2f)  %s"
          % (rho_s, E5_BAR, "FALSIFIED" if rho_s > E5_FALSIFIER else
             ("HIT" if rho_s <= E5_BAR else "MISS")))

    # ---------------- E6: harder instances are less certain -----------------
    hi, lo = rhos[0], rhos[-1]
    fails = []
    for c in cells:
        a = (pooled_H(c, "reason_node", lo) + pooled_H(c, "reason_color", lo)) / 2
        b = (pooled_H(c, "reason_node", hi) + pooled_H(c, "reason_color", hi)) / 2
        if not (a > b):
            fails.append(c)
    print("\nE6 H(rho=%s) > H(rho=%s): fails in %d of %d cells  %s"
          % (lo, hi, len(fails), len(cells),
             "FALSIFIED" if len(fails) > E6_MAX_FAILS else "HIT"))

    # ---------------- E7: the N penalty as uncertainty ----------------------
    pairs = defaultdict(dict)
    for r in rows:
        if r["pair_key"]:
            pairs[r["pair_key"]][int(r["n_nodes"])] = r["run_id"]
    print("\nE7 reasoning entropy at matched pair, N=48 vs N=96")
    n_up = 0
    for pk in sorted(pairs):
        if 48 not in pairs[pk] or 96 not in pairs[pk]:
            continue
        a = pooled_H(pairs[pk][48], "reason_node")
        b = pooled_H(pairs[pk][96], "reason_node")
        n_up += int(b > a)
        print("  %-8s N=48 %.4f  N=96 %.4f  %s"
              % (pk, a, b, "up" if b > a else "DOWN"))
    print("  rose in %d of %d pairs  %s"
          % (n_up, len([p for p in pairs if 48 in pairs[p] and 96 in pairs[p]]),
             "FALSIFIED" if n_up == 0 else ("HIT" if n_up == 3 else "SPLIT")))

    # ---------------- calibration: is the collapse onto the RIGHT symbol? ---
    print("\nCALIBRATION on reason_node (H vs NLL; NLL >> H = confidently "
          "wrong)")
    print("  %-14s %8s %8s %9s %9s" % ("cell", "H", "NLL", "NLL-H", "top1"))
    for c in cells:
        num_n = den = num_p = 0.0
        for r in rhos:
            k = (c, r, "reason_node")
            if k not in ix:
                continue
            n = int(ix[k]["n_tokens"])
            num_n += fnum(ix[k], "mean_nll") * n
            num_p += fnum(ix[k], "mean_top1") * n
            den += n
        h = pooled_H(c, "reason_node")
        print("  %-14s %8.4f %8.4f %+9.4f %9.4f"
              % (c, h, num_n / den, num_n / den - h, num_p / den))
    return 0


if __name__ == "__main__":
    sys.exit(main())
