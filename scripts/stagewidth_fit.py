"""Scorer for stage WIDTH.  Committed BEFORE the probe ran; see the sbatch
header for the registered predictions W1-W5 and diagnostics DW1-DW3.

Stdlib only.  argv[1] overrides the scan dir.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

SCAN = "/scratch/users/hshunt/spin_gnn/runs/stagewidth_2026-11-20"
SCALE = "/scratch/users/hshunt/spin_gnn/runs/stagescale_2026-11-20"

PAIRS = ["L08_s0", "L12_s0", "L08_s1"]
NOT_CEILING = ["L08_s0", "L08_s1"]   # the N=48 cells that are NOT at ceiling
W1_LO, W1_HI, W1_FLO, W1_FHI = 1.3, 1.8, 1.1, 2.5
W2_BAR, W2_FALSIFIER = 0.08, 0.03    # drop in acc across w = 1..6 at N=96
W3_FACTOR, W3_FALSIFIER = 2.0, 1.0   # w-spread / j-spread at N=96
W4_BAR, W4_FALSIFIER = 0.05, 0.05    # N gap at matched width
W5_BAR, W5_FALSIFIER = 0.03, 0.08    # N=48 w-spread, scored off-ceiling only
W_RANGE = list(range(1, 7))
J_RANGE = list(range(0, 6))
MIN_TOK = 200
RHO_MAIN = 0.60


def load(scan, pat):
    rows = []
    for path in sorted(glob.glob(os.path.join(scan, pat))):
        with open(path) as fh:
            rows.extend(list(csv.DictReader(fh)))
    return rows


def median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def pooled(rows, key, n_nodes, only_pairs=None):
    """key -> [correct, total], pooled over the matched pairs at one N."""
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        if int(r["n_nodes"]) != n_nodes or not r["pair_key"]:
            continue
        if only_pairs and r["pair_key"] not in only_pairs:
            continue
        for k, (c, n) in json.loads(r[key]).items():
            cell = out[k]
            cell[0] += c
            cell[1] += n
    return out


def curve(pool, keys):
    """accuracy at each key with >= MIN_TOK tokens, as (key, acc, n)."""
    out = []
    for k in keys:
        v = pool.get(str(k))
        if v and v[1] >= MIN_TOK:
            out.append((k, v[0] / v[1], v[1]))
    return out


def main():
    scan = sys.argv[1] if len(sys.argv) > 1 else SCAN
    rows = load(scan, "width_*of*.csv")
    if not rows:
        print("no CSVs in %s" % scan)
        return 1
    print("%d rows, %d runs" % (len(rows), len({r["run_id"] for r in rows})))

    # ---------------- diagnostics -------------------------------------------
    wbad = sum(int(r["w_bad"]) for r in rows)
    print("\nDW1 round walk consumes the trace: %d bad  %s"
          % (wbad, "PASS" if wbad == 0 else "FAIL"))
    mism = 0
    for r in rows:
        n = int(r["n_tokens"])
        for key in ("by_w", "by_j", "by_tj"):
            if sum(v[1] for v in json.loads(r[key]).values()) != n:
                mism += 1
    print("DW2 by_w/by_j/by_tj totals match n_tokens: %d bad  %s"
          % (mism, "PASS" if mism == 0 else "FAIL"))

    # DW3: same instances, same checkpoints as stage SCALE -- the overall
    # reason_node accuracy must agree exactly.  A real cross-stage check.
    sc = {}
    for r in load(SCALE, "scale_*of*.csv"):
        if r["section"] == "reason_node":
            sc[(r["run_id"], round(float(r["rho"]), 4))] = float(r["accuracy"])
    worst, n_cmp = 0.0, 0
    for r in rows:
        k = (r["run_id"], round(float(r["rho"]), 4))
        if k in sc and r["accuracy"]:
            worst = max(worst, abs(float(r["accuracy"]) - sc[k]))
            n_cmp += 1
    print("DW3 agrees with stage SCALE's reason_node accuracy: max %.3e over "
          "%d cells  %s" % (worst, n_cmp, "PASS" if worst < 1e-6 else "FAIL"))

    # ---------------- W1: round width grows with N --------------------------
    print("\nW1 mean round width, N=48 -> N=96, at rho %.2f" % RHO_MAIN)
    ratios = []
    for p in PAIRS:
        v = {}
        for r in rows:
            if r["pair_key"] == p and round(float(r["rho"]), 4) == RHO_MAIN:
                v[int(r["n_nodes"])] = float(r["mean_width"])
        if 48 in v and 96 in v:
            ratios.append(v[96] / v[48])
            print("   %-8s %.3f -> %.3f   ratio %.3f" % (p, v[48], v[96],
                                                        v[96] / v[48]))
    if not ratios:
        # A nan median must never print as "MISS": absent is not negative.
        print("   NO matched pair at this density -- W1 UNSCOREABLE")
    else:
        med = median(ratios)
        fl = sum(1 for x in ratios if x < W1_FLO or x > W1_FHI)
        print("   median ratio %.3f, band [%.1f, %.1f]  %s"
              % (med, W1_LO, W1_HI, "FALSIFIED" if fl else
                 ("HIT" if W1_LO <= med <= W1_HI else "MISS")))

    # ---------------- W2: at N=96 accuracy falls with round width -----------
    print("\nW2 N=96 accuracy vs ROUND WIDTH (pooled over pairs and densities)")
    c96 = curve(pooled(rows, "by_w", 96), W_RANGE)
    for k, a, n in c96:
        print("   w=%-3d %.4f  (%d tokens)" % (k, a, n))
    if len(c96) >= 2:
        drop = c96[0][1] - c96[-1][1]
        inv = sum(1 for i in range(len(c96) - 1) if c96[i + 1][1] > c96[i][1])
        print("   drop across w=%d..%d: %.4f, %d inversion(s)  %s"
              % (c96[0][0], c96[-1][0], drop, inv,
                 "FALSIFIED" if drop < W2_FALSIFIER or inv > 1 else
                 ("HIT" if drop >= W2_BAR and inv <= 1 else "MISS")))

    # ---------------- W3: within-round POSITION does not explain it ---------
    print("\nW3 N=96: spread across WIDTH vs spread across POSITION")
    cj = curve(pooled(rows, "by_j", 96), J_RANGE)
    for k, a, n in cj:
        print("   j=%-3d %.4f  (%d tokens)" % (k, a, n))
    if c96 and cj:
        ws = max(a for _k, a, _n in c96) - min(a for _k, a, _n in c96)
        js = max(a for _k, a, _n in cj) - min(a for _k, a, _n in cj)
        fac = ws / js if js > 0 else float("inf")
        print("   width spread %.4f, position spread %.4f, factor %.2f  %s"
              % (ws, js, fac,
                 "FALSIFIED" if fac <= W3_FALSIFIER else
                 ("HIT" if fac >= W3_FACTOR else "MISS")))

    # ---------------- W4: does matched width close the N gap? ---------------
    print("\nW4 the N gap AT MATCHED ROUND WIDTH")
    p48, p96 = pooled(rows, "by_w", 48), pooled(rows, "by_w", 96)
    gaps = []
    print("   %5s %10s %10s %10s %9s" % ("w", "N=48", "N=96", "gap", "min tok"))
    for w in range(1, 13):
        a, b = p48.get(str(w)), p96.get(str(w))
        if not a or not b or min(a[1], b[1]) < MIN_TOK:
            continue
        g = a[0] / a[1] - b[0] / b[1]
        gaps.append(g)
        print("   %5d %10.4f %10.4f %+10.4f %9d"
              % (w, a[0] / a[1], b[0] / b[1], g, min(a[1], b[1])))
    if gaps:
        med = median(gaps)
        print("   median gap over %d common widths %+.4f  %s"
              % (len(gaps), med,
                 "FALSIFIED (width fully mediates finding 111)"
                 if med < W4_FALSIFIER else "HIT (it does not)"))

    # ---------------- W5: N=48 is flat in width, off-ceiling cells only -----
    print("\nW5 N=48 accuracy vs round width, L08 cells only (L12 is at "
          "ceiling and cannot show structure)")
    c48 = curve(pooled(rows, "by_w", 48, only_pairs=NOT_CEILING), W_RANGE)
    for k, a, n in c48:
        print("   w=%-3d %.4f  (%d tokens)" % (k, a, n))
    if len(c48) < 2:
        print("   fewer than 2 width bands present -- W5 UNSCOREABLE")
    else:
        spread = max(a for _k, a, _n in c48) - min(a for _k, a, _n in c48)
        print("   spread %.4f  %s"
              % (spread, "FALSIFIED" if spread > W5_FALSIFIER else
                 ("HIT" if spread <= W5_BAR else "MISS")))
        if c96:
            s96 = max(a for _k, a, _n in c96) - min(a for _k, a, _n in c96)
            print("   for contrast, N=96 spread %.4f (%.1fx)"
                  % (s96, s96 / spread if spread else float("inf")))

    # ---------------- context: the joint (T, j) table -----------------------
    print("\nJoint (T band | j band) accuracy, pooled over the matched pairs")
    for n_nodes in (48, 96):
        pool = pooled(rows, "by_tj", n_nodes)
        ks = sorted(pool, key=lambda s: (s.split("|")[0], s.split("|")[1]))
        print("   N=%d: %s" % (n_nodes, "  ".join(
            "%s %.3f" % (k, pool[k][0] / pool[k][1])
            for k in ks if pool[k][1] >= MIN_TOK)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
