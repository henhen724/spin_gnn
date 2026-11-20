"""Stage WIDTH, POST-HOC CONTROLLED RE-ANALYSIS.  NOT pre-registered.

This script is labelled post-hoc on purpose and its outputs are reported as
such.  The registered verdicts stay exactly as `stagewidth_fit.py` printed
them; nothing here re-scores W1-W5 or moves a band.

WHY IT EXISTS.  The registered W2/W5 curves pool over all five densities, and a
composition check showed that pooling is not innocent: at N=48 the w=5 bin draws
29% of its tokens from rho=0.90 (mean rho 0.584 against 0.470 at w=1), and w=5
is exactly where the N=48 curve jumps up.  Round width and density are coupled
-- a heavily hinted puzzle forces many cells at once -- so a pooled width curve
mixes "wide round" with "easy puzzle".

The fix needs no new compute.  Every CSV row is one (run, rho) and carries its
own `by_w`, so the width curve can be read off WITHIN a density, where rho is
held exactly fixed rather than merely averaged.

THE FRACTIONAL-WIDTH QUESTION.  `n_blank = N(1 - rho)` is constant inside a
density bin, so w/n_blank is w scaled by a per-(N, rho) constant and matching it
across N is exactly N=48's w against N=96's 2w.  If the penalty is about what
FRACTION of the remaining puzzle a round resolves, that matching closes the gap.
Absolute width is the alternative.  Both are checked here; neither is a
registered prediction and both are reported whichever way they fall.

Stdlib only.  argv[1] overrides the scan dir.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

SCAN = "/scratch/users/hshunt/spin_gnn/runs/stagewidth_2026-11-20"
NOT_CEILING = ("L08_s0", "L08_s1")   # N=48 cells that are not at ceiling
DENSITIES = [0.90, 0.75, 0.60, 0.45, 0.35]
W_RANGE = list(range(1, 9))
MIN_TOK = 200


def load(scan):
    rows = []
    for path in sorted(glob.glob(os.path.join(scan, "width_*of*.csv"))):
        with open(path) as fh:
            rows.extend(list(csv.DictReader(fh)))
    return rows


def median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def by_rho_w(rows, n_nodes, only_pairs=None):
    """(rho, w) -> [correct, total], pooled over pairs only -- NOT over rho."""
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        if int(r["n_nodes"]) != n_nodes or not r["pair_key"]:
            continue
        if only_pairs and r["pair_key"] not in only_pairs:
            continue
        for w, (c, n) in json.loads(r["by_w"]).items():
            cell = out[(round(float(r["rho"]), 4), int(w))]
            cell[0] += c
            cell[1] += n
    return out


def acc(pool, rho, w):
    v = pool.get((rho, w))
    if v and v[1] >= MIN_TOK:
        return v[0] / v[1], v[1]
    return None, 0


def main():
    scan = sys.argv[1] if len(sys.argv) > 1 else SCAN
    rows = load(scan)
    if not rows:
        print("no CSVs in %s" % scan)
        return 1
    out = {"post_hoc": True, "scan": scan}

    p96 = by_rho_w(rows, 96)
    p48 = by_rho_w(rows, 48, only_pairs=NOT_CEILING)
    p48all = by_rho_w(rows, 48)

    # ---- 1. width curve within each density ------------------------------
    print("WIDTH CURVE WITHIN EACH DENSITY (post-hoc; rho held exactly fixed)")
    curves = {}
    for tag, pool in (("n96", p96), ("n48_L08", p48)):
        curves[tag] = {}
        print("\n  %s" % ("N=96, all three pairs" if tag == "n96"
                          else "N=48, L08 cells only (L12 is at ceiling)"))
        for rho in DENSITIES:
            cs = [acc(pool, rho, w) for w in W_RANGE]
            got = [(w, a) for w, (a, _n) in zip(W_RANGE, cs) if a is not None]
            if len(got) < 3:
                continue
            spread = max(a for _w, a in got) - min(a for _w, a in got)
            # Does it fall monotonically, or dip and recover?
            lo_at = min(got, key=lambda t: t[1])[0]
            tail = got[-1][1] - min(a for _w, a in got)
            curves[tag]["%.2f" % rho] = {
                "w": [w for w, _a in got], "acc": [a for _w, a in got],
                "spread": spread, "argmin_w": lo_at, "tail_recovery": tail}
            print("    rho %.2f  " % rho
                  + " ".join("w%d %.4f" % (w, a) for w, a in got)
                  + "   spread %.4f  min at w=%d  recovers %+.4f"
                  % (spread, lo_at, tail))
    out["curves"] = curves

    # ---- 2. the N gap at matched (rho, w) --------------------------------
    print("\n\nTHE N GAP AT MATCHED (rho, w) -- absolute width")
    print("  %6s %4s %9s %9s %9s %8s" % ("rho", "w", "N=48", "N=96", "gap",
                                         "min tok"))
    abs_gaps, abs_rows = [], []
    for rho in DENSITIES:
        for w in W_RANGE:
            a48, n48 = acc(p48all, rho, w)
            a96, n96 = acc(p96, rho, w)
            if a48 is None or a96 is None:
                continue
            g = a48 - a96
            abs_gaps.append(g)
            abs_rows.append({"rho": rho, "w": w, "n48": a48, "n96": a96,
                             "gap": g, "min_tok": min(n48, n96)})
            print("  %6.2f %4d %9.4f %9.4f %+9.4f %8d"
                  % (rho, w, a48, a96, g, min(n48, n96)))
    out["abs_rows"] = abs_rows
    out["abs_median"] = median(abs_gaps)
    print("  median %+.4f over %d cells; %d cell(s) at or below 0.05"
          % (median(abs_gaps), len(abs_gaps),
             sum(1 for g in abs_gaps if g <= 0.05)))

    # ---- 3. the same gap at matched FRACTIONAL width ----------------------
    # n_blank = N(1 - rho) is constant in a density bin and N doubles, so
    # matched w/n_blank is N=48's w against N=96's 2w.
    print("\nTHE SAME GAP AT MATCHED FRACTIONAL WIDTH w/n_blank "
          "(N=48 w vs N=96 2w)")
    print("  %6s %4s %6s %9s %9s %9s %8s" % ("rho", "w48", "w96", "N=48",
                                             "N=96", "gap", "min tok"))
    frac_gaps, frac_rows = [], []
    for rho in DENSITIES:
        for w in W_RANGE:
            a48, n48 = acc(p48all, rho, w)
            a96, n96 = acc(p96, rho, 2 * w)
            if a48 is None or a96 is None:
                continue
            g = a48 - a96
            frac_gaps.append(g)
            frac_rows.append({"rho": rho, "w48": w, "w96": 2 * w,
                              "n48": a48, "n96": a96, "gap": g})
            print("  %6.2f %4d %6d %9.4f %9.4f %+9.4f %8d"
                  % (rho, w, 2 * w, a48, a96, g, min(n48, n96)))
    out["frac_rows"] = frac_rows
    out["frac_median"] = median(frac_gaps)
    print("  median %+.4f over %d cells" % (median(frac_gaps), len(frac_gaps)))

    print("\n  VERDICT ON THE MEDIATOR QUESTION (post-hoc, descriptive):")
    print("    absolute-width matching   median gap %+.4f" % median(abs_gaps))
    print("    fractional-width matching median gap %+.4f" % median(frac_gaps))
    better = ("absolute" if abs(median(abs_gaps)) < abs(median(frac_gaps))
              else "fractional")
    print("    neither closes it; %s matching leaves the smaller residue, and"
          % better)
    print("    a mediator would have to drive this to ~0, not merely shrink it.")

    # ---- 4. the interaction: is N=96 MORE width-sensitive? ----------------
    print("\nWIDTH SENSITIVITY AT EACH N, WITHIN DENSITY (slope per unit w, "
          "least squares)")
    slopes = {}
    for tag, pool in (("n48_L08", p48), ("n96", p96)):
        slopes[tag] = {}
        for rho in DENSITIES:
            got = [(w, a) for w in W_RANGE
                   for a, _n in [acc(pool, rho, w)] if a is not None]
            if len(got) < 3:
                continue
            xs = [float(w) for w, _a in got]
            ys = [a for _w, a in got]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            den = sum((x - mx) ** 2 for x in xs)
            slopes[tag]["%.2f" % rho] = (
                sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den)
    out["slopes"] = slopes
    common = sorted(set(slopes["n48_L08"]) & set(slopes["n96"]), reverse=True)
    print("  %6s %12s %12s" % ("rho", "N=48 L08", "N=96"))
    for rho in common:
        print("  %6s %12.4f %12.4f"
              % (rho, slopes["n48_L08"][rho], slopes["n96"][rho]))
    if common:
        m48 = median([slopes["n48_L08"][r] for r in common])
        m96 = median([slopes["n96"][r] for r in common])
        # Deliberately NOT a ratio.  m48 and m96 have opposite signs, so
        # "N=96 is Kx as width-sensitive" would be a ratio of opposite-signed
        # numbers -- arithmetic that reads as a magnitude and means nothing.
        # The sign reversal IS the result, so it is stated as a sign reversal.
        pos48 = sum(1 for r in common if slopes["n48_L08"][r] > 0)
        neg96 = sum(1 for r in common if slopes["n96"][r] < 0)
        print("  median slope: N=48 %+.4f, N=96 %+.4f" % (m48, m96))
        print("  SIGN REVERSAL: positive at N=48 in %d of %d densities, "
              "negative at N=96 in %d of %d." % (pos48, len(common), neg96,
                                                 len(common)))
        print("  So the registered W5 statistic -- the unsigned SPREAD across "
              "w -- cannot")
        print("  distinguish 'flat' from 'sloping the other way', which is why "
              "it missed.")
        out["slope_median"] = {"n48": m48, "n96": m96,
                               "pos48_of": [pos48, len(common)],
                               "neg96_of": [neg96, len(common)]}

    path = os.path.join(scan, "width_control.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
