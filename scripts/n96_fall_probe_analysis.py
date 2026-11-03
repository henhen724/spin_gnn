"""Score the four pre-registrations of the N=96 fall probe array (50979059).

The pre-registration is the header of
scripts/slurm/sudoku2_n96_fall_probes_2026-11-03.sbatch; this file computes
the expressions it names, and it is committed BEFORE the data exists so that
the statistic cannot be chosen after seeing it (finding 39's lesson).

    P1  the frontier_mask peak sits at the same ABSOLUTE layer index (+/-1)
        across L=8/12/16 -- a fixed-size circuit with the extra layers as
        slack, which is what finding 11 and finding 36 predict. The
        informative alternative is peak_layer/L constant (+/-0.12), i.e. the
        computation stretching to fill the depth it is given.
    P2  the yardsticks, and they come first:
          (a) probe-fit noise  = spread across probe seeds within one cell;
          (b) training-seed noise = |t01-t05| at L=8 and |t00-t04| at L=12.
        A depth effect is reportable only if it clears (b), and (b) must
        clear (a) or the training seeds are not resolved either.
    P3  known_mask must NOT show the same depth pattern, or the pattern is
        bookkeeping rather than deduction.
    P4  every frontier number is read against the label-only N=96-fall
        baseline: frontier is a subset of not-known, so a reader that
        recovers only known_mask scores on frontier for free.

THE STATISTIC. Per (cell, probe seed, target): lift(layer) = balanced
accuracy at that layer minus balanced accuracy at layer 0, the embedding.
Layer 0 is the tokenization, so the lift is what the network computed rather
than what the encoding handed it. peak_lift is the maximum over layers >= 1
and peak_layer is where it occurs. Cross-probe-seed values are averaged for
the cell-level number and their spread (max - min) is yardstick (a).

Partial runs are expected -- the array writes one CSV per probe seed, each
only after its last checkpoint -- so this reports whatever is on disk and
names what is missing.

Usage:  python3 scripts/n96_fall_probe_analysis.py [--probe-dir DIR]
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

PROBE_DEFAULT = os.path.join(
    os.environ.get("SCRATCH", "/scratch/users/hshunt"),
    "spin_gnn", "sudoku_n96_fall_probes_2026-11-03")

# task prefix -> (label, num_layers, n_train, train_seed). The task prefix is
# the parent directory of the checkpoint, which sudoku_probes puts in `run`;
# the checkpoint basename alone is NOT unique across cells.
CELLS = {
    "t00": ("t0 L=12 240k s0", 12, 240000, 0),
    "t01": ("t1 L=8  240k s0", 8, 240000, 0),
    "t02": ("t2 L=16 240k s0", 16, 240000, 0),
    "t03": ("t3 L=12 120k s0", 12, 120000, 0),
    "t04": ("t4 L=12 240k s1", 12, 240000, 1),
    "t05": ("t5 L=8  240k s1", 8, 240000, 1),
}
TARGETS = ("frontier_mask", "known_mask", "next_node", "n_known_nbrs")


def load(probe_dir):
    """rows[(cell, probe_seed, target)][layer] = balanced_accuracy."""
    out = defaultdict(dict)
    paths = sorted(glob.glob(os.path.join(probe_dir, "probes_seed*.csv")))
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                cell = r["run"].split("/")[0][:3]
                if cell not in CELLS:
                    continue
                out[(cell, int(r["seed"]), r["target"])][int(r["layer"])] = \
                    float(r["balanced_accuracy"])
    return out, [os.path.basename(p) for p in paths]


def peak(profile):
    """(peak_lift, peak_layer) over layers >= 1, against the layer-0 control."""
    if not profile or 0 not in profile:
        return None
    base = profile[0]
    deep = [(v - base, k) for k, v in profile.items() if k >= 1]
    return max(deep) if deep else None


def cell_stats(data, cell, target):
    """Probe-seed mean of (peak_lift, peak_layer, layer0) and their spreads."""
    seeds = sorted(s for (c, s, t) in data if c == cell and t == target)
    vals = []
    for s in seeds:
        pk = peak(data[(cell, s, target)])
        if pk is None:
            continue
        vals.append((pk[0], pk[1], data[(cell, s, target)][0]))
    if not vals:
        return None
    n = len(vals)
    mean = tuple(sum(v[i] for v in vals) / n for i in range(3))
    spread = tuple(max(v[i] for v in vals) - min(v[i] for v in vals) for i in range(3))
    return {"n_probe_seeds": n, "peak_lift": mean[0], "peak_layer": mean[1],
            "layer0": mean[2], "spread_lift": spread[0], "spread_layer": spread[1]}


def pair_gap(stats, a, b, key):
    if stats.get(a) and stats.get(b):
        return abs(stats[a][key] - stats[b][key])
    return None


def fmt(x, nd=4):
    return "   n/a" if x is None else f"{x:.{nd}f}"


def report(probe_dir):
    data, files = load(probe_dir)
    print(f"probe dir: {probe_dir}")
    print(f"CSVs found: {', '.join(files) if files else 'NONE'}")
    present = sorted({c for (c, _s, _t) in data})
    missing = [c for c in CELLS if c not in present]
    if missing:
        print(f"MISSING cells: {', '.join(missing)}")
    if not present:
        print("nothing to score yet")
        return

    stats = {t: {c: cell_stats(data, c, t) for c in present} for t in TARGETS}

    print("\n=== per-cell frontier_mask (lift over layer 0 = the embedding) ===")
    print(f"{'cell':<18} {'L':>3} {'layer0':>8} {'peak':>8} {'lift':>8} "
          f"{'@layer':>7} {'frac':>6} {'seedspread(lift/layer)':>24}")
    for c in present:
        s = stats["frontier_mask"][c]
        label, L, _n, _sd = CELLS[c]
        if s is None:
            print(f"{label:<18} {L:>3}  (no rows)")
            continue
        print(f"{label:<18} {L:>3} {s['layer0']:>8.4f} "
              f"{s['layer0'] + s['peak_lift']:>8.4f} {s['peak_lift']:>8.4f} "
              f"{s['peak_layer']:>7.2f} {s['peak_layer'] / L:>6.3f} "
              f"{s['spread_lift']:>11.4f} / {s['spread_layer']:>8.2f}")

    print("\n=== P2(a) probe-fit noise, the floor under everything ===")
    for t in TARGETS:
        ok = [stats[t][c] for c in present if stats[t][c]]
        if ok:
            print(f"  {t:<14} max spread across probe seeds: "
                  f"lift {max(s['spread_lift'] for s in ok):.4f}, "
                  f"layer {max(s['spread_layer'] for s in ok):.2f} "
                  f"({ok[0]['n_probe_seeds']} seeds)")

    print("\n=== P2(b) training-seed noise, the bar a depth claim must clear ===")
    for t in TARGETS:
        s = stats[t]
        l8 = pair_gap(s, "t01", "t05", "peak_lift")
        l12 = pair_gap(s, "t00", "t04", "peak_lift")
        l8L = pair_gap(s, "t01", "t05", "peak_layer")
        l12L = pair_gap(s, "t00", "t04", "peak_layer")
        print(f"  {t:<14} lift  |t01-t05| L=8 {fmt(l8)}   |t00-t04| L=12 {fmt(l12)}")
        print(f"  {'':<14} layer |t01-t05| L=8 {fmt(l8L, 2)}   |t00-t04| L=12 {fmt(l12L, 2)}")
    yard_lift = max([v for v in (pair_gap(stats["frontier_mask"], "t01", "t05", "peak_lift"),
                                 pair_gap(stats["frontier_mask"], "t00", "t04", "peak_lift"))
                     if v is not None], default=None)
    yard_layer = max([v for v in (pair_gap(stats["frontier_mask"], "t01", "t05", "peak_layer"),
                                  pair_gap(stats["frontier_mask"], "t00", "t04", "peak_layer"))
                      if v is not None], default=None)
    print(f"  YARDSTICK IN FORCE (frontier, larger of the two pairs): "
          f"lift {fmt(yard_lift)}, layer {fmt(yard_layer, 2)}")

    print("\n=== P1 depth: does the peak sit at a fixed layer or a fixed fraction? ===")
    for t in ("frontier_mask", "known_mask"):   # known_mask here IS P3
        depth_cells = [("t01", 8), ("t00", 12), ("t02", 16)]
        have = [(c, L) for c, L in depth_cells if stats[t].get(c)]
        if len(have) < 2:
            print(f"  {t}: need at least two of t01/t00/t02; have {len(have)}")
            continue
        lay = [stats[t][c]["peak_layer"] for c, _L in have]
        frac = [stats[t][c]["peak_layer"] / L for c, L in have]
        lifts = [stats[t][c]["peak_lift"] for c, _L in have]
        abs_spread, frac_spread = max(lay) - min(lay), max(frac) - min(frac)
        print(f"  {t}: peak layers {[round(v, 2) for v in lay]} at L={[L for _c, L in have]}, "
              f"fractions {[round(v, 3) for v in frac]}")
        print(f"      absolute spread {abs_spread:.2f} (P1 bar <=1), "
              f"fractional spread {frac_spread:.3f} (alternative bar <=0.12)")
        print(f"      peak lifts {[round(v, 4) for v in lifts]}, "
              f"spread {max(lifts) - min(lifts):.4f}"
              + (f" vs yardstick {yard_lift:.4f}" if yard_lift is not None else ""))
        if abs_spread <= 1:
            # P1 is the no-movement answer, so a wide yardstick does not
            # contradict it -- it limits what the confirmation is worth.
            verdict = "FIXED ABSOLUTE LAYER (P1 as predicted)"
            if yard_layer is not None and yard_layer > 1:
                verdict += (f"  [worth little: the training-seed yardstick is {yard_layer:.2f}"
                            f" layers, wider than P1's own 1-layer bar, so a drifting peak"
                            f" would have looked the same]")
        elif frac_spread <= 0.12:
            verdict = "FIXED FRACTION OF DEPTH (the alternative)"
        else:
            verdict = "NEITHER -- the peak moves, but not with depth"
        if abs_spread > 1 and yard_layer is not None and abs_spread <= yard_layer:
            verdict += "  [and the depth spread does not clear the training-seed yardstick]"
        print(f"      verdict: {verdict}")

    print("\n=== P3 control: is the depth pattern specific to the DD predicate? ===")
    f, k = stats["frontier_mask"], stats["known_mask"]
    for c in present:
        if f.get(c) and k.get(c):
            print(f"  {CELLS[c][0]:<18} frontier lift {f[c]['peak_lift']:+.4f} @L{f[c]['peak_layer']:.1f}   "
                  f"known lift {k[c]['peak_lift']:+.4f} @L{k[c]['peak_layer']:.1f}")

    print("\n=== P4 label-only baseline (a property of the ensemble, no model) ===")
    bpath = os.path.join(probe_dir, "frontier_baseline_n96_fall.json")
    if os.path.exists(bpath):
        b = json.load(open(bpath))
        free = {k: b[k]["balanced_accuracy"] for k in ("not_known", "known_probe", "degree_aug")
                if k in b}
        best = max(free.values())
        print(f"  free scores: " + ", ".join(f"{k} {v:.4f}" for k, v in free.items()))
        print(f"  best free score {best:.4f} -- any frontier peak at or below it means nothing")
        for c in present:
            if f.get(c):
                tot = f[c]["layer0"] + f[c]["peak_lift"]
                print(f"    {CELLS[c][0]:<18} frontier peak {tot:.4f}  "
                      f"{'ABOVE' if tot > best else 'AT OR BELOW'} the free score")
        if "ensemble" in b:
            e = b["ensemble"]
            assert e["n_nodes"] == 96 and e["fall"], f"baseline is not the N=96 fall ensemble: {e}"
    else:
        print(f"  MISSING {os.path.basename(bpath)} (task 0 writes it); frontier numbers "
              f"cannot be interpreted without it")

    print("\n=== secondary: data axis, t03 (120k) vs t00 (240k) at L=12 ===")
    d = pair_gap(stats["frontier_mask"], "t00", "t03", "peak_lift")
    if d is None:
        print("  need both t00 and t03")
    else:
        print(f"  |t00-t03| frontier peak lift {d:.4f}"
              + (f" vs training-seed yardstick {yard_lift:.4f} -> "
                 + ("RESOLVED" if d > yard_lift else "unreadable, as in rho_50")
                 if yard_lift is not None else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default=PROBE_DEFAULT)
    report(ap.parse_args().probe_dir)


if __name__ == "__main__":
    main()
