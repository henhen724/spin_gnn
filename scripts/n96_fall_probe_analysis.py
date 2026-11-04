"""Score the pre-registrations of the N=96 fall probe array (50979059).

The pre-registration is the header of
scripts/slurm/sudoku2_n96_fall_probes_2026-11-03.sbatch:

    P1  the frontier_mask peak sits at the same ABSOLUTE layer index (+/-1)
        across L=8/12/16 -- a fixed-size circuit with the extra layers as
        slack, which is what finding 11 and finding 36 predict. The
        informative alternative is peak_layer/L constant (+/-0.12).
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

WHICH COLUMN. For the multilabel targets this scores AUROC, not balanced
accuracy, and the choice decides the headline. frontier_mask is ~10%
positive, and balanced accuracy is read at a fixed 0-logit threshold, so it
ranks readers by how willingly they predict positives rather than by what
they know -- which is exactly what frontier_from_known_baseline.py's
docstring says after its first run put the trivial `not_known` RULE (0.6734)
above a trained probe given the same information (0.5585). Scored on
balanced accuracy every cell here sits below the free score and the run looks
like a null; scored on AUROC, on the same rows, every cell clears every free
score by 0.06-0.10. The threshold-free columns are the ones the baseline
script says to quote, so they are the ones scored here, and the balanced
accuracy reading is printed at the end as the trap it is.

THE STATISTIC. Per (cell, probe seed, target): lift(layer) = metric at that
layer minus metric at layer 0, the embedding. Layer 0 is the tokenization, so
the lift is what the network computed rather than what the encoding handed
it. peak_lift is the maximum over layers >= 1, peak_layer is where it occurs,
and the spread across probe seeds is yardstick (a).

TWO KINDS OF TARGET, and they are not read the same way. The multilabel ones
(frontier_mask, known_mask, remaining_mask) are N independent per-node bits
and carry auroc / average_precision; AP's chance level is the positive rate,
not 0.5. The categorical ones (next_node, next_color) are a k-way choice with
no auroc column at all: fit_linear_probe reports top-1 accuracy in the
balanced_accuracy column and chance is the majority class -- 0.015 for
next_node at N=96. Reading next_node against 0.5 would turn the one channel
with real signal into a null.

Usage:  python3 -m scripts.n96_fall_probe_analysis [--probe-dir DIR]
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

try:                                     # -m scripts.n96_fall_probe_analysis
    from scripts import n96_fallscan_analysis as scan
except ImportError:                      # python3 scripts/n96_fall_probe_analysis.py
    import n96_fallscan_analysis as scan

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
MULTILABEL = ("frontier_mask", "known_mask", "remaining_mask")
CATEGORICAL = ("next_node", "next_color")
TARGETS = MULTILABEL + CATEGORICAL
SEED_PAIRS = (("t01", "t05", 8), ("t00", "t04", 12))
DEPTH_CELLS = (("t01", 8), ("t00", 12), ("t02", 16))

# The column each kind of target is scored on. Categorical probes have no
# auroc column at all (fit_linear_probe writes nan), so "balanced_accuracy"
# there is top-1 accuracy read against majority_accuracy.
PRIMARY = {"multilabel": "auroc", "categorical": "balanced_accuracy"}
METRICS = ("balanced_accuracy", "auroc", "average_precision")


def load(probe_dir):
    """(profiles[(cell, probe_seed, target)][layer][metric], majority, files)."""
    prof = defaultdict(dict)
    majority, kind = {}, {}
    for p in sorted(glob.glob(os.path.join(probe_dir, "probes_seed*.csv"))):
        with open(p) as f:
            for r in csv.DictReader(f):
                cell = r["run"].split("/")[0][:3]
                if cell not in CELLS:
                    continue
                kind[r["target"]] = r["kind"]
                prof[(cell, int(r["seed"]), r["target"])][int(r["layer"])] = {
                    m: float(r[m]) for m in METRICS}
                try:
                    majority[r["target"]] = float(r["majority_accuracy"])
                except ValueError:
                    pass
    files = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(probe_dir, "probes_seed*.csv")))]
    return prof, majority, kind, files


def metric_of(kind, target):
    return PRIMARY[kind.get(target, "multilabel")]


def peak(profile, metric):
    """(peak_lift, peak_layer) over layers >= 1, against the layer-0 control."""
    if not profile or 0 not in profile:
        return None
    base = profile[0][metric]
    deep = [(v[metric] - base, k) for k, v in profile.items() if k >= 1]
    return max(deep) if deep else None


def cell_stats(prof, cell, target, metric):
    """Probe-seed mean of the peak, plus the spreads that are yardstick (a)."""
    vals = []
    for (c, _s, t), p in prof.items():
        if c != cell or t != target:
            continue
        pk = peak(p, metric)
        if pk is not None:
            vals.append((pk[0], pk[1], p[0][metric], p[max(p)][metric]))
    if not vals:
        return None
    n = len(vals)
    mean = [sum(v[i] for v in vals) / n for i in range(4)]
    spread = [max(v[i] for v in vals) - min(v[i] for v in vals) for i in range(4)]
    return {"n_probe_seeds": n, "peak_lift": mean[0], "peak_layer": mean[1],
            "layer0": mean[2], "final": mean[3], "peak_abs": mean[2] + mean[0],
            "spread_lift": spread[0], "spread_layer": spread[1], "spread_final": spread[3]}


def pair_gap(stats, a, b, key):
    if stats.get(a) and stats.get(b):
        return abs(stats[a][key] - stats[b][key])
    return None


def spearman(xs, ys):
    """Rank correlation on a handful of points; ties get the average rank."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def rho50_by_cell(scan_dir):
    """rho_50 and val_nll per cell key, from the COMMITTED scan reader, so the
    two analyses cannot disagree about what the scan measured."""
    out = {}
    rows = scan.load(os.path.join(scan_dir, "rho_curve_*.csv"))
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["num_layers"], r["n_train_instances"], r["seed"])].append(r)
    for c, (_lab, L, n, sd) in CELLS.items():
        rows_c = by_cell.get((L, n, sd))
        if rows_c:
            # rho_50 wants (realized_rho, solve_rate) pairs and returns
            # (value, flag); the flag is 'ok' for all six cells (H1 passed).
            val, flag = scan.rho_50([(r["realized_rho"], r["solve_rate"]) for r in rows_c])
            out[c] = (val, rows_c[0]["val_nll"], flag)
    return out


def fmt(x, nd=4):
    return "   n/a" if x is None else f"{x:.{nd}f}"


def report(probe_dir, scan_dir):
    prof, majority, kind, files = load(probe_dir)
    print(f"probe dir: {probe_dir}")
    print(f"CSVs found: {', '.join(files) if files else 'NONE'}")
    present = [c for c in CELLS if any(k[0] == c for k in prof)]
    missing = [c for c in CELLS if c not in present]
    if missing:
        print(f"MISSING cells: {', '.join(missing)}")
    if not present:
        print("nothing to score yet")
        return
    # Every target on its own primary column, plus frontier on the other two:
    # the AP row is the sanity check on AUROC at a 10%-positive target, and the
    # balanced-accuracy row is the trap, printed last.
    stats = {t: {c: cell_stats(prof, c, t, metric_of(kind, t)) for c in present}
             for t in TARGETS}
    alt = {m: {c: cell_stats(prof, c, "frontier_mask", m) for c in present}
           for m in ("average_precision", "balanced_accuracy")}

    # ---- P4 first: it decides whether any frontier number means anything ----
    print("\n=== P4 label-only baseline -- read this BEFORE the frontier table ===")
    bpath = os.path.join(probe_dir, "frontier_baseline_n96_fall.json")
    free = {}
    if os.path.exists(bpath):
        b = json.load(open(bpath))
        if "ensemble" in b:
            e = b["ensemble"]
            assert e["n_nodes"] == 96 and e["fall"], f"baseline is not N=96 fall: {e}"
        print(f"  {'':>13}{'auroc':>9}{'avg_prec':>10}{'bal_acc':>9}   (no model: DD's own labels)")
        for k in ("positive_rate", "shuffled", "not_known", "known_probe", "degree_aug"):
            if k in b:
                print(f"  {k:>13}{b[k]['auroc']:>9.4f}{b[k]['average_precision']:>10.4f}"
                      f"{b[k]['balanced_accuracy']:>9.4f}")
        for m in ("auroc", "average_precision"):
            free[m] = max(b[k][m] for k in ("not_known", "known_probe", "degree_aug") if k in b)
        print(f"  BEST FREE SCORE  auroc {free['auroc']:.4f}, avg_prec "
              f"{free['average_precision']:.4f} (AP chance = the {b['positive_rate']['average_precision']:.4f} "
              f"positive rate) -- a frontier probe at or below these has shown nothing")
    else:
        print(f"  MISSING {os.path.basename(bpath)}; frontier numbers cannot be interpreted")

    print("\n=== frontier_mask, the DD predicate, on AUROC (threshold-free) ===")
    print(f"{'cell':<18}{'L':>3}{'layer0':>8}{'peak':>8}{'lift':>8}{'@layer':>7}{'frac':>6}"
          f"{'probeseed spread l/L':>22}{'vs free':>10}{'AP peak':>9}{'AP free':>9}")
    for c in present:
        s, (label, L, _n, _sd) = stats["frontier_mask"][c], CELLS[c]
        if s is None:
            print(f"{label:<18}{L:>3}  (no rows)")
            continue
        v = "" if not free else ("ABOVE" if s["peak_abs"] > free["auroc"] else "below")
        ap = alt["average_precision"][c]
        apv = "" if not free else ("ABOVE" if ap["peak_abs"] > free["average_precision"] else "below")
        print(f"{label:<18}{L:>3}{s['layer0']:>8.4f}{s['peak_abs']:>8.4f}{s['peak_lift']:>8.4f}"
              f"{s['peak_layer']:>7.2f}{s['peak_layer'] / L:>6.3f}"
              f"{s['spread_lift']:>12.4f} /{s['spread_layer']:>6.2f}{v:>10}"
              f"{ap['peak_abs']:>9.4f}{apv:>9}")

    print("\n=== every target, absolute: layer 0 -> best deep layer (probe-seed mean) ===")
    print(f"{'cell':<18}{'target':<16}{'column':<10}{'chance':>8}{'layer0':>8}{'best':>8}"
          f"{'@L':>4}{'final':>8}")
    for c in present:
        for t in TARGETS:
            s = stats[t].get(c)
            if s is None:
                continue
            m = metric_of(kind, t)
            ch = 0.5 if m == "auroc" else majority.get(t, float("nan"))
            print(f"{CELLS[c][0]:<18}{t:<16}{m:<10}{ch:>8.4f}{s['layer0']:>8.4f}"
                  f"{s['peak_abs']:>8.4f}{s['peak_layer']:>4.0f}{s['final']:>8.4f}")

    print("\n=== P2(a) probe-fit noise, the floor under everything ===")
    for t in TARGETS:
        ok = [stats[t][c] for c in present if stats[t][c]]
        if ok:
            print(f"  {t:<15} ({metric_of(kind, t):<17}) max spread over "
                  f"{ok[0]['n_probe_seeds']} probe seeds: "
                  f"lift {max(s['spread_lift'] for s in ok):.4f}, "
                  f"peak layer {max(s['spread_layer'] for s in ok):.2f}, "
                  f"final layer {max(s['spread_final'] for s in ok):.4f}")

    print("\n=== P2(b) training-seed noise, the bar a depth claim must clear ===")
    for t in TARGETS:
        s = stats[t]
        line = "  " + f"{t:<15}"
        for a, b2, L in SEED_PAIRS:
            line += (f"L={L:<3} lift {fmt(pair_gap(s, a, b2, 'peak_lift'))} "
                     f"layer {fmt(pair_gap(s, a, b2, 'peak_layer'), 2)} "
                     f"final {fmt(pair_gap(s, a, b2, 'final'))}   ")
        print(line)
    yard = {}
    for t in TARGETS:
        for key in ("peak_lift", "peak_layer", "final"):
            vals = [pair_gap(stats[t], a, b2, key) for a, b2, _L in SEED_PAIRS]
            vals = [v for v in vals if v is not None]
            yard[(t, key)] = max(vals) if vals else None
    print(f"  YARDSTICK IN FORCE (frontier): lift {fmt(yard[('frontier_mask', 'peak_lift')])}, "
          f"peak layer {fmt(yard[('frontier_mask', 'peak_layer')], 2)}")

    print("\n=== P1 depth (frontier), and P3 control (known) ===")
    for t in ("frontier_mask", "known_mask"):
        have = [(c, L) for c, L in DEPTH_CELLS if stats[t].get(c)]
        if len(have) < 2:
            print(f"  {t}: need two of t01/t00/t02; have {len(have)}")
            continue
        lay = [stats[t][c]["peak_layer"] for c, _L in have]
        frac = [stats[t][c]["peak_layer"] / L for c, L in have]
        lifts = [stats[t][c]["peak_lift"] for c, _L in have]
        abs_spread, frac_spread = max(lay) - min(lay), max(frac) - min(frac)
        yl, yf = yard[(t, "peak_layer")], yard[(t, "peak_lift")]
        print(f"  {t} ({metric_of(kind, t)}): peaks at layers {[round(v, 2) for v in lay]} "
              f"for L={[L for _c, L in have]}, fractions {[round(v, 3) for v in frac]}")
        print(f"      absolute spread {abs_spread:.2f} (P1 bar <=1), fractional "
              f"{frac_spread:.3f} (alternative bar <=0.12); peak lifts "
              f"{[round(v, 4) for v in lifts]} spread {max(lifts) - min(lifts):.4f}"
              + (f" vs yardstick {yf:.4f}" if yf is not None else ""))
        if abs_spread <= 1:
            v = "FIXED ABSOLUTE LAYER (P1 as predicted)"
            if yl is not None and yl > 1:
                v += (f"  [worth little: the training seed moves the peak {yl:.2f} layers, "
                      f"wider than P1's own 1-layer bar]")
        elif frac_spread <= 0.12:
            v = "FIXED FRACTION OF DEPTH (the alternative)"
            if all(abs(f - 1.0) < 1e-9 for f in frac):
                v += " -- and the fraction is 1.0: the peak IS the last layer in every cell"
        else:
            v = "NEITHER -- the peak moves, but not with depth"
            if yl is not None and abs_spread <= yl:
                v += "  [and it does not clear the training-seed yardstick]"
        print(f"      verdict: {v}")

    print("\n=== the other channel: next_node (categorical, "
          f"chance {majority.get('next_node', float('nan')):.4f}) ===")
    print(f"{'cell':<18}{'L':>3}{'layer0':>8}{'final':>8}{'best':>8}{'@L':>4}")
    for c in present:
        s = stats["next_node"].get(c)
        if s:
            print(f"{CELLS[c][0]:<18}{CELLS[c][1]:>3}{s['layer0']:>8.4f}{s['final']:>8.4f}"
                  f"{s['peak_abs']:>8.4f}{s['peak_layer']:>4.0f}")
    print("  layer 0 sits at ~0.50 because half the labelled positions are pre_color ones,"
          "\n  where the node id is the token being probed (sudoku_activations."
          "reasoning_positions);\n  the part above 0.50 is the pre_node half, where the "
          "model has to choose.")
    for a, b2, L in SEED_PAIRS:
        print(f"  training-seed gap at L={L}: {fmt(pair_gap(stats['next_node'], a, b2, 'final'))}")
    depth = [(c, L) for c, L in DEPTH_CELLS if stats["next_node"].get(c)]
    for i in range(len(depth) - 1):
        (ca, La), (cb, Lb) = depth[i], depth[i + 1]
        g = abs(stats["next_node"][ca]["final"] - stats["next_node"][cb]["final"])
        y = yard[("next_node", "final")]
        print(f"  depth gap L={La}->{Lb}: {g:.4f}"
              + (f" vs seed yardstick {y:.4f} -> "
                 + ("RESOLVED" if y is not None and g > y else "unreadable")
                 if y is not None else ""))

    # ---- does the probe channel rank the cells the way rho_50 does? ----
    try:
        r50 = rho50_by_cell(scan_dir)
    except Exception as exc:                       # scan dir absent or partial
        print(f"\n(no rho_50 correlation: {exc})")
        r50 = {}
    shared = [c for c in present if c in r50 and stats["next_node"].get(c)]
    if len(shared) >= 4:
        chan = {"next_node final": [stats["next_node"][c]["final"] for c in shared],
                "frontier AUROC peak": [stats["frontier_mask"][c]["peak_abs"] for c in shared]}
        rho = [r50[c][0] for c in shared]
        nll = [r50[c][1] for c in shared]
        print(f"\n=== does the probe rank the cells the way the task does? (n={len(shared)}) ===")
        for c in sorted(shared, key=lambda c: r50[c][0]):
            print(f"  {CELLS[c][0]:<18} rho_50 {r50[c][0]:.4f}  val_nll {r50[c][1]:.4f}  "
                  f"next_node {stats['next_node'][c]['final']:.4f}  "
                  f"frontier_auroc {stats['frontier_mask'][c]['peak_abs']:.4f}")
        for name, xs in chan.items():
            print(f"  Spearman {name:<20} vs rho_50 (lower is better) {spearman(xs, rho):+.3f}"
                  f"   vs val_nll {spearman(xs, nll):+.3f}")

    print("\n=== secondary: data axis, t03 (120k) vs t00 (240k) at L=12 ===")
    for t in ("frontier_mask", "next_node"):
        key = "final" if t in CATEGORICAL else "peak_lift"
        d = pair_gap(stats[t], "t00", "t03", key)
        y = yard[(t, key)]
        print(f"  {t:<15} |t00-t03| {fmt(d)}"
              + (f" vs seed yardstick {y:.4f} -> "
                 + ("RESOLVED" if d is not None and y is not None and d > y else "unreadable")
                 if y is not None else ""))

    # ---- the trap, printed last so it cannot be mistaken for the result ----
    print("\n=== THE SAME ROWS ON BALANCED ACCURACY -- why this column is not quoted ===")
    print(f"{'cell':<18}{'layer0':>8}{'peak':>8}{'@layer':>7}{'vs free 0.6734':>16}")
    for c in present:
        s = alt["balanced_accuracy"][c]
        print(f"{CELLS[c][0]:<18}{s['layer0']:>8.4f}{s['peak_abs']:>8.4f}{s['peak_layer']:>7.2f}"
              f"{'below' if not free else ('ABOVE' if s['peak_abs'] > 0.6734 else 'below'):>16}")
    lay = [alt["balanced_accuracy"][c]["peak_layer"] for c, _L in DEPTH_CELLS
           if alt["balanced_accuracy"].get(c)]
    sl = [alt["balanced_accuracy"][c]["peak_layer"] for c in ("t00", "t04")
          if alt["balanced_accuracy"].get(c)]
    if len(lay) == 3:
        print(f"  Read here the run is a null: every cell below the 0.6734 free score, and the "
              f"peak layer\n  wanders {max(lay) - min(lay):.2f} layers across L=8/12/16 while the "
              f"TRAINING SEED alone moves it {abs(sl[0] - sl[1]):.2f}\n  at fixed L=12. That is a "
              f"flat curve's argmax. The 0-logit threshold on a 10%-positive\n  target is doing the "
              f"work -- the same artifact that put the trivial not_known RULE above a\n  trained "
              f"probe in frontier_from_known_baseline.py. AUROC is the honest column.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default=PROBE_DEFAULT)
    ap.add_argument("--scan-dir", default=scan.SCAN_DEFAULT)
    a = ap.parse_args()
    report(a.probe_dir, a.scan_dir)


if __name__ == "__main__":
    main()
