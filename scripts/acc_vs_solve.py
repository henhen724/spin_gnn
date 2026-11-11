"""Is next-token accuracy a sufficient statistic for direct deduction?

Stage C's first four cells raised the question. `full` at d_model 128 trains to
train NLL 1.62 -- essentially where `no_mlp` at d_model 256 sits (1.63) -- and
yet scores mean_solve 0.027 against no_mlp's 0.304, a factor of eleven. Train
NLL plainly does not order these models, which matters because the whole grid's
"did not train at this budget" hedge is pinned to a train-NLL threshold
(PLATEAU = 1.64, finding 44). Validation TOKEN ACCURACY does order them
(0.459 vs 0.465-0.483 vs 0.494 for the full d256 hub), which raises the sharper
question this script exists to answer:

    once you know a cell's token accuracy, does knowing its ARCHITECTURE tell
    you anything more about whether it solves?

The two answers lead opposite ways. If architecture adds nothing, then every
cut in the grid -- the MLP cuts, the head cut, the width cut -- acts ONLY by
making the model fit next tokens slightly worse, there is no "deduction
circuit" that the MLP supplies and the minimal-architecture question collapses
into "what is the cheapest architecture that reaches token accuracy X", on a
metric that never censors and is far less seed-noisy than rho_50. If instead
some cut sits BELOW the curve, that cut damages something solve-specific: its
token errors are structured rather than merely more frequent, and it is the
interesting one.

METHOD. The reference curve is the DEPTH LADDER ALONE -- `full`, d256, 4 heads,
L = 1..12, 21 cells. It is the only axis that varies one knob over a wide span
of accuracy while leaving the architecture intact, so it measures the
accuracy-to-solve mapping of a healthy model and nothing else. Every other cell
(the arch axis, the head axis, stage C's width axis) is then HELD OUT and asked
whether it lands on that curve.

The fit is isotonic (pool-adjacent-violators) rather than a polynomial or a
logistic: the only thing assumed is that more accurate models do not solve less
often, which is the premise of using accuracy as a covariate at all. Assuming a
shape would decide the question by choosing a functional form.

The bar is the reference's own LEAVE-ONE-OUT residual: refit on 20 of the 21
depth cells, predict the 21st. A held-out cell counts as off the curve only
when its residual exceeds what a depth cell's own residual would be -- the same
non-circular discipline the rest of the grid scores by, and the reason the
reference is not allowed to include the cells under test.

CPU only, seconds, no checkpoint is loaded: everything here comes from the
rho-curve CSVs and history.json files already on disk.
"""
import argparse
import glob
import json
import os

from scripts.minarch_analysis import (SCAN_DIRS, cell_key, load_all, rung_label,
                                      sort_key)

# Above this predicted solve rate a cell is doing something rather than
# babbling, so agreeing with the curve there is a constraint. 0.10 is the same
# threshold stage A registered for rf_all's redirect, reused rather than
# invented so the two readings of "did it do any of DD" use one number.
FLOOR = 0.10


# A reference cell is a full-width, 4-head, unmodified transformer: rung keys
# of length 3 are exactly the stage A/B defaults (d256, ff1024, no warmup), so
# "len == 3 and nhead 4 and full" IS the depth ladder and nothing else.
def is_reference(rung):
    return len(rung) == 3 and rung[1] == 4 and rung[2] == "full"


def val_accuracy(path):
    """Final-epoch validation token accuracy, or nan if the history is unreadable."""
    try:
        with open(os.path.join(path, "history.json")) as fh:
            h = json.load(fh)
        return float(h[-1]["accuracy"])
    except (OSError, KeyError, IndexError, ValueError):
        return float("nan")


def pava(pts):
    """Isotonic (non-decreasing) fit: [(x, y)] -> [(x, y_fitted)], sorted by x.

    Pool-adjacent-violators. Ties in x are averaged first so that two seeds at
    the same accuracy cannot create a violation out of pure seed noise.
    """
    agg = {}
    for x, y in pts:
        agg.setdefault(x, []).append(y)
    xs = sorted(agg)
    blocks = [[x, sum(agg[x]) / len(agg[x]), len(agg[x])] for x in xs]  # [xmax, mean, weight]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][1] > out[-1][1]:
            x2, m2, w2 = out.pop()
            x1, m1, w1 = out.pop()
            out.append([x2, (m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    fitted, i = [], 0
    for b in out:
        # every original x up to this block's right edge takes the block's value
        while i < len(xs) and xs[i] <= b[0]:
            fitted.append((xs[i], b[1]))
            i += 1
    return fitted


def predict(fit, x):
    """Interpolate an isotonic fit at x -> (prediction, out_of_range, coverage).

    Clamped rather than extrapolated on purpose: a cell more accurate than
    every depth cell, or less accurate than all of them, has no reference to be
    compared against, and inventing one by extending the last segment would
    manufacture exactly the residual the script is trying to measure. Such
    cells are reported as OUT OF RANGE instead of scored.

    COVERAGE is the vertical span of the two reference points the cell falls
    between, and it is the thing that keeps "on the curve" from being vacuous.
    The depth ladder's accuracies are badly clustered -- three L=1 cells near
    0.33, everything else between 0.46 and 0.50, nothing in between -- so a
    cell landing in the empty stretch is bracketed by points 0.13 apart in
    accuracy and its prediction is an interpolation across a void. Where the
    bracket is TIGHT the prediction is pinned by real neighbours and a small
    residual is evidence; where it is WIDE the curve could have been drawn
    almost anywhere and a small residual mostly says both numbers are near the
    floor. Reported per cell so the two cases are never pooled.
    """
    xs = [p[0] for p in fit]
    ys = [p[1] for p in fit]
    if x <= xs[0]:
        return ys[0], x < xs[0], 0.0
    if x >= xs[-1]:
        return ys[-1], x > xs[-1], 0.0
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]) if xs[i] > xs[i - 1] else 0.0
            return ys[i - 1] + t * (ys[i] - ys[i - 1]), False, ys[i] - ys[i - 1]
    return ys[-1], False, 0.0


def collect(scan_dirs):
    """One record per cell: rung, seed, token accuracy, mean solve rate."""
    rows = load_all(scan_dirs)
    cells = {}
    for r in rows:
        cells.setdefault(cell_key(r), []).append(r)
    out = []
    # sort_key, not the keys themselves: a cell key with a string tail ("T24",
    # "k2") meets the SEED of the same rung's shorter key at that position, and
    # int < str raises. See minarch_analysis.sort_key.
    for key, rs in sorted(cells.items(), key=lambda kv: sort_key(kv[0])):
        rung, seed = key[:-1], key[-1]
        acc = val_accuracy(rs[0]["path"])
        out.append({
            "rung": rung, "seed": seed, "acc": acc,
            "mean_solve": sum(x["solve_rate"] for x in rs) / len(rs),
            "best_solve": max(x["solve_rate"] for x in rs),
            "n_trainable": int(rs[0].get("n_trainable_params") or rs[0]["n_params"]),
            "label": rung_label(*rung).strip(),
            "bins": len(rs),
        })
    return out


def rms(v):
    return (sum(x * x for x in v) / len(v)) ** 0.5 if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-dir", nargs="+", default=SCAN_DIRS)
    ap.add_argument("--plot", default="", help="write a PDF here (optional)")
    a = ap.parse_args()

    cells = [c for c in collect(a.scan_dir) if c["acc"] == c["acc"]]
    ref = [c for c in cells if is_reference(c["rung"])]
    held = [c for c in cells if not is_reference(c["rung"])]
    if len(ref) < 4:
        raise SystemExit(f"only {len(ref)} reference (depth-ladder) cells found; "
                         "nothing to fit a curve against")

    print(f"{len(cells)} cells with a readable history: "
          f"{len(ref)} reference (full d256 h4 depth ladder), {len(held)} held out")

    pts = [(c["acc"], c["mean_solve"]) for c in ref]
    fit = pava(pts)
    print(f"\n=== the reference curve: solve rate as a function of token accuracy ===")
    print("  isotonic fit over the depth ladder, one row per distinct accuracy")
    for x, y in fit:
        who = ", ".join(sorted({c["label"] for c in ref if c["acc"] == x}))
        print(f"    acc {x:.4f}  ->  mean_solve {y:.4f}    ({who})")
    lo, hi = fit[0][0], fit[-1][0]
    span = (fit[-1][1] - fit[0][1]) / (hi - lo) if hi > lo else float("nan")
    print(f"  over accuracy {lo:.4f} to {hi:.4f} the curve rises "
          f"{fit[0][1]:.4f} -> {fit[-1][1]:.4f}, i.e. {span:.1f} points of solve rate "
          f"per point of token accuracy")

    # The bar: refit on 20 of 21 and predict the 21st.
    loo = []
    for i, c in enumerate(ref):
        sub = pava([(d["acc"], d["mean_solve"]) for j, d in enumerate(ref) if j != i])
        p, _oor, _cov = predict(sub, c["acc"])
        loo.append(c["mean_solve"] - p)
    bar = rms(loo)
    print(f"\n=== the bar: leave-one-out residual of the reference itself ===")
    print(f"  RMS {bar:.4f}, max |residual| {max(abs(v) for v in loo):.4f} over "
          f"{len(loo)} depth cells")
    print("  a held-out cell is OFF THE CURVE only when |residual| > this RMS: that is what a "
          "depth cell's own distance from the curve looks like, so anything smaller is noise.")

    print(f"\n=== held-out cells: does architecture say anything accuracy did not? ===")
    print("  'cover' is the vertical span of the two depth cells this one falls between: "
          "small = the\n  curve is pinned there by real neighbours, so a small residual is "
          "evidence; LARGE (> the bar)\n  = the bracket is wide and the agreement mostly says "
          "both numbers are near the floor.")
    print(f"  {'rung':<26s} {'seed':>4s} {'acc':>7s} {'solve':>7s} {'pred':>7s} "
          f"{'resid':>8s} {'cover':>7s}  verdict")
    by_rung, weak, sharp = {}, {}, []
    for c in sorted(held, key=lambda c: -c["acc"]):
        p, oor, cov = predict(fit, c["acc"])
        res = c["mean_solve"] - p
        if oor:
            v = "OUT OF RANGE (no depth cell at this accuracy)"
        elif abs(res) <= bar:
            v = "on the curve" + ("  [weakly determined]" if cov > bar else "")
        else:
            v = ("BELOW the curve" if res < 0 else "ABOVE the curve") + \
                ("  [weakly determined]" if cov > bar else "")
        print(f"  {c['label']:<26s} {c['seed']:>4d} {c['acc']:>7.4f} {c['mean_solve']:>7.4f} "
              f"{p:>7.4f} {res:>+8.4f} {cov:>7.4f}  {v}")
        if not oor:
            (weak if cov > bar else by_rung).setdefault(c["rung"], []).append(res)
            if cov <= bar and p > FLOOR:
                sharp.append((c, p, res))

    print(f"\n=== per rung, mean residual against the bar {bar:.4f} ===")
    verdicts = []
    for title, group in (("WELL-DETERMINED -- bracketed by depth cells closer together than the "
                          "bar, so these rows carry the result", by_rung),
                         ("WEAKLY DETERMINED -- they land in the empty stretch of the ladder, "
                          "where the curve is flat and near zero; agreement here is consistency, "
                          "not evidence", weak)):
        if not group:
            continue
        print(f"  -- {title}")
        for rung, rs in sorted(group.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            m = sum(rs) / len(rs)
            tag = ("EXPLAINED by token accuracy" if abs(m) <= bar else
                   "BELOW: token errors are structured, not just more frequent" if m < 0 else
                   "ABOVE: solves better than its token accuracy predicts")
            print(f"     {rung_label(*rung).strip():<26s} n={len(rs)}  mean residual {m:+.4f}"
                  f"   {tag}")
            if group is by_rung:
                verdicts.append((rung, m, abs(m) <= bar))

    n_exp = sum(1 for _, _, ok in verdicts if ok)
    n_cells = sum(len(v) for v in by_rung.values())
    print(f"\n>>> {n_exp}/{len(verdicts)} WELL-DETERMINED held-out rungs ({n_cells} cells) sit on "
          f"the depth ladder's own accuracy-to-solve curve.")
    # Most well-determined cells sit where the curve is flat AND near zero, so
    # "on the curve" there only says both numbers are small -- true, but it
    # would also be true of a model that had learned nothing. The cells that
    # RULE SOMETHING OUT are the ones the curve sends well above the floor,
    # because there a cut that broke deduction specifically would have to miss.
    if sharp:
        print(f"    {len(sharp)} of those {n_cells} are DISCRIMINATING -- the curve predicts "
              f"solve > {FLOOR:.2f} for them, so landing on it is a real constraint and not "
              f"two numbers agreeing near zero:")
        for c, p, res in sorted(sharp, key=lambda t: -t[1]):
            print(f"      {c['label']:<24s} s{c['seed']}  acc {c['acc']:.4f}  "
                  f"predicted {p:.4f}, observed {c['mean_solve']:.4f}  ({res:+.4f})")
        print(f"    Worst miss among them: {max(abs(r) for _, _, r in sharp):.4f}, "
              f"against the bar {bar:.4f}.")
    if n_exp == len(verdicts) and verdicts:
        print("    Then TOKEN ACCURACY IS A SUFFICIENT STATISTIC over this grid: no cut buys or "
              "destroys anything solve-specific, each one only moves the model along a curve the "
              "depth ladder already traces. The minimal-architecture question becomes 'what is "
              "the cheapest architecture reaching accuracy X', readable on a metric that never "
              "censors and does not turn on one unlucky seed. The claim is CORRELATIONAL -- "
              "nothing here was intervened on -- and it is a statement about this grid's range, "
              "which no cell except rf_all leaves.")
        print("    Stated at its weakest and most defensible: across these architectures model "
              "quality is ONE-DIMENSIONAL. Removing machinery does not open a second, "
              "deduction-specific way to fail -- it moves a cell along the single axis the "
              "depth ladder already spans. Note this is NOT the trivial compounding law: "
              "independent token errors at accuracy 0.48 over a trace this long would put solve "
              "rate at essentially zero everywhere, so the errors these models make are heavily "
              "concentrated on positions the solve metric forgives.")
    elif verdicts:
        off = [rung_label(*r).strip() for r, m, ok in verdicts if not ok]
        print(f"    The exceptions are the interesting cells: {'; '.join(off)}. For these the "
              "architecture matters beyond how well it fits next tokens.")

    if a.plot:
        _plot(a.plot, ref, held, fit, bar)
        print(f"\nwrote {a.plot}")


def _plot(path, ref, held, fit, bar):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot([p[0] for p in fit], [p[1] for p in fit], "-", color="0.35", lw=2,
            label="isotonic fit to the depth ladder", zorder=2)
    ax.fill_between([p[0] for p in fit], [p[1] - bar for p in fit], [p[1] + bar for p in fit],
                    color="0.80", alpha=0.6, zorder=1,
                    label=f"the depth ladder's own LOO band (+-{bar:.3f})")
    ax.scatter([c["acc"] for c in ref], [c["mean_solve"] for c in ref], s=70, marker="o",
               facecolor="white", edgecolor="black", lw=1.4, zorder=4,
               label="reference: full d256, L=1..12")
    marks = {}
    for c in held:
        marks.setdefault(c["label"], []).append(c)
    for i, (lab, cs) in enumerate(sorted(marks.items())):
        ax.scatter([c["acc"] for c in cs], [c["mean_solve"] for c in cs], s=60, zorder=5,
                   marker="sv^Dp*X<>h"[i % 10], label=lab)
    ax.set_xlabel("final validation token accuracy")
    ax.set_ylabel("mean solve rate over the 14 rho bins")
    ax.set_title("Does architecture predict deduction beyond token accuracy?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
