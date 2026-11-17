#!/usr/bin/env python3
"""Presentation figures for the stage POS deck.

`stagepos_figs.py` draws the ten DIAGNOSTIC figures registered before the job:
seven-panel small multiples, 79-line overlays, a colourbar for validation
accuracy. Those are right for scoring a prediction and wrong for a phone
screen. This script draws the same numbers under the deck rules -- at most
three series, labelled at their right-hand end, one claim per figure, no dual
axis, no legend box -- and nothing here recomputes a verdict.

Reads `A_position_*.csv`, `B_hazard_*.csv` and `B_summary_*.csv` from POS_OUT
and writes to POS_OUT/deckfigs/.

  module purge
  module load devel math viz python/3.12.1 py-numpy/1.26.3_py312 \\
      py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  python3 scripts/stagepos_deckfigs.py
"""

import collections
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, "/home/users/hshunt/spin_gnn/scripts")
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line  # noqa: E402
from stagelogic_fit import read_shards, f, i, b  # noqa: E402

POS = os.environ.get("POS_OUT") or "/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17"
FIGS = os.path.join(POS, "deckfigs")
DPI = 150
N_NODES = 48
SHOW_RHO = 0.60
SIGN_RHOS = [0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
MIN_AT_RISK = 100


def fig(name, w=7.4, h=4.4):
    f_, ax = plt.subplots(figsize=(w, h))
    return f_, ax, os.path.join(FIGS, name)


def save(f_, path):
    f_.tight_layout()
    f_.savefig(path, dpi=DPI)
    plt.close(f_)
    print("  wrote %s" % os.path.basename(path))


def stagger(ax, ends, gap=0.07):
    """Right-hand-end labels, pushed apart so two curves that finish at the
    same height do not print on top of each other.

    `label_line` places text at the data point it is given, which is right
    until two series converge -- and these series converge by construction,
    because the last deductions are easy for every model. `gap` is a fraction
    of the y range.
    """
    lo, hi = ax.get_ylim()
    sep = gap * (hi - lo)
    ends = sorted(ends, key=lambda e: e[1])
    for k in range(1, len(ends)):
        x, y, text, col = ends[k]
        if y - ends[k - 1][1] < sep:
            ends[k] = (x, ends[k - 1][1] + sep, text, col)
    for x, y, text, col in ends:
        label_line(ax, x, y, text, color=col)


def three_runs(summ):
    """Weakest in79 run, the median one, the strongest -- by validation token
    accuracy, which is the axis every other result in this project is ranked
    on. Chosen by rank, never by how the curve came out."""
    runs = {}
    for r in summ:
        if b(r, "in79") and r["run_id"] not in runs:
            runs[r["run_id"]] = (f(r, "val_acc"), r["cell"])
        if not b(r, "in79"):
            continue
    order = sorted(runs.items(), key=lambda kv: kv[1][0])
    picks = [order[0], order[len(order) // 2], order[-1]]
    return [(rid, cell, acc) for rid, (acc, cell) in picks]


def short(cell):
    """`L=16 h=4 no_mlp d192` -> `no_mlp L=16 d192`, which is how every other
    deck in this project names a cell."""
    parts = dict(p.split("=", 1) for p in cell.split() if "=" in p)
    mode = [p for p in cell.split() if "=" not in p]
    name = mode[0] if mode else "?"
    out = "%s L=%s" % (name, parts.get("L", "?"))
    if "d" in parts:
        out += " d%s" % parts["d"]
    return out


# --------------------------------------------------------------------------

def f1_teacher_forced(arows, picks):
    """Teacher-forced NODE accuracy against deduction index, three runs."""
    f_, ax, path = fig("pos_f1_teacher_forced_vs_j.png")
    ends = []
    for (rid, cell, acc), col in zip(picks, SERIES):
        per_j = {}
        for r in arows:
            if (r["run_id"] != rid or r.get("scheme") != "j"
                    or r.get("tok_type") != "node"
                    or abs(f(r, "rho") - SHOW_RHO) > 1e-9):
                continue
            tot = i(r, "n_total")
            if tot >= MIN_AT_RISK:
                per_j[i(r, "index")] = i(r, "n_correct") / tot
        js = sorted(per_j)
        if not js:
            continue
        ax.plot(js, [per_j[j] for j in js], color=col, lw=1.8)
        ends.append((js[-1], per_j[js[-1]],
                     "%s  (acc %.3f)" % (short(cell), acc), col))
    ax.set_ylim(0, 1.05)
    stagger(ax, ends)
    style_axes(ax, xlabel="deduction index $j$ along the reasoning trace",
               ylabel="accuracy of the next node token")
    save(f_, path)


def f2_hazard(hrows, picks):
    """The free-generation hazard against j, same three runs, same density."""
    f_, ax, path = fig("pos_f2_hazard_vs_j.png")
    ends = []
    for (rid, cell, acc), col in zip(picks, SERIES):
        per_j = {}
        for r in hrows:
            if r["run_id"] != rid or abs(f(r, "rho") - SHOW_RHO) > 1e-9:
                continue
            at = i(r, "n_at_risk")
            if at >= MIN_AT_RISK:
                per_j[i(r, "j")] = (i(r, "n_err") + i(r, "n_stop")) / at
        js = sorted(per_j)
        if not js:
            continue
        ax.plot(js, [per_j[j] for j in js], color=col, lw=1.8)
        ends.append((js[-1], per_j[js[-1]], short(cell), col))
    stagger(ax, ends)
    style_axes(ax, xlabel="deduction index $j$ along the reasoning trace",
               ylabel="hazard $h(j)$: fails here given it reached here")
    save(f_, path)


def f3_sign(hrows):
    """How many (run, rho) cells have a RISING hazard. The P1 verdict."""
    f_, ax, path = fig("pos_f3_hazard_trend.png")
    per_cell = collections.defaultdict(dict)
    for r in hrows:
        rho = round(f(r, "rho"), 4)
        if rho not in [round(x, 4) for x in SIGN_RHOS] or not b(r, "in79"):
            continue
        at = i(r, "n_at_risk")
        if at >= MIN_AT_RISK:
            per_cell[(r["run_id"], rho)][i(r, "j")] = (i(r, "n_err") + i(r, "n_stop")) / at
    frac, meds = [], []
    for rho in SIGN_RHOS:
        rs = []
        for (_rid, rr), per_j in per_cell.items():
            if abs(rr - rho) > 1e-9 or len(per_j) < 4:
                continue
            js = sorted(per_j)
            s = stats.spearmanr(js, [per_j[j] for j in js]).statistic
            if s == s:
                rs.append(s)
        frac.append(100.0 * np.mean([x > 0 for x in rs]) if rs else float("nan"))
        meds.append(np.median(rs) if rs else float("nan"))
    x = np.arange(len(SIGN_RHOS))
    ax.axhline(50, color=MUTED, lw=1.2, ls="--")
    label_line(ax, x[-1], 50, "coin flip", color=MUTED, dy=9)
    ax.bar(x, frac, color=SERIES[0], width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["%.2f" % r for r in SIGN_RHOS])
    ax.set_ylim(0, 100)
    style_axes(ax, xlabel="hint density $\\rho$",
               ylabel="% of models whose hazard RISES along the trace")
    save(f_, path)
    return frac, meds


def f4_integrated(hrows):
    """Integrated hazard against n -- the n^2 result, without the colourbar."""
    f_, ax, path = fig("pos_f4_integrated_hazard.png")
    tot = collections.defaultdict(float)
    nblank = {}
    for r in hrows:
        rho = f(r, "rho")
        if rho < 0.50 - 1e-9 or not b(r, "in79"):
            continue
        at = i(r, "n_at_risk")
        if at < 1:
            # The registered P2(b) integrates over EVERY measured bin, not only
            # the ones past P1's >=100 floor: a tail bin carries little weight
            # but dropping it shortens the sum systematically at large n and
            # would quote a different exponent than VERDICT.txt. Checked: with
            # the 100 floor this figure reads 1.787, with the registered rule
            # 2.058, and 2.058 is the number that was registered.
            continue
        key = (r["run_id"], round(rho, 4))
        tot[key] += (i(r, "n_err") + i(r, "n_stop")) / at
        nblank[key] = N_NODES * (1.0 - f(r, "realized_rho"))
    x = np.array([nblank[k] for k in tot])
    y = np.array([tot[k] for k in tot])
    m = (x > 0) & (y > 0)
    ax.scatter(x[m], y[m], s=9, color=SERIES[0], alpha=0.45, rasterized=True)
    lr = stats.linregress(np.log(x[m]), np.log(y[m]))
    xs = np.linspace(x[m].min(), x[m].max(), 60)
    ax.plot(xs, np.exp(lr.intercept) * xs ** lr.slope, color=SERIES[1], lw=2)
    label_line(ax, xs[-1], np.exp(lr.intercept) * xs[-1] ** lr.slope,
               "$n^{%.2f \\pm %.2f}$" % (lr.slope, lr.stderr), color=SERIES[1])
    ref = np.exp(lr.intercept) * xs[0] ** (lr.slope - 1) * xs
    ax.plot(xs, ref, color=MUTED, lw=1.2, ls="--")
    label_line(ax, xs[-1], ref[-1], "slope 1: one failure per deduction",
               color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axes(ax, xlabel="blank cells  $n = 48(1-\\rho_{\\rm real})$",
               ylabel=r"integrated hazard  $\sum_j h(j)$")
    save(f_, path)
    return lr.slope, lr.stderr, int(m.sum()), lr.rvalue ** 2


def f5_agreement(hrows, summ):
    """Survival predicted from the hazard against the measured solve rate."""
    f_, ax, path = fig("pos_f5_survival_vs_solve.png", w=5.4, h=5.0)
    surv = collections.defaultdict(lambda: 1.0)
    for r in hrows:
        at = i(r, "n_at_risk")
        if at:
            surv[(r["run_id"], round(f(r, "rho"), 4))] *= \
                1.0 - (i(r, "n_err") + i(r, "n_stop")) / at
    meas = {(r["run_id"], round(f(r, "rho"), 4)): f(r, "solve_rate") for r in summ}
    keys = [k for k in surv if k in meas]
    x = np.array([surv[k] for k in keys])
    y = np.array([meas[k] for k in keys])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2)
    ax.scatter(x, y, s=9, color=SERIES[0], alpha=0.5, rasterized=True)
    style_axes(ax, xlabel=r"$\prod_j (1 - h(j))$, predicted from the hazard",
               ylabel="solve rate actually measured")
    save(f_, path)
    return float(np.abs(x - y).max()), len(keys)


def f6_round(hrows):
    """Both trend directions at once, over the whole population.

    The per-run version of this plot is unreadable and that is itself a fact:
    a weak model's survivors almost never reach round 1, so its curve is a
    vertical line at x = 0. The population statistic is the honest display --
    one Spearman per (run, rho) cell against each axis, and both distributions
    sit left of zero.
    """
    f_, ax, path = fig("pos_f6_trend_distributions.png")
    # EXACTLY the cell set the registered P4 scores on -- in79, rho in
    # [0.50, 0.75], at least 6 bins past the at-risk floor -- so the DD-round
    # median here reproduces VERDICT.txt's -0.406 rather than being a third
    # variant of the same statistic. Both correlations are then computed on
    # those same cells, which is what makes them comparable on one axis.
    cells = collections.defaultdict(list)
    for r in hrows:
        rho = f(r, "rho")
        if not b(r, "in79") or not (0.4999 <= rho <= 0.7501):
            continue
        at = i(r, "n_at_risk")
        if at < MIN_AT_RISK:
            continue
        cells[(r["run_id"], round(rho, 4))].append(
            (i(r, "j"), (i(r, "n_err") + i(r, "n_stop")) / at,
             f(r, "n_round_sum") / at))
    sj, sr = [], []
    for pts in cells.values():
        if len(pts) < 6:
            continue
        pts.sort()
        h = [p[1] for p in pts]
        a = stats.spearmanr([p[0] for p in pts], h).statistic
        c = stats.spearmanr([p[2] for p in pts], h).statistic
        if a == a:
            sj.append(a)
        if c == c:
            sr.append(c)
    bins = np.linspace(-1, 1, 33)
    for vals, col, name in ((sj, SERIES[0], "against token position $j$"),
                            (sr, SERIES[1], "against DD round")):
        ax.hist(vals, bins=bins, histtype="step", lw=2, color=col,
                weights=np.full(len(vals), 100.0 / len(vals)))
        ax.axvline(np.median(vals), color=col, lw=1, ls=":")
    ax.axvline(0, color=MUTED, lw=1.2)
    top = ax.get_ylim()[1]
    label_line(ax, np.median(sj), top * 0.92,
               "position, median %+.2f" % np.median(sj), color=SERIES[0])
    label_line(ax, np.median(sr), top * 0.72,
               "DD round, median %+.2f" % np.median(sr), color=SERIES[1])
    style_axes(ax, xlabel="Spearman correlation within one (model, density) cell",
               ylabel="% of cells")
    save(f_, path)
    return len(sj), float(np.median(sj)), len(sr), float(np.median(sr))


def main():
    os.makedirs(FIGS, exist_ok=True)
    arows = read_shards(POS, "A_position_*.csv")
    hrows = read_shards(POS, "B_hazard_*.csv")
    summ = read_shards(POS, "B_summary_*.csv")
    print("A %d   hazard %d   summary %d" % (len(arows), len(hrows), len(summ)))
    picks = three_runs(summ)
    print("three runs by validation accuracy rank:")
    for rid, cell, acc in picks:
        print("  %-5s %-32s acc %.4f" % (rid, cell, acc))

    print("figures:")
    f1_teacher_forced(arows, picks)
    f2_hazard(hrows, picks)
    frac, meds = f3_sign(hrows)
    slope, se, npts, r2 = f4_integrated(hrows)
    dmax, ncells = f5_agreement(hrows, summ)
    nj, mj, nr, mr = f6_round(hrows)

    print("\nnumbers the deck quotes, printed so they are never retyped from memory:")
    for rho, fr, md in zip(SIGN_RHOS, frac, meds):
        print("  rho %.2f: %.1f%% of models rise, median Spearman %+.3f" % (rho, fr, md))
    print("  integrated hazard ~ n^(%.3f +- %.3f), R2 %.3f, %d points"
          % (slope, se, r2, npts))
    print("  survival vs solve rate: max |difference| %.4f on %d cells" % (dmax, ncells))
    print("  trend vs position: median %+.3f over %d cells" % (mj, nj))
    print("  trend vs DD round: median %+.3f over %d cells" % (mr, nr))


if __name__ == "__main__":
    main()
