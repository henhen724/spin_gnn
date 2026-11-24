#!/usr/bin/env python3
"""Figures for stage RULE8 -- the INTERVAL on h_rule for the governing cell.

Every number comes from stagerule8_fit, imported rather than retyped, so a
figure and VERDICT.txt cannot disagree. Nothing is trained and nothing is
submitted.

Run under the FIGURE env:

  module purge --force; module load devel math viz
  module load python/3.12.1
  module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig
"""

import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

from slides import SERIES, MUTED, style_axes, label_line  # noqa: E402

from stagegate_fit import GRID, J2_LABEL                   # noqa: E402
from stagerule_fit import load, per_cell                   # noqa: E402
from stagerule8_fit import (OUT, CELL, Z, T_DF2,           # noqa: E402
                            intervals, jackknife, per_run_counts,
                            poisson_exact, wilson)

FIGS = os.path.join(OUT, "figs")
UNIT = 1e-5          # every axis is in units of 1e-5, so the labels stay short


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def bar(ax, y, lo, hi, color, lw=3.2):
    """One interval as a horizontal bar with end caps."""
    ax.plot([lo / UNIT, hi / UNIT], [y, y], color=color, lw=lw,
            solid_capstyle="butt", rasterized=True)
    for x in (lo, hi):
        ax.plot([x / UNIT, x / UNIT], [y - 0.14, y + 0.14], color=color,
                lw=lw, rasterized=True)


# ------------------------------------------------------------------ the data

def gather():
    cells, _, _, _ = load()
    arch = per_cell(cells, rhos=GRID)
    a14 = next(x for x in arch if x["cell"] == CELL)
    j2 = next(x for x in arch if x["cell"] == J2_LABEL)
    runs = sorted({c["run_id"] for c in cells if c["cell"] == CELL})
    iv = intervals(a14)
    per_run = per_run_counts(cells, runs, GRID)
    # jackknife() returns 5; score() widens it to 7 with the Z bounds, and the
    # figures read the same 7-tuple so they cannot drift from VERDICT.txt.
    full, reps, se, bc, n = jackknife(per_run)
    jk = (full, reps, se, bc, n, full - Z * se, full + Z * se)

    grid = set(round(r, 4) for r in GRID)
    per_rho = collections.defaultdict(lambda: [0.0, 0.0])
    for c in cells:
        if c["cell"] == CELL and c["rho"] in grid:
            per_rho[c["rho"]][0] += c["err_in_F"]
            per_rho[c["rho"]][1] += c["in_F"]
    per_rho = {k: tuple(v) for k, v in per_rho.items()}
    return iv, jk, per_run, per_rho, j2


# ------------------------------------------------------------------- figures

def fig_intervals(iv, jk):
    """The four two-sided intervals on the same two counts, same eval."""
    full, reps, se, bc, n, jk_lo, jk_hi = jk
    rows = [
        ("jackknife, $t$(df=2)", full - T_DF2 * se, full + T_DF2 * se, SERIES[1]),
        ("jackknife, $Z$=1.96", jk_lo, jk_hi, SERIES[0]),
        ("exact Poisson", iv["pois"][0], iv["pois"][1], MUTED),
        ("Wilson (before)", iv["w"][0], iv["w"][1], MUTED),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    for i, (name, lo, hi, color) in enumerate(rows):
        bar(ax, i, lo, hi, color)
        ax.annotate(name, xy=(0, i), xycoords=("axes fraction", "data"),
                    xytext=(6, 11), textcoords="offset points",
                    color=color, fontsize=13, va="bottom", ha="left")
    ax.axvline(0.0, color=MUTED, lw=1.0, ls=":")
    ax.axvline(iv["h"] / UNIT, color="#333333", lw=1.2)
    label_line(ax, iv["h"] / UNIT, len(rows) - 0.42,
               "$h_{\\rm rule}=2.47$", color="#333333", dy=4)
    style_axes(ax, xlabel="$h_{\\rm rule}$   (units of $10^{-5}$)")
    ax.set_yticks([])
    ax.set_ylim(-0.7, len(rows) - 0.05)
    ax.set_xlim(-4.2, 11.0)
    save(fig, "intervals.png")


def fig_replicates(jk, per_run):
    """Each seed dropped whole: three replicates around the full estimate."""
    full, reps, se, bc, n, jk_lo, jk_hi = jk
    xs = list(range(len(reps)))
    ys = [v / UNIT for _, v in reps]
    names = [k for k, _ in reps]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.axhline(full / UNIT, color=MUTED, lw=1.2, ls="--")
    # above the line and at the left tick: below it would land on the r69 point
    label_line(ax, 0.0, full / UNIT, "all three seeds, 2.47", color=MUTED, dy=9)
    cols = [SERIES[1] if per_run[k][0] == 0 else SERIES[0] for k in names]
    for x, y, c in zip(xs, ys, cols):
        ax.plot([x], [y], "o", color=c, ms=11, rasterized=True)
    label_line(ax, xs[-1], ys[-1], "  the clean seed", color=SERIES[1])
    ax.set_xticks(xs)
    ax.set_xticklabels(["drop %s\n%d/%d" % (k, int(per_run[k][0]),
                                            int(per_run[k][1]))
                        for k in names])
    style_axes(ax, xlabel="run deleted",
               ylabel="$h_{\\rm rule}$ on the other two  ($10^{-5}$)")
    ax.set_xlim(-0.5, len(reps) - 0.2)
    ax.set_ylim(0, 4.3)
    save(fig, "replicates.png")


def fig_density(iv, per_rho):
    """The other resampling unit: one density deleted at a time, 14 replicates."""
    f2, r2, se2, bc2, n2 = jackknife(per_rho)
    xs = [k for k, _ in r2]
    ys = [v / UNIT for _, v in r2]
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    ax.axhline(iv["h"] / UNIT, color=MUTED, lw=1.2, ls="--")
    # the curve sits above the line everywhere except the two dips, so the only
    # clear space is under the line at the hard end
    label_line(ax, 0.40, iv["h"] / UNIT, "all 14 densities, 2.47",
               color=MUTED, dy=-17, dx=-6)
    ax.plot(xs, ys, "o-", color=SERIES[0], lw=2.0, ms=7, rasterized=True)
    # the two dips are the claim: they are the densities that carry an error
    lo = sorted(range(len(ys)), key=lambda i: ys[i])[:2]
    for i in lo:
        label_line(ax, xs[i], ys[i], "  $\\rho$=%.2f" % xs[i], color=SERIES[0],
                   dy=-13)
    ax.invert_xaxis()
    style_axes(ax, xlabel="density $\\rho$ deleted",
               ylabel="$h_{\\rm rule}$ on the other 13  ($10^{-5}$)")
    ax.set_ylim(0, 3.4)
    save(fig, "density.png")


def fig_denominator(iv):
    """What tightening costs: the one-sided bound falls only as F grows."""
    k = iv["k"]
    c2 = chi2.ppf(0.95, 2 * (k + 1)) / 2.0
    need = c2 / 1e-5
    xs = [iv["F"] + i * (7.2e5 - iv["F"]) / 240.0 for i in range(241)]
    ys = [c2 / x / UNIT for x in xs]
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    ax.plot([x / 1e5 for x in xs], ys, color=SERIES[0], lw=2.4,
            rasterized=True)
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--")
    # at x >= 2.4 the curve has fallen far enough to cross this label
    label_line(ax, 1.1, 1.0, "a bound of $10^{-5}$", color=MUTED, dy=9)
    ax.plot([iv["F"] / 1e5], [iv["up1"] / UNIT], "o", color=SERIES[1], ms=11,
            rasterized=True)
    label_line(ax, iv["F"] / 1e5, iv["up1"] / UNIT,
               "  today: 3 seeds, bound 7.78", color=SERIES[1])
    ax.plot([need / 1e5], [1.0], "o", color=SERIES[0], ms=11, rasterized=True)
    label_line(ax, need / 1e5, 1.0, "  24 seeds in all", color=SERIES[0],
               dy=-15)
    style_axes(ax, xlabel="forced emissions $n_F$  ($10^5$)",
               ylabel="one-sided 95% upper on $h_{\\rm rule}$  ($10^{-5}$)")
    ax.set_xlim(0.6, 7.9)
    ax.set_ylim(0, 8.6)
    save(fig, "denominator.png")


def fig_compare(iv, j2):
    """The comparison the deck ends on: the two cells are not resolved."""
    k2, F2 = int(j2["err_in_F"]), int(j2["in_F"])
    l2, h2 = wilson(k2, F2)
    # plain text, not mathtext: a backslash-escaped underscore renders literally
    rows = [
        ("tied_no_mlp d320, 613,818 params", iv["h"], iv["w"][0],
         iv["w"][1], SERIES[0]),
        ("no_mlp d192, 2,504,890 params", k2 / F2, l2, h2, SERIES[1]),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    lo_ov, hi_ov = max(rows[0][2], rows[1][2]), min(rows[0][3], rows[1][3])
    ax.axvspan(lo_ov / UNIT, hi_ov / UNIT, color="#e8e5de", zorder=0)
    for i, (name, h, lo, hi, color) in enumerate(rows):
        bar(ax, i, lo, hi, color)
        ax.plot([h / UNIT], [i], "o", color=color, ms=9, rasterized=True)
        ax.annotate(name, xy=(0, i), xycoords=("axes fraction", "data"),
                    xytext=(6, 12), textcoords="offset points",
                    color=color, fontsize=13, va="bottom", ha="left")
    ax.annotate("both Wilson 95% intervals contain this band",
                xy=(6.4, -0.52), xycoords="data",
                color=MUTED, fontsize=12, va="center", ha="center")
    style_axes(ax, xlabel="$h_{\\rm rule}$   (units of $10^{-5}$)")
    ax.set_yticks([])
    ax.set_ylim(-0.8, 1.55)
    ax.set_xlim(0, 10.0)
    save(fig, "compare.png")


def main():
    iv, jk, per_run, per_rho, j2 = gather()
    fig_intervals(iv, jk)
    fig_replicates(jk, per_run)
    fig_density(iv, per_rho)
    fig_denominator(iv)
    fig_compare(iv, j2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
