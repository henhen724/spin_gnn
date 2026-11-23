"""Figures for stage RULE3 -- what censoring removed from h_rule.

Same shards, same loader, same categorical rho axis as the RULE2 figures so
the two decks read identically. Nothing is trained and nothing is submitted.

Run under the FIGURE env:

  module purge --force; module load devel math viz
  module load python/3.12.1
  module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slides import SERIES, MUTED, style_axes, label_line  # noqa: E402

from stagegate_fit import GRID  # noqa: E402
from stagerule_fit import load, line, R1_BAR  # noqa: E402
from stagerule2_fit import by_density  # noqa: E402
from stagerule3_fit import (OUT, ADDED, REF_RHO, sequences,  # noqa: E402
                            censoring, pool, h_of, lam_star)

FIGS = os.path.join(OUT, "figs")
ADDED_SET = set(ADDED)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def axis(ax, rhos):
    style_axes(ax, xlabel=r"hint density $\rho$   (fewer hints $\rightarrow$)")
    ax.set_xticks(list(range(len(rhos))))
    ax.set_xticklabels(["%.2f" % r for r in rhos], fontsize=8)


def fig_survival(acc, rhos):
    """The instrument failing: almost nothing reaches the end at low rho."""
    y = [100.0 * acc[r]["surv"] for r in rhos]
    x = list(range(len(rhos)))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x, y, "-", color=MUTED, lw=1.2, zorder=1)
    for i, r in enumerate(rhos):
        c = SERIES[1] if r in ADDED_SET else SERIES[0]
        ax.plot([i], [y[i]], "o", ms=7 if r in ADDED_SET else 6, color=c,
                zorder=3)
    axis(ax, rhos)
    ax.set_yscale("log")
    ax.set_ylabel("rollouts reaching the last scored index  (%)")
    cut = min(i for i, r in enumerate(rhos) if r in ADDED_SET) - 0.5
    ax.axvline(cut, color=MUTED, lw=1.0, ls="--", zorder=0)
    ax.text(cut - 0.12, 60.0, r"$\rho_c \approx 0.465$", ha="right",
            va="top", fontsize=9, color=MUTED)
    label_line(ax, x[-1], y[-1], "0.7%", color=SERIES[1], dx=6, dy=0)
    label_line(ax, x[0], y[0], "96.9%", color=SERIES[0], dx=-6, dy=-14)
    save(fig, "survival.png")


def fig_scored_vs_lost(acc, rhos):
    """The denominator that was scored, against the one censoring removed."""
    f = [acc[r]["F_obs"] / 1000.0 for r in rhos]
    l = [acc[r]["L"] / 1000.0 for r in rhos]
    x = list(range(len(rhos)))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x, f, "-o", ms=5, color=SERIES[0], lw=1.6, zorder=3)
    ax.plot(x, l, "-o", ms=5, color=SERIES[1], lw=1.6, zorder=3)
    axis(ax, rhos)
    ax.set_ylabel("forced emissions  (thousands)")
    ax.set_ylim(0, max(f) * 1.22)
    label_line(ax, x[-1], f[-1], "scored", color=SERIES[0], dx=6, dy=0)
    label_line(ax, x[-1], l[-1], "removed by\ncensoring", color=SERIES[1],
               dx=6, dy=-6)
    save(fig, "scored_vs_lost.png")


def fig_band(acc, rhos):
    """h(1) against the all-or-nothing worst case. The band straddles the bar
    at every density, which is why the worst case says nothing."""
    lo = [h_of(acc[r], 1.0) for r in rhos]
    hi = [h_of(acc[r], None) for r in rhos]
    x = list(range(len(rhos)))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.fill_between(x, lo, hi, color=SERIES[0], alpha=0.13, zorder=1)
    ax.plot(x, lo, "-o", ms=5, color=SERIES[0], lw=1.6, zorder=3)
    ax.plot(x, hi, "-o", ms=5, color=SERIES[1], lw=1.6, zorder=3)
    ax.axhline(R1_BAR, color=MUTED, lw=1.2, ls="--", zorder=2)
    axis(ax, rhos)
    ax.set_yscale("log")
    ax.set_ylabel(r"$h^{\mathrm{rule}}$")
    ax.text(0.1, R1_BAR * 1.25, "rule bar 0.01", fontsize=9, color=MUTED)
    label_line(ax, x[-1], hi[-1], "every censored\nrollout breaks it",
               color=SERIES[1], dx=6, dy=-6)
    label_line(ax, x[-1], lo[-1], "censoring\nignorable", color=SERIES[0],
               dx=6, dy=-6)
    save(fig, "band.png")


def fig_tipping(acc, rhos, h_ref):
    """How much worse the censored rollouts would have to be. Only defined
    where the bin reads BETTER than the reference."""
    pts = [(i, lam_star(acc[r], h_ref), r) for i, r in enumerate(rhos)]
    pts = [(i, v, r) for i, v, r in pts if v == v]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=MUTED,
            lw=1.2, zorder=2)
    for i, v, r in pts:
        c = SERIES[1] if r in ADDED_SET else SERIES[2]
        ax.plot([i], [v], "o", ms=8, color=c, zorder=3)
    axis(ax, rhos)
    ax.set_ylabel(r"$\lambda^{*}$  (frailty needed to erase the fall)")
    ax.set_ylim(0.0, 1.9)
    ax.set_xlim(-0.5, len(rhos) - 0.5)
    ax.text(0.1, 1.06, r"$\lambda = 1$: censoring ignorable", fontsize=9,
            color=MUTED)
    label_line(ax, pts[-1][0], pts[-1][1], "1.19", color=SERIES[1], dx=6,
               dy=0)
    # The n/a bins carry no point. Say so inside the figure rather than
    # letting the empty left-hand stretch read as a missing measurement.
    ax.text(4.5, 0.18, "undefined where the bin already\nreads worse than "
                       r"$\rho = 0.50$", fontsize=9, color=MUTED,
            ha="center")
    save(fig, "tipping.png")


def fig_offsupport(acc, rhos):
    """POST-HOC. How often an emission on an UNFORCED cell gets the colour
    right, against a blind guess at q = 4."""
    y = []
    for r in rhos:
        a = acc[r]
        unf = a["at_risk"] - a["F_obs"]
        y.append(100.0 * (1.0 - a["C_tot"] / unf))
    x = list(range(len(rhos)))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhline(25.0, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.plot(x, y, "-", color=MUTED, lw=1.2, zorder=2)
    for i, r in enumerate(rhos):
        c = SERIES[1] if r in ADDED_SET else SERIES[0]
        ax.plot([i], [y[i]], "o", ms=7 if r in ADDED_SET else 6, color=c,
                zorder=3)
    axis(ax, rhos)
    ax.set_ylabel("colour correct on an unforced cell  (%)")
    ax.set_ylim(0, 100)
    ax.text(0.1, 28.0, "blind guess at $q=4$:  25%", fontsize=9, color=MUTED)
    label_line(ax, x[0], y[0], "71%", color=SERIES[0], dx=-4, dy=10)
    label_line(ax, x[-1], y[-1], "50%", color=SERIES[1], dx=6, dy=0)
    save(fig, "offsupport.png")


def main():
    seq, _n, _blank = sequences()
    cells, _v, _h, _s = load()
    dens = by_density(cells)
    cen, _viol, _dj = censoring(seq)
    acc = pool(cen)
    assert len(acc) == 14, len(acc)
    h_ref = dens[REF_RHO]["h"]
    rhos = sorted(acc, reverse=True)
    fig_survival(acc, rhos)
    fig_scored_vs_lost(acc, rhos)
    fig_band(acc, rhos)
    fig_tipping(acc, rhos, h_ref)
    fig_offsupport(acc, rhos)


if __name__ == "__main__":
    main()
