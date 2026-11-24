"""Figures for stage RULE5 -- the selection-free j = 0 slice.

Same shards, same loader and the same categorical rho axis as the RULE2, RULE3
and RULE4 figures, so the four decks read identically. Nothing is trained and
nothing is submitted.

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

from stagerule3_fit import sequences  # noqa: E402
from stagerule5_fit import OUT, ADDED, slice0, rates  # noqa: E402

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


def rho_c_line(ax, rhos, y, va="top"):
    cut = min(i for i, r in enumerate(rhos) if r in ADDED_SET) - 0.5
    ax.axvline(cut, color=MUTED, lw=1.0, ls="--", zorder=0)
    ax.text(cut - 0.12, y, r"$\rho_c \approx 0.465$", ha="right", va=va,
            fontsize=9, color=MUTED)


def fig_all(v, rhos):
    """THE RESULT. With a denominator the model does not choose, the
    first-emission error rate rises monotonically. No hump."""
    y = [v[r]["h_all"] for r in rhos]
    x = list(range(len(rhos)))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, y, "-o", ms=6, color=SERIES[0], lw=1.7, zorder=3)
    axis(ax, rhos)
    ax.set_ylabel(r"$h^{\mathrm{all}}(0)$   (no censoring, no selection)")
    ax.set_ylim(0, max(y) * 1.22)
    ax.set_xlim(-0.5, len(rhos) + 0.9)
    rho_c_line(ax, rhos, max(y) * 1.18)
    # The curve rises steeply out of the first point, so the left-hand label
    # goes BELOW it; above collides with the line.
    label_line(ax, x[0], y[0], "0.020", color=SERIES[0], dx=2, dy=-17)
    label_line(ax, x[-1], y[-1], "0.139", color=SERIES[0], dx=6, dy=0)
    save(fig, "all.png")


def fig_arms(v, rhos):
    """The gap that every earlier stage conditioned away: 151x to 451x between
    a forced cell and an unforced one. Log y, same units on both series, so
    this is not a dual axis."""
    hu = [v[r]["h_unf"] for r in rhos]
    hr = [v[r]["h_rule"] for r in rhos]
    x = list(range(len(rhos)))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.75, color=MUTED, lw=1.0, ls=":", zorder=0)
    ax.text(len(rhos) - 0.4, 0.78, "blind colour guess, $1-1/q$",
            fontsize=8.5, color=MUTED, ha="right")
    ax.plot(x, hu, "-o", ms=5, color=SERIES[1], lw=1.7, zorder=3)
    ax.plot(x, hr, "-o", ms=5, color=SERIES[0], lw=1.7, zorder=3)
    axis(ax, rhos)
    ax.set_yscale("log")
    ax.set_ylabel("error rate of the first emitted colour")
    ax.set_xlim(-0.5, len(rhos) + 2.9)
    ticks = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 7.5e-1]
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0.001", "0.003", "0.01", "0.03", "0.1", "0.3",
                        "0.75"], fontsize=8)
    ax.set_yticks([], minor=True)
    ax.set_ylim(8e-4, 1.1)
    label_line(ax, x[-1], hu[-1], "unforced cell", color=SERIES[1], dx=6, dy=0)
    label_line(ax, x[-1], hr[-1], "forced cell", color=SERIES[0], dx=6, dy=0)
    save(fig, "arms.png")


def fig_factors(v, rhos):
    """The decomposition. Everything normalised to its value at rho = 0.95, so
    the three curves are directly comparable: the rise in h_all is the rise in
    the unforced SHARE, not in the unforced error rate."""
    base = v[rhos[0]]
    y_all = [v[r]["h_all"] / base["h_all"] for r in rhos]
    y_shr = [(1 - v[r]["s_F"]) / (1 - base["s_F"]) for r in rhos]
    y_unf = [v[r]["h_unf"] / base["h_unf"] for r in rhos]
    x = list(range(len(rhos)))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(1.0, color=MUTED, lw=1.0, ls="--", zorder=0)
    ax.plot(x, y_shr, "-o", ms=5, color=SERIES[1], lw=1.7, zorder=3)
    ax.plot(x, y_all, "-o", ms=5, color=SERIES[0], lw=1.7, zorder=4)
    ax.plot(x, y_unf, "-o", ms=5, color=SERIES[2], lw=1.7, zorder=3)
    axis(ax, rhos)
    ax.set_ylabel(r"relative to $\rho = 0.95$")
    ax.set_ylim(0, max(max(y_shr), max(y_all)) * 1.18)
    ax.set_xlim(-0.5, len(rhos) + 3.6)
    # These two nearly coincide -- that IS the claim -- so their labels have
    # to be pulled apart by hand or they stack on top of each other.
    label_line(ax, x[-1], y_shr[-1], r"unforced share $1-s_F$",
               color=SERIES[1], dx=6, dy=16)
    label_line(ax, x[-1], y_all[-1], r"$h^{\mathrm{all}}(0)$",
               color=SERIES[0], dx=6, dy=-18)
    label_line(ax, x[-1], y_unf[-1], r"unforced rate $h^{\mathrm{unf}}(0)$",
               color=SERIES[2], dx=6, dy=0)
    save(fig, "factors.png")


def fig_share(v, rhos):
    """The selection itself: the share of first emissions that name a cell
    one-step propagation does NOT force, which is what h_rule throws away."""
    y = [100.0 * (1 - v[r]["s_F"]) for r in rhos]
    x = list(range(len(rhos)))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x, y, "-o", ms=6, color=SERIES[1], lw=1.7, zorder=3)
    axis(ax, rhos)
    ax.set_ylabel("first emission named an UNFORCED cell  (%)")
    ax.set_ylim(0, max(y) * 1.25)
    ax.set_xlim(-0.5, len(rhos) + 0.9)
    rho_c_line(ax, rhos, max(y) * 1.21)
    label_line(ax, x[0], y[0], "3.7%", color=SERIES[1], dx=2, dy=-17)
    label_line(ax, x[-1], y[-1], "24.1%", color=SERIES[1], dx=6, dy=0)
    save(fig, "share.png")


def fig_contrib(v, rhos):
    """How much of h_all the forced arm actually accounts for: 5.5% at the
    densest bin, 1.1% at the sparsest. Four stages were spent on that term."""
    y = [100.0 * v[r]["s_F"] * v[r]["h_rule"] / v[r]["h_all"] for r in rhos]
    x = list(range(len(rhos)))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x, y, "-o", ms=6, color=SERIES[0], lw=1.7, zorder=3)
    axis(ax, rhos)
    ax.set_ylabel(r"forced arm's share of $h^{\mathrm{all}}(0)$  (%)")
    ax.set_ylim(0, max(y) * 1.3)
    ax.set_xlim(-0.5, len(rhos) + 0.9)
    rho_c_line(ax, rhos, max(y) * 1.26)
    label_line(ax, x[0], y[0], "5.5%", color=SERIES[0], dx=-2, dy=13)
    label_line(ax, x[-1], y[-1], "1.1%", color=SERIES[0], dx=6, dy=0)
    save(fig, "contrib.png")


def main():
    seq, _n, _blank = sequences()
    acc, _pa, _pr = slice0(seq)
    rhos = sorted(acc, reverse=True)
    assert len(rhos) == 14, len(rhos)
    v = {r: rates(acc[r]) for r in rhos}
    fig_all(v, rhos)
    fig_arms(v, rhos)
    fig_factors(v, rhos)
    fig_share(v, rhos)
    fig_contrib(v, rhos)


if __name__ == "__main__":
    main()
