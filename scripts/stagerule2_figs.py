"""Figures for stage RULE2 -- h_rule refit over all fourteen densities.

Reads the same F_hazard shards stage RULE read, through the same loader, and
draws the three things the deck claims. Nothing is trained and nothing is
submitted; this is seconds on a login node.

Run under the FIGURE env (matplotlib is imported at module level by
stageexp_fit, which stagerule_fit imports):

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
from stagerule_fit import load, per_cell, R1_BAR  # noqa: E402
from stagerule2_fit import OUT, ADDED, HEADLINE, PARAM_BEST, by_density, RHO_C  # noqa: E402

FIGS = os.path.join(OUT, "figs")
FLOOR = 1e-6          # where a measured zero is drawn, as an OPEN marker
ADDED_SET = set(ADDED)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_density(dens):
    """h_rule per density, pooled over all 79 runs. The four never-fitted bins
    are accented; rho_c marks where the instances leave DD's fast phase."""
    rhos = sorted(dens, reverse=True)
    h = [dens[r]["h"] for r in rhos]
    x = list(range(len(rhos)))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x, h, "-", color=MUTED, lw=1.2, zorder=1)
    fit_i = [i for i, r in enumerate(rhos) if r not in ADDED_SET]
    add_i = [i for i, r in enumerate(rhos) if r in ADDED_SET]
    ax.plot(fit_i, [h[i] for i in fit_i], "o", ms=6, color=SERIES[0], zorder=3)
    ax.plot(add_i, [h[i] for i in add_i], "o", ms=7, color=SERIES[1], zorder=3)

    # rho_c sits between the 0.50 and 0.45 bins on this grid
    cut = min(add_i) - 0.5
    ax.axvline(cut, color=MUTED, lw=1.0, ls="--", zorder=0)
    ax.text(cut - 0.12, max(h) * 0.98, r"$\rho_c \approx 0.465$",
            ha="right", va="top", fontsize=9, color=MUTED)

    style_axes(ax, xlabel=r"hint density $\rho$   (fewer hints $\rightarrow$)",
               ylabel=r"$h^{\mathrm{rule}}$")
    ax.set_xticks(x)
    ax.set_xticklabels(["%.2f" % r for r in rhos], fontsize=8)
    ax.set_ylim(0.0, max(h) * 1.18)
    # Only the added bins are labelled: anchoring a second label anywhere on
    # the ten-density stretch collides with the 0.85-0.70 plateau, and the
    # dashed rho_c line already says where the fitted range stopped.
    label_line(ax, add_i[-1], h[add_i[-1]], "never fitted",
               color=SERIES[1], dx=6, dy=0)
    save(fig, "rule_vs_density.png")


def fig_forced(dens):
    """Forced emissions per density -- the denominator behind the curve."""
    rhos = sorted(dens, reverse=True)
    f = [dens[r]["in_F"] / 1000.0 for r in rhos]
    x = list(range(len(rhos)))
    cols = [SERIES[1] if r in ADDED_SET else SERIES[0] for r in rhos]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x, f, width=0.72, color=cols, zorder=3)
    style_axes(ax, xlabel=r"hint density $\rho$   (fewer hints $\rightarrow$)",
               ylabel="forced emissions scored  (thousands)")
    ax.set_xticks(x)
    ax.set_xticklabels(["%.2f" % r for r in rhos], fontsize=8)
    add = sum(f[i] for i, r in enumerate(rhos) if r in ADDED_SET)
    tot = sum(f)
    label_line(ax, x[-1], f[-1],
               "added bins:\n%.0fk of %.0fk  (%.0f%%)"
               % (add, tot, 100.0 * add / tot),
               color=SERIES[1], dx=-10, dy=58)
    save(fig, "forced_by_density.png")


def fig_before_after(a10, a14):
    """h_rule on the ten fitted densities against h_rule on all fourteen, one
    point per architecture. Measured zeros are drawn open at the floor."""
    m10 = {a["cell"]: a for a in a10}
    pts = []
    for a in a14:
        b = m10[a["cell"]]
        pts.append((b["h_rule"], a["h_rule"], a["cell"]))

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    lo, hi = FLOOR * 0.55, 0.2
    ax.plot([lo, hi], [lo, hi], "-", color=MUTED, lw=1.0, zorder=1)
    ax.axhline(R1_BAR, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.axvline(R1_BAR, color=MUTED, lw=1.0, ls="--", zorder=1)

    for bx, ay, cell in pts:
        openish = (bx <= 0.0) or (ay <= 0.0)
        X = max(bx, FLOOR) if bx > 0 else FLOOR
        Y = max(ay, FLOOR) if ay > 0 else FLOOR
        if cell == HEADLINE:
            col, ms, z = SERIES[1], 11, 6
        elif cell == PARAM_BEST:
            col, ms, z = SERIES[2], 11, 6
        else:
            col, ms, z = SERIES[0], 6, 4
        ax.plot([X], [Y], "o", ms=ms, zorder=z,
                mfc="none" if openish else col, mec=col, mew=1.6)

    style_axes(ax, xlabel=r"$h^{\mathrm{rule}}$, ten fitted densities",
               ylabel=r"$h^{\mathrm{rule}}$, all fourteen",
               logx=True, logy=True)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.text(R1_BAR * 1.15, lo * 1.6, "bar 0.01", fontsize=9, color=MUTED)

    # The exact zeros stack on top of each other at the floor. The plot cannot
    # show how many without fabricating positions, so the counts go in the
    # page body; the figure only says that a floored marker is a measured zero.
    ax.text(lo * 1.6, FLOOR * 1.30, "open = a measured zero, drawn at a floor",
            fontsize=9, color=MUTED)
    both = sum(1 for bx, ay, _ in pts if bx <= 0 and ay <= 0)
    woke = sum(1 for bx, ay, _ in pts if bx <= 0 and ay > 0)
    print("zeros: %d on both, %d only on the ten" % (both, woke))
    label_line(ax, 1e-5, 4e-5, "object order", color=SERIES[1], dx=-14, dy=24)
    label_line(ax, 3e-5, 2e-5, "parameter count", color=SERIES[2],
               dx=10, dy=-14)
    save(fig, "before_after.png")


def main():
    cells, _viol, _n_haz, _n_summ = load()
    a10 = per_cell(cells)
    a14 = per_cell(cells, GRID)
    assert len(a10) == 31 and len(a14) == 31, (len(a10), len(a14))
    dens = by_density(cells)
    assert len(dens) == 14, len(dens)
    fig_density(dens)
    fig_forced(dens)
    fig_before_after(a10, a14)


if __name__ == "__main__":
    main()
