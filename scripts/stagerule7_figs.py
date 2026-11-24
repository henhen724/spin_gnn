"""Figures for stage RULE7 -- where on the density grid the stopping condition
survives.

Every number here comes from stagerule7_fit, imported rather than retyped, so a
figure and the verdict cannot disagree. That defect was real in RULE6: two
figures each carried their own rounded copy of the clump boundary and moved a
run across it.

Nothing is trained and nothing is submitted.

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

from stagerule6_fit import (cells_j0, clump_split,          # noqa: E402
                            uniform_by_rho)
from stagerule7_fit import (OUT, RHOS, N_INST, E1_BAR, E3_RHO,  # noqa: E402
                            MIN_SIDE, per_run_dens, normalise, by_rho,
                            split_vals, jaccard)

FIGS = os.path.join(OUT, "figs")


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def label_right(ax, x, y, text, color=None, dx=-6, dy=0):
    """`label_line` with the text growing leftwards, for a point at the right
    edge where a left-aligned label would run off the canvas."""
    ax.annotate(text, xy=(x, y), xycoords="data", xytext=(dx, dy),
                textcoords="offset points", color=color or MUTED,
                fontsize=13, va="center", ha="right", annotation_clip=False)


def short(cell):
    return cell.replace("h=4 ", "").strip()


def xaxis(ax):
    """The density axis, categorical and evenly spaced, as in every deck on
    this line -- rho is a designed grid, not a sampled variable, and the four
    bins below rho_c are not evenly spaced in rho."""
    ax.set_xticks(range(len(RHOS)))
    ax.set_xticklabels(["%.2f" % r for r in RHOS], fontsize=10)
    ax.set_xlabel(r"hint density $\rho$  (0.33 is the floor)")


# --------------------------------------------------------------------------

def fig_chance(unif):
    """What u would be with no model in it: pick any blank cell."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    y = [unif[r][2] for r in RHOS]
    x = list(range(len(RHOS)))
    ax.plot(x, y, color=SERIES[0], lw=2.4, marker="o", ms=5,
            rasterized=True)
    label_line(ax, x[0], y[0], "0.0492", color=SERIES[0], dx=6, dy=-14)
    label_right(ax, x[-1], y[-1], "0.8095", color=SERIES[0], dx=-6, dy=10)
    ax.set_ylabel(r"$u_{\mathrm{unif}}(\rho)$")
    ax.set_ylim(0.0, 0.92)
    xaxis(ax)
    style_axes(ax)
    ax.text(0.03, 0.93, "3 blank cells of which 2.85 are forced",
            transform=ax.transAxes, color=MUTED, fontsize=12, va="top")
    ax.text(0.97, 0.30, "32.0 blank cells\nof which 6.10 are forced",
            transform=ax.transAxes, color=MUTED, fontsize=12, va="top",
            ha="right")
    save(fig, "chance.png")


def fig_clumps(u_lo, u_hi, unif):
    """The whole stage in one page: both clumps' raw u against chance."""
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = list(range(len(RHOS)))
    ch = [unif[r][2] for r in RHOS]
    hi = [u_hi[r] for r in RHOS]
    lo = [u_lo[r] for r in RHOS]
    ax.plot(x, ch, color=MUTED, lw=2.0, ls="--", rasterized=True)
    ax.plot(x, hi, color=SERIES[1], lw=2.4, marker="o", ms=4, rasterized=True)
    ax.plot(x, lo, color=SERIES[2], lw=2.4, marker="o", ms=4, rasterized=True)
    label_right(ax, x[-1], ch[-1], "pick any blank cell", color=MUTED,
                dx=-8, dy=8)
    label_right(ax, x[-1], hi[-1], "58 runs", color=SERIES[1], dx=-8, dy=12)
    label_right(ax, x[-1], lo[-1], "21 runs", color=SERIES[2], dx=-8, dy=14)
    ax.set_ylabel(r"$u$  (first emitted cell not forced)")
    ax.set_ylim(-0.03, 0.92)
    xaxis(ax)
    style_axes(ax)
    save(fig, "clumps.png")


def fig_lowflat(u_lo):
    """The low clump alone, on its own scale, so 'nearly flat' is visible."""
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = list(range(len(RHOS)))
    y = [u_lo[r] for r in RHOS]
    ax.plot(x, y, color=SERIES[2], lw=2.4, marker="o", ms=5, rasterized=True)
    label_line(ax, x[0], y[0], "0.26%", color=SERIES[2], dx=8, dy=-6)
    label_right(ax, x[-1], y[-1], "0.74%", color=SERIES[2], dx=-8, dy=10)
    ax.set_ylabel(r"$u$ for the 21 low-clump runs")
    ax.set_ylim(0.0, 0.0092)
    ax.set_yticks([0.0, 0.002, 0.004, 0.006, 0.008])
    xaxis(ax)
    style_axes(ax)
    ax.text(0.03, 0.95, "note the scale: the axis tops out at 0.9%,\n"
                        "and chance over this range rises 0.05 to 0.81",
            transform=ax.transAxes, color=MUTED, fontsize=12, va="top")
    save(fig, "lowflat.png")


def fig_normalised(u_lo, u_hi, unif):
    """u divided by chance. 1.0 is a uniform draw over the blank cells."""
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = list(range(len(RHOS)))
    hi = [u_hi[r] / unif[r][2] for r in RHOS]
    lo = [u_lo[r] / unif[r][2] for r in RHOS]
    ax.axhline(1.0, color=MUTED, lw=1.6, ls="--")
    ax.plot(x, hi, color=SERIES[1], lw=2.4, marker="o", ms=4, rasterized=True)
    ax.plot(x, lo, color=SERIES[2], lw=2.4, marker="o", ms=4, rasterized=True)
    label_right(ax, x[-1], hi[-1], "58 runs", color=SERIES[1], dx=-8, dy=12)
    # below the green line is the tick-label row; there is empty space above
    # it all the way to the orange curve, so the label goes up.
    label_right(ax, x[-1], lo[-1], "21 runs", color=SERIES[2], dx=-8, dy=16)
    ax.set_ylabel(r"$\tilde u = u / u_{\mathrm{unif}}(\rho)$")
    ax.set_ylim(-0.04, 1.16)
    xaxis(ax)
    style_axes(ax)
    ax.text(0.03, 0.97, "1.0 = indistinguishable from picking any blank cell",
            transform=ax.transAxes, color=MUTED, fontsize=12, va="top")
    save(fig, "normalised.png")


def fig_jaccard(memb, low):
    """E3: does the clump membership reproduce inside one density?"""
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x, y = [], []
    for i, r in enumerate(RHOS):
        if r in memb:
            x.append(i)
            y.append(jaccard(memb[r], low))
    ax.plot(x, y, color=SERIES[0], lw=2.4, marker="o", ms=6, rasterized=True)
    ax.set_ylabel("Jaccard against the pooled 21-run clump")
    ax.set_ylim(0.0, 1.06)
    xaxis(ax)
    style_axes(ax)
    ax.text(0.03, 0.46, "%.2f at the three easiest bins is %d runs of %d,\n"
                        "not a different set of runs"
            % (min(y), round(min(y) * len(low)), len(low)),
            transform=ax.transAxes, color=MUTED, fontsize=12, va="top")
    # label the ends with their OWN values: the left end is not 1.00.
    label_line(ax, x[0], y[0], "%.2f" % y[0], color=SERIES[0], dx=8, dy=-14)
    label_right(ax, x[-1], y[-1], "%.2f" % y[-1], color=SERIES[0],
                dx=-8, dy=-14)
    save(fig, "jaccard.png")


def fig_gap(cells):
    """The gap itself, density by density: it never closes."""
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x, a_s, b_s = [], [], []
    for i, r in enumerate(RHOS):
        at = sorted(a["u"] for (run, rr), a in cells.items() if rr == r)
        sp = split_vals(at)
        if sp is None:
            continue
        _gap, a_b, b_b, _k = sp
        x.append(i)
        a_s.append(a_b)
        b_s.append(b_b)
    ax.fill_between(x, a_s, b_s, color=MUTED, alpha=0.22, lw=0,
                    rasterized=True)
    ax.plot(x, a_s, color=SERIES[2], lw=2.4, marker="o", ms=4, rasterized=True)
    ax.plot(x, b_s, color=SERIES[1], lw=2.4, marker="o", ms=4, rasterized=True)
    label_right(ax, x[-1], b_s[-1], "top of the gap", color=SERIES[1],
                dx=-8, dy=12)
    # above the lower boundary is the band's own curve; the label goes below
    # it, where there is nothing, rather than across four of its markers.
    label_right(ax, x[-1], a_s[-1], "bottom of the gap", color=SERIES[2],
                dx=-8, dy=-20)
    ax.set_ylabel(r"$u$ at the clump boundary")
    ax.set_ylim(-0.02, 0.34)
    xaxis(ax)
    style_axes(ax)
    ax.text(0.03, 0.95, "the shaded band is empty: no run sits in it,\n"
                        "at any density",
            transform=ax.transAxes, color=MUTED, fontsize=12, va="top")
    save(fig, "gap.png")


def fig_resolution(unif):
    """Why E1's uniform bar was the wrong shape."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = list(range(len(RHOS)))
    y = [E1_BAR / (1.0 / (N_INST * unif[r][2])) for r in RHOS]
    ax.plot(x, y, color=SERIES[0], lw=2.4, marker="o", ms=5, rasterized=True)
    # the curve rises through anything placed up-and-right of the first point
    # and still crossed the label at dy=-24, so both end values live in the
    # caption and the curve carries no label of its own.
    ax.set_ylabel(r"rollouts of 400 needed to reach $\tilde u = 0.50$")
    ax.set_ylim(0, 178)
    xaxis(ax)
    style_axes(ax)
    ax.text(0.03, 0.95, "one registered bar, sixteen-fold different "
                        "resolution:\n%.1f rollouts at the left end against "
                        "%.1f at the right,\nand E1 failed in the leftmost "
                        "column, on 12 rollouts" % (y[0], y[-1]),
            transform=ax.transAxes, color=MUTED, fontsize=12, va="top")
    save(fig, "resolution.png")


def fig_e1(low, ut):
    """Per-run worst normalised bin, the 21 low-clump runs, against the bar."""
    per = sorted((max(ut[(run, r)] for r in RHOS), run) for run in low)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y = list(range(len(per)))
    v = [p[0] for p in per]
    ax.axvline(E1_BAR, color=MUTED, lw=1.8, ls="--")
    cols = [SERIES[1] if a >= E1_BAR else SERIES[2] for a in v]
    ax.scatter(v, y, s=46, c=cols, zorder=3, rasterized=True)
    ax.set_yticks([])
    ax.set_ylim(-1.4, len(per) + 0.6)
    ax.set_xlim(-0.035, 0.70)
    ax.set_xlabel(r"worst $\tilde u$ over the 14 densities, one dot per run")
    style_axes(ax)
    label_right(ax, E1_BAR, len(per) - 0.4, "registered bar 0.50", color=MUTED,
                dx=-8, dy=0)
    label_line(ax, v[-1], y[-1], "r50", color=SERIES[1], dx=8, dy=0)
    n_zero = sum(1 for a in v if a == 0.0)
    # right-aligned so the caption ends before the dashed bar rather than
    # running through it, and clear of the column of zeros on the left.
    ax.text(0.69, 0.06, "%d of the %d runs sit at exactly zero;\n"
                        "one rollout of 400 is worth %.4f here"
            % (n_zero, len(per), 1.0 / (N_INST * 0.0492)),
            transform=ax.transAxes, color=MUTED, fontsize=12, va="bottom",
            ha="right")
    save(fig, "e1.png")


def fig_seeds(spanning):
    """Finding 130 per density: seeds of one cell on both sides."""
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    rows = sorted(spanning, key=lambda kv: kv[1][-1] - kv[1][0], reverse=True)
    for i, (cell, vals) in enumerate(rows):
        ax.plot([min(vals), max(vals)], [i, i], color=MUTED, lw=1.4,
                zorder=1)
        ax.scatter(vals, [i] * len(vals), s=52, c=SERIES[0], zorder=3,
                   rasterized=True)
        # two seeds within 0.0005 of each other draw as ONE marker, so a
        # 3-seed row silently reads as 2. Count the coincident ones on the
        # page instead -- the same defect RULE6's answer.png had.
        groups = [[vals[0]]]
        for q in vals[1:]:
            if q - groups[-1][-1] > 0.006:
                groups.append([])
            groups[-1].append(q)
        for g in groups:
            if len(g) > 1:
                # the coincident pair sits at u = 0, hard against the left
                # spine, so a centred label would hang outside the axes.
                ax.annotate(r"$\times %d$ seeds" % len(g),
                            xy=(max(g), i), xycoords="data",
                            xytext=(10, 9), textcoords="offset points",
                            color=MUTED, fontsize=11, ha="left",
                            va="bottom", annotation_clip=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([short(c) for c in [r[0] for r in rows]], fontsize=11)
    ax.set_xlabel(r"pooled $u$, one dot per seed")
    ax.set_xlim(-0.022, 0.30)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    style_axes(ax)
    # bottom-right sits on the last row's markers; the gap between two rows
    # carries nothing but whitespace, so the caption goes there.
    ax.text(0.125, len(rows) / 2.0 - 0.5, "same architecture, same "
            "hyperparameters,\nonly the random seed differs",
            color=MUTED, fontsize=12, va="center", ha="center")
    save(fig, "seeds.png")


def main():
    (meta, cell_of, tot, seeds, by_dens, orphan, seq, n_haz,
     nblank) = cells_j0()
    cells = per_run_dens(seq)
    unif = uniform_by_rho(seq, nblank)
    ut = normalise(cells, unif)
    keys = sorted(cells)

    pr, _g, _gap, a_bound, _b, _k = clump_split(meta, seeds, MIN_SIDE)
    low = {run for v, _c, run in pr if v <= a_bound}
    high = {run for v, _c, run in pr if v > a_bound}
    u_lo = by_rho(keys, cells, low)
    u_hi = by_rho(keys, cells, high)

    memb = {}
    for r in RHOS:
        at = sorted((a["u"], run) for (run, rr), a in cells.items() if rr == r)
        sp = split_vals([v for v, _x in at])
        if sp is not None:
            memb[r] = {run for v, run in at if v <= sp[1]}

    spanning = []
    for c in meta:
        runs = {rn for rn, _r in cells if cell_of[rn] == c}
        if len(runs) < 2:
            continue
        v = sorted(1.0 - sum(cells[(run, r)]["in_F"] for r in RHOS)
                   / sum(cells[(run, r)]["at_risk"] for r in RHOS)
                   for run in runs)
        if v[0] < a_bound <= v[-1]:
            spanning.append((c, v))

    fig_chance(unif)
    fig_clumps(u_lo, u_hi, unif)
    fig_lowflat(u_lo)
    fig_normalised(u_lo, u_hi, unif)
    fig_jaccard(memb, low)
    fig_gap(cells)
    fig_resolution(unif)
    fig_e1(low, ut)
    fig_seeds(spanning)
    n_zero = sum(1 for run in low
                 if max(ut[(run, r)] for r in RHOS) == 0.0)
    print("low-clump runs at exactly zero everywhere:", n_zero,
          "of", len(low))
    print("spanning cells:", len(spanning))


if __name__ == "__main__":
    main()
