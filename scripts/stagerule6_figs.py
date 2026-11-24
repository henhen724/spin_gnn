"""Figures for stage RULE6 -- DD's stopping condition, per architecture.

Same shards and the same loader as the RULE2-RULE5 figures. The per-density
figure reuses the categorical rho axis so it reads identically to those decks;
the rest are per-architecture and per-run, which is new to this stage.

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

from stagerule_fit import smaller_than                     # noqa: E402
from stagerule6_fit import (OUT, RHOS, ANSWER, J2, BROKEN, E3_FACTOR,  # noqa
                            RULE_BAR, cells_j0, row_of, uniform_by_rho,
                            clump_split, straddlers)

FIGS = os.path.join(OUT, "figs")
ADDED_SET = {0.45, 0.4, 0.35, 0.33}


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def label_right(ax, x, y, text, color=None, dx=-6, dy=0):
    """`label_line` with the text growing leftwards. The rightmost point of a
    sorted roster is where the interesting cell sits, and a left-aligned label
    there runs off the canvas and over whatever annotates the top of the axes."""
    ax.annotate(text, xy=(x, y), xycoords="data", xytext=(dx, dy),
                textcoords="offset points", color=color or MUTED,
                fontsize=13, va="center", ha="right", annotation_clip=False)


def short(cell):
    """The cell strings all carry `h=4`, which is constant across the roster
    and only costs width on an axis label."""
    return cell.replace("h=4 ", "").strip()


def fig_chance(seq, nblank):
    """EXPLANATION. What u would be if the first cell were picked uniformly
    among the blanks. Without this the number 0.24 has nothing to mean."""
    by = uniform_by_rho(seq, nblank)
    y = [by[r][2] for r in RHOS]
    x = list(range(len(RHOS)))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, y, "-o", ms=6, color=MUTED, lw=1.7, zorder=3)
    style_axes(ax, xlabel=r"hint density $\rho$   (fewer hints $\rightarrow$)")
    ax.set_xticks(x)
    ax.set_xticklabels(["%.2f" % r for r in RHOS], fontsize=8)
    ax.set_ylabel("chance level for $u$")
    # Room under the axis for the left-hand end label: the curve starts at
    # 0.05, so anything placed below that point lands on the tick labels.
    ax.set_ylim(-0.055, 1.0)
    ax.set_xlim(-0.5, len(RHOS) + 1.4)
    cut = min(i for i, r in enumerate(RHOS) if r in ADDED_SET) - 0.5
    ax.axvline(cut, color=MUTED, lw=1.0, ls="--", zorder=0)
    ax.text(cut - 0.12, 0.96, r"$\rho_c \approx 0.465$", ha="right", va="top",
            fontsize=9, color=MUTED)
    label_line(ax, x[0], y[0], "0.05", color=MUTED, dx=4, dy=-15)
    label_line(ax, x[-1], y[-1], "0.81", color=MUTED, dx=6, dy=0)
    save(fig, "chance.png")


def fig_roster(rows, meta, u_bar):
    """THE RESULT. Every architecture's u, sorted. Three regimes, not a
    continuum: at the forced set, half chance, and chance."""
    order = sorted(meta, key=lambda c: rows[c]["u"])
    y = [rows[c]["u"] for c in order]
    x = list(range(len(order)))
    col = [SERIES[2] if v < 0.01 else
           (SERIES[1] if v >= 0.9 * u_bar else SERIES[0]) for v in y]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhline(u_bar, color=MUTED, lw=1.2, ls="--", zorder=0)
    ax.text(-0.6, u_bar + 0.015,
            "chance: pick any blank cell, $u = %.3f$" % u_bar,
            fontsize=8.5, color=MUTED, ha="left")
    for xi, yi, ci in zip(x, y, col):
        ax.plot([xi, xi], [0, yi], color=ci, lw=1.1, zorder=2, alpha=0.55)
    ax.scatter(x, y, s=34, c=col, zorder=3)
    style_axes(ax, xlabel="the 31 architectures, sorted by $u$")
    ax.set_ylabel(r"$u = 1 - s_F(0)$   (first cell not forced)")
    ax.set_ylim(0, 0.52)
    ax.set_xlim(-1.0, len(order) + 0.5)
    ax.set_xticks([])
    # No leader line to the green cells: at this scale any line from the text
    # down to y ~ 0 reads as one more stem. The colour is the pointer.
    ax.text(-0.6, 0.345, "four cells track\nthe forced set", fontsize=13,
            color=SERIES[2], ha="left", va="top")
    label_right(ax, x[-1], y[-1], short(BROKEN), color=SERIES[1], dx=-9, dy=-2)
    save(fig, "roster.png")


def fig_clumps(rows, meta, seeds, u_bar):
    """The 79 runs are not spread along u -- they sit in two clumps, and the
    gap is wider than either clump."""
    per_run, _gaps, gap, a_b, b_b, k = clump_split(meta, seeds)
    per = [v for v, _c, _r in per_run]
    x = list(range(1, len(per) + 1))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.axhline(u_bar, color=MUTED, lw=1.2, ls="--", zorder=0)
    ax.text(1.0, u_bar + 0.015, "chance, $u = %.3f$" % u_bar,
            fontsize=8.5, color=MUTED, ha="left")
    ax.plot(x[:k], per[:k], "o", ms=5, color=SERIES[2], zorder=3)
    ax.plot(x[k:], per[k:], "o", ms=5, color=SERIES[0], zorder=3)
    ax.axhspan(a_b, b_b, color=MUTED, alpha=0.11, zorder=0)
    style_axes(ax, xlabel="the 79 runs, sorted by $u$")
    ax.set_ylabel(r"$u$ for one run (5,600 first emissions)")
    ax.set_ylim(-0.02, 0.52)
    ax.set_xlim(0, len(per) + 14)
    label_line(ax, x[k - 1], per[k - 1], "%d runs" % k,
               color=SERIES[2], dx=6, dy=-4)
    label_line(ax, x[-1], per[-1], "%d runs" % (len(per) - k),
               color=SERIES[0], dx=6, dy=0)
    ax.text(len(per) + 1, (a_b + b_b) / 2.0, "gap\n%.4f" % gap, fontsize=8.5,
            color=MUTED, ha="left", va="center")
    save(fig, "clumps.png")


def fig_straddle(rows, meta, seeds):
    """Seven cells have seeds on both sides of the gap. Same architecture,
    same hyperparameters, only the seed differs."""
    _pr, _gaps, _gap, a_b, b_b, _k = clump_split(meta, seeds)
    sel = straddlers(meta, rows, seeds, a_b)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axvspan(a_b, b_b, color=MUTED, alpha=0.11, zorder=0)
    for i, (cell, p) in enumerate(sel):
        ax.plot([min(p), max(p)], [i, i], color=MUTED, lw=1.0, zorder=1)
        ax.plot(p, [i] * len(p), "o", ms=6, color=SERIES[0], zorder=3)
    style_axes(ax, xlabel=r"$u$ for each seed")
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([short(c) for c, _p in sel], fontsize=8.5)
    ax.set_xlim(-0.02, 0.33)
    ax.set_ylim(-0.6, len(sel) - 0.35)
    ax.text((a_b + b_b) / 2.0, len(sel) - 0.45, "the gap", fontsize=8.5,
            color=MUTED, ha="center", va="top")
    save(fig, "straddle.png")


def fig_ruledead(rows, meta):
    """The other arm, on the same rollouts: 27 of 31 cells saw zero rule
    errors at j = 0, so the rule arm has almost nothing to rank."""
    order = sorted(meta, key=lambda c: rows[c]["err_in_F"])
    y = [rows[c]["err_in_F"] for c in order]
    x = list(range(len(order)))
    col = [MUTED if v == 0 else SERIES[1] for v in y]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(x, y, color=col, width=0.72, zorder=3)
    style_axes(ax, xlabel="the 31 architectures, sorted by rule errors")
    ax.set_ylabel("rule errors at $j = 0$  (count)")
    ax.set_xticks([])
    ax.set_xlim(-1.0, len(order) + 0.5)
    ax.set_ylim(0, max(y) * 1.22)
    n0 = sum(1 for v in y if v == 0)
    ax.text(n0 / 2.0, max(y) * 0.30, "%d cells: ZERO" % n0, fontsize=10,
            color=MUTED, ha="center")
    label_right(ax, x[-1] - 1.4, max(y) * 1.06,
                "%s\n%.0f errors" % (short(order[-1]), y[-1]),
                color=SERIES[1], dx=0, dy=0)
    save(fig, "ruledead.png")


def fig_smaller(rows, meta, seeds):
    """The headline question. Every object strictly smaller than J2 keeps the
    rule and none keeps the stopping condition."""
    j2 = {**meta[J2], "cell": J2}
    u0 = rows[J2]["u"]
    sel = sorted((rows[c]["u"], c) for c in meta if c != J2
                 and smaller_than({**meta[c], "cell": c}, j2))
    y = [v / u0 for v, _c in sel]
    x = list(range(len(sel)))
    col = [SERIES[1] if rows[c]["h_rule"] >= RULE_BAR else SERIES[0]
           for _v, c in sel]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--", zorder=0)
    ax.axhline(E3_FACTOR, color=MUTED, lw=1.0, ls=":", zorder=0)
    # Both reference lines sit within 0.25 of each other on an axis that runs
    # to ~6, so labelling them in place puts two captions in the same 15 px.
    # They go in the empty top-left instead, where no bar reaches.
    ax.text(-0.6, max(y) * 1.12, "dashed: J2 itself, $1.00\\times$\n"
            "dotted: the registered bar, $%.2f\\times$" % E3_FACTOR,
            fontsize=9, color=MUTED, ha="left", va="top")
    ax.bar(x, y, color=col, width=0.62, zorder=3)
    style_axes(ax)
    ax.set_ylabel(r"$u$ relative to J2")
    ax.set_xticks(x)
    ax.set_xticklabels([short(c) for _v, c in sel], fontsize=7.6, rotation=38,
                       ha="right")
    ax.set_ylim(0, max(y) * 1.16)
    ax.set_xlim(-0.7, len(sel) - 0.3)
    save(fig, "smaller.png")


def fig_answer(rows, meta, seeds, u_bar):
    """The registered head-to-head, per seed, because the pooled ratio is a
    seed-composition effect and the page should show that."""
    pair = [(J2, SERIES[0]), (ANSWER, SERIES[1])]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axvline(u_bar, color=MUTED, lw=1.2, ls="--", zorder=0)
    ax.text(u_bar - 0.006, 1.42, "chance", fontsize=8.5, color=MUTED,
            ha="right", va="top", rotation=90)
    for i, (cell, c) in enumerate(pair):
        p = sorted(1.0 - (a["in_F"] / a["at_risk"])
                   for a in seeds[cell].values())
        ax.plot(p, [i] * len(p), "o", ms=8, color=c, zorder=3)
        ax.plot([rows[cell]["u"]], [i], "|", ms=22, color=c, zorder=4)
        ax.text(0.50, i, "pooled $u = %.4f$" % rows[cell]["u"], fontsize=9,
                color=c, va="center")
        # Near-coincident seeds hide behind one marker -- J2 has two inside
        # 0.0004 of each other -- and a reader counting dots would read two
        # seeds where there are three. Cluster at the marker's own width.
        groups = [[p[0]]]
        for q in p[1:]:
            if q - groups[-1][-1] > 0.006:
                groups.append([])
            groups[-1].append(q)
        for g in groups:
            if len(g) > 1:
                ax.text(sum(g) / len(g), i + 0.15,
                        r"$\times %d$ seeds" % len(g), fontsize=8.5, color=c,
                        ha="center", va="bottom")
    style_axes(ax, xlabel=r"$u$ per seed; the bar is the pooled value")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([short(J2) + "\n(J2)", short(ANSWER) + "\n(answer)"],
                       fontsize=8.5)
    ax.set_xlim(-0.02, 0.66)
    ax.set_ylim(-0.5, 1.5)
    save(fig, "answer.png")


def main():
    (meta, _cell_of, tot, seeds, _by_dens, _orphan, seq, _n,
     nblank) = cells_j0()
    rows = {c: row_of(tot[c]) for c in meta}
    by = uniform_by_rho(seq, nblank)
    u_bar = sum(by[r][2] for r in RHOS) / len(RHOS)
    fig_chance(seq, nblank)
    fig_roster(rows, meta, u_bar)
    fig_clumps(rows, meta, seeds, u_bar)
    fig_straddle(rows, meta, seeds)
    fig_ruledead(rows, meta)
    fig_smaller(rows, meta, seeds)
    fig_answer(rows, meta, seeds, u_bar)


if __name__ == "__main__":
    main()
