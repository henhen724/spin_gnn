#!/usr/bin/env python3
"""Stage SIB figures for decks/spin-gnn/dropped-seeds.

Reads the same shards the scorer reads, so a figure cannot disagree with
VERDICT.txt. Written before the job landed; no number is hard-coded here except
the gate margins, which come from the read-only audit (finding 135) and not from
this job at all.

  module purge --force; module load devel math viz; module load python/3.12.1
  module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig
  python3 scripts/stagesib_figs.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slides import SERIES, MUTED, style_axes  # noqa: E402
import stagesib_fit as F  # noqa: E402

FIGS = "/home/users/hshunt/slides/decks/spin-gnn/dropped-seeds/figures"
DPI = 150
FLOOR = 3e-6          # where an exact zero is drawn on a log axis
BAR = F.BAR


def short(c):
    """Cell label short enough for a tick."""
    return (c["cell"].replace("h=4 ", "").replace("  ", " ")
            .replace("tied_no_mlp", "tied").replace("ep100", "e100").strip())


def save(fig, name):
    p = os.path.join(FIGS, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_gate_margin():
    """How many of 400 each seed solved at rho=0.50 -- the whole admission margin."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ys = range(len(F.CELLS))
    adm = [c["solve50"] for c in F.CELLS]
    ax.barh(list(ys), adm, height=0.45, color=SERIES[0], zorder=3)
    ax.scatter([0] * (2 * len(F.CELLS)),
               [y + 0.42 for y in ys for _ in (0, 1)],
               s=46, marker="|", color=SERIES[1], zorder=4, linewidths=2)
    for y, c in zip(ys, F.CELLS):
        ax.text(c["solve50"] + 0.4, y, str(c["solve50"]), va="center",
                fontsize=9, color=SERIES[0])
    ax.set_yticks(list(ys))
    ax.set_yticklabels([short(c) for c in F.CELLS], fontsize=9)
    ax.set_xlabel("instances solved of 400 at $\\rho = 0.50$")
    ax.set_xlim(0, 17)
    ax.text(9.0, 0.15, "admitted seed", color=SERIES[0], fontsize=10)
    ax.text(0.6, len(F.CELLS) - 0.55, "both dropped siblings: 0 of 400",
            color=SERIES[1], fontsize=10)
    style_axes(ax)
    save(fig, "gate_margin.png")


def _floored(x):
    return max(x, FLOOR)


def fig_hrule_by_cell(v):
    """Admitted vs its two siblings, per cell, on the rule arm."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    xs = range(len(F.CELLS))
    for x, c in zip(xs, F.CELLS):
        a = v[c["adm"]]["h_rule"]
        ax.scatter([x], [_floored(a)], s=70, marker="o" if a > 0 else "v",
                   facecolor=SERIES[0] if a > 0 else "none",
                   edgecolor=SERIES[0], zorder=4, linewidths=1.6)
        for r, _ in c["sibs"]:
            s = v[r]["h_rule"]
            ax.scatter([x + 0.17], [_floored(s)], s=70,
                       marker="o" if s > 0 else "v",
                       facecolor=SERIES[1] if s > 0 else "none",
                       edgecolor=SERIES[1], zorder=4, linewidths=1.6)
    ax.axhline(BAR, color=MUTED, lw=1.2, ls="--", zorder=2)
    ax.text(len(F.CELLS) - 0.55, BAR * 1.25, "the 0.01 bar", color=MUTED,
            fontsize=9, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 2, 1.3)
    ax.set_xticks([x + 0.085 for x in xs])
    ax.set_xticklabels([short(c) for c in F.CELLS], fontsize=8, rotation=18,
                       ha="right")
    ax.set_ylabel("$h_\\mathrm{rule}$")
    ax.text(0.08, 0.08, "admitted", color=SERIES[0], fontsize=10,
            transform=ax.transAxes)
    ax.text(0.08, 0.02, "gate-dropped siblings", color=SERIES[1], fontsize=10,
            transform=ax.transAxes)
    ax.text(0.98, 0.02, "open $\\triangledown$ = exactly 0, drawn at the floor",
            color=MUTED, fontsize=8, ha="right", transform=ax.transAxes)
    style_axes(ax)
    save(fig, "hrule_by_cell.png")


def fig_pooled_shift(v):
    """Published (admitted seed alone) -> pooled over all three seeds."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    xs = range(len(F.CELLS))
    for x, c in zip(xs, F.CELLS):
        before = v[c["adm"]]["h_rule"]
        _, _, after = F.pooled(c, v)
        ax.annotate("", xy=(x, _floored(after)), xytext=(x, _floored(before)),
                    arrowprops=dict(arrowstyle="-|>", color=SERIES[1], lw=1.6,
                                    shrinkA=2, shrinkB=2))
        ax.scatter([x], [_floored(before)], s=58, facecolor="none",
                   edgecolor=SERIES[0], zorder=4, linewidths=1.6)
        ax.scatter([x], [_floored(after)], s=58, color=SERIES[1], zorder=4)
    ax.axhline(BAR, color=MUTED, lw=1.2, ls="--", zorder=2)
    ax.text(len(F.CELLS) - 0.55, BAR * 1.25, "the 0.01 bar", color=MUTED,
            fontsize=9, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 2, 1.3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([short(c) for c in F.CELLS], fontsize=8, rotation=18,
                       ha="right")
    ax.set_ylabel("$h_\\mathrm{rule}$")
    ax.text(0.08, 0.08, "published, admitted seed alone", color=SERIES[0],
            fontsize=10, transform=ax.transAxes)
    ax.text(0.08, 0.02, "pooled over all three seeds", color=SERIES[1],
            fontsize=10, transform=ax.transAxes)
    style_axes(ax)
    save(fig, "pooled_shift.png")


def fig_nF(v):
    """Forced emissions reached -- the denominator, which censoring shrinks."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = range(len(F.CELLS))
    for x, c in zip(xs, F.CELLS):
        ax.scatter([x], [v[c["adm"]]["F"]], s=70, color=SERIES[0], zorder=4)
        for r, _ in c["sibs"]:
            ax.scatter([x + 0.17], [v[r]["F"]], s=70, color=SERIES[1], zorder=4)
    ax.set_xticks([x + 0.085 for x in xs])
    ax.set_xticklabels([short(c) for c in F.CELLS], fontsize=8, rotation=18,
                       ha="right")
    ax.set_ylabel("forced emissions reached, $n_F$")
    ax.text(0.08, 0.92, "admitted", color=SERIES[0], fontsize=10,
            transform=ax.transAxes)
    ax.text(0.08, 0.86, "gate-dropped siblings", color=SERIES[1], fontsize=10,
            transform=ax.transAxes)
    style_axes(ax)
    save(fig, "nF.png")


def fig_u(vj0):
    """u(0): the fraction of instances whose first move was not forced."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = range(len(F.CELLS))
    for x, c in zip(xs, F.CELLS):
        ax.scatter([x], [vj0[c["adm"]]["u"]], s=70, color=SERIES[0], zorder=4)
        for r, _ in c["sibs"]:
            ax.scatter([x + 0.17], [vj0[r]["u"]], s=70, color=SERIES[1], zorder=4)
    ax.set_xticks([x + 0.085 for x in xs])
    ax.set_xticklabels([short(c) for c in F.CELLS], fontsize=8, rotation=18,
                       ha="right")
    ax.set_ylabel("$u(0) = 1 - s_F(0)$")
    ax.set_ylim(0, 0.62)
    ax.text(0.08, 0.92, "admitted", color=SERIES[0], fontsize=10,
            transform=ax.transAxes)
    ax.text(0.08, 0.86, "gate-dropped siblings", color=SERIES[1], fontsize=10,
            transform=ax.transAxes)
    style_axes(ax)
    save(fig, "u_by_cell.png")


def main():
    os.makedirs(FIGS, exist_ok=True)
    fig_gate_margin()
    v, vj0, _, _ = F.measure()
    missing = [r for r in F.ALL_RUNS if r not in v]
    if missing:
        print("shards missing for %s -- only gate_margin rendered" % sorted(missing))
        return 1
    fig_hrule_by_cell(v)
    fig_pooled_shift(v)
    fig_nF(v)
    fig_u(vj0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
