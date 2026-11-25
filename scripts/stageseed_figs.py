#!/usr/bin/env python3
"""Figures for stage SEED -- the three seeds of `tied_no_mlp L=16 d192`.

Every number is imported from stageseed_fit rather than retyped, so a figure
and VERDICT.txt cannot disagree. Nothing is trained and nothing is submitted.

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

from stageseed_fit import (GRID, OUT, PUB_H, SEEDS, SEED0, SEED1,  # noqa: E402
                           SEED2, blocks, hazard_rows, rates, read_rows,
                           LOGIC, wilson)

FIGS = os.path.join(OUT, "figs")

# The d320 cell stage RULE8 measured, for the closing comparison. Same eval,
# same shards, read from disk rather than retyped.
D320 = ["r69", "r70", "r71"]

LAB = {SEED0: "seed 0  (r68)", SEED1: "seed 1  (r80)", SEED2: "seed 2  (r81)"}
COL = {SEED0: SERIES[1], SEED1: SERIES[0], SEED2: SERIES[0]}


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def bar(ax, y, lo, hi, color, lw=3.2):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt",
            rasterized=True)
    for x in (lo, hi):
        ax.plot([x, x], [y - 0.13, y + 0.13], color=color, lw=lw,
                rasterized=True)


def gather():
    rows = hazard_rows()
    per_run, per_rho = blocks(rows, GRID)
    per_run_j0, _ = blocks(rows, GRID, j0_only=True)
    v = {r: rates(per_run[r]) for r in SEEDS}
    vj0 = {r: rates(per_run_j0[r]) for r in SEEDS}
    return v, vj0, per_rho


def d320_rates():
    rows = read_rows(os.path.join(LOGIC, "F_hazard_all_*of10.csv"))
    rows = [r for r in rows if r["run_id"] in set(D320)]
    per_run, _ = blocks(rows, GRID)
    return {r: rates(per_run[r]) for r in D320}


# ------------------------------------------------------------------ figures

def fig_seeds(v):
    """The headline: one seed at 0.08, two at 0.56, on the same instances."""
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for i, rid in enumerate(SEEDS):
        a = v[rid]
        bar(ax, i, a["h_lo"], a["h_hi"], COL[rid])
        ax.plot([a["h_rule"]], [i], "o", color=COL[rid], ms=10, rasterized=True)
        ax.annotate("%s   %d / %d" % (LAB[rid], a["k"], a["F"]),
                    xy=(0, i), xycoords=("axes fraction", "data"),
                    xytext=(6, 11), textcoords="offset points",
                    color=COL[rid], fontsize=13, va="bottom", ha="left")
    ax.axvline(0.01, color=MUTED, lw=1.2, ls="--")
    # below every bar: at the top this label runs into the seed-2 annotation
    label_line(ax, 0.01, -0.5, "  the 0.01 bar", color=MUTED, dy=2)
    style_axes(ax, xlabel="$h_{\\rm rule}$   (share of forced emissions "
                          "that write the wrong colour)")
    ax.set_yticks([])
    ax.set_ylim(-0.7, 2.75)
    ax.set_xlim(0, 0.63)
    save(fig, "seeds.png")


def fig_u(vj0):
    """The other arm, which moved the other way."""
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for i, rid in enumerate(SEEDS):
        a = vj0[rid]
        bar(ax, i, a["u_lo"], a["u_hi"], COL[rid])
        ax.plot([a["u"]], [i], "o", color=COL[rid], ms=10, rasterized=True)
        ax.annotate("%s   %d / %d" % (LAB[rid], a["u_k"], a["at_risk"]),
                    xy=(0, i), xycoords=("axes fraction", "data"),
                    xytext=(6, 11), textcoords="offset points",
                    color=COL[rid], fontsize=13, va="bottom", ha="left")
    style_axes(ax, xlabel="$u = 1 - s_F(0)$   (share of first emissions "
                          "naming an UNFORCED cell)")
    ax.set_yticks([])
    ax.set_ylim(-0.7, 2.75)
    ax.set_xlim(0, 0.52)
    save(fig, "u_by_seed.png")


def fig_density(per_rho):
    """The gap is not one density: it is every density on the grid."""
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for rid in SEEDS:
        xs, ys = [], []
        for rho in GRID:
            a = per_rho.get((rid, round(rho, 4)))
            if a and a["in_F"] > 0:
                xs.append(rho)
                ys.append(a["err_in_F"] / a["in_F"])
        ax.plot(xs, ys, "o-", color=COL[rid], lw=2.0, ms=5, rasterized=True)
        # seeds 1 and 2 land on each other at every density, so two labels sat
        # on top of one another; one label names the pair.
        text = {SEED0: "  seed 0", SEED1: "  seeds 1 and 2"}.get(rid)
        if text:
            label_line(ax, xs[-1], ys[-1], text, color=COL[rid])
    ax.invert_xaxis()
    style_axes(ax, xlabel="hint density $\\rho$",
               ylabel="$h_{\\rm rule}$ at that density")
    ax.set_ylim(0, 1.0)
    save(fig, "density.png")


def fig_denominator(v):
    """Survivorship: the worse seeds never reach as many forced emissions."""
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    xs = list(range(len(SEEDS)))
    ys = [v[r]["F"] for r in SEEDS]
    for x, rid in zip(xs, SEEDS):
        ax.bar([x], [v[rid]["F"]], width=0.55, color=COL[rid], rasterized=True)
        ax.annotate("%d" % v[rid]["F"], xy=(x, v[rid]["F"]),
                    xytext=(0, 6), textcoords="offset points",
                    color=COL[rid], fontsize=13, ha="center", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([LAB[r].split("  ")[0] for r in SEEDS])
    style_axes(ax, ylabel="forced emissions reached  ($n_F$, 14 densities)")
    ax.set_ylim(0, max(ys) * 1.22)
    save(fig, "denominator.png")


def fig_compare(v, d320):
    """Against the previous best on the same eval: d320's seeds do agree."""
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    # Each row names its own cell. Two groups of bare "seed 0/1/2" needed group
    # annotations, and those landed on the 0.01-bar label and on each other.
    rows = ([("d192  %s" % LAB[r].split("  ")[0], v[r], SERIES[0])
             for r in SEEDS]
            + [("d320  seed %d" % i, d320[r], SERIES[1])
               for i, r in enumerate(D320)])
    for i, (name, a, color) in enumerate(rows):
        y = len(rows) - 1 - i
        lo = max(a["h_lo"], 1e-7)
        bar(ax, y, lo, a["h_hi"], color)
        ax.plot([max(a["h_rule"], 1e-7)], [y], "o", color=color, ms=9,
                rasterized=True)
        ax.annotate(name, xy=(0, y), xycoords=("axes fraction", "data"),
                    xytext=(6, 10), textcoords="offset points",
                    color=color, fontsize=12, va="bottom", ha="left")
    ax.set_xscale("log")
    ax.axvline(0.01, color=MUTED, lw=1.2, ls="--")
    label_line(ax, 0.01, -0.55, "  the 0.01 bar", color=MUTED, dy=2)
    style_axes(ax, xlabel="$h_{\\rm rule}$  (log scale, same 14-density eval)")
    ax.set_yticks([])
    ax.set_ylim(-0.8, 6.0)
    ax.set_xlim(5e-7, 2.0)
    save(fig, "compare.png")


def main():
    v, vj0, per_rho = gather()
    d320 = d320_rates()
    fig_seeds(v)
    fig_u(vj0)
    fig_density(per_rho)
    fig_denominator(v)
    fig_compare(v, d320)
    print("\npublished d192 h_rule for reference: %.5f" % PUB_H)
    for r in D320:
        print("  d320 %s  %d/%d  h_rule %.6e" % (r, d320[r]["k"], d320[r]["F"],
                                                 d320[r]["h_rule"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
