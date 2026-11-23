"""Figures for stage RULE4 -- h_rule on the censoring-free j = 0 slice.

Same shards, same loader, same categorical rho axis as the RULE2 and RULE3
figures so the three decks read identically. Nothing is trained and nothing is
submitted.

Run under the FIGURE env:

  module purge --force; module load devel math viz
  module load python/3.12.1
  module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig
"""

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slides import SERIES, MUTED, style_axes, label_line  # noqa: E402

from stagerule_fit import R1_BAR  # noqa: E402
from stagerule3_fit import sequences  # noqa: E402
from stagerule4_fit import (OUT, ADDED, HIGH_RHOS, REF_RHO,  # noqa: E402
                            slice_at, group, ratio_poisson)

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


def fig_hump(acc, rhos):
    """THE RESULT. h_rule on the one slice with no censoring at all: a hump,
    peaking well above rho_c, falling on both sides."""
    y = [acc[r]["E"] / acc[r]["F"] for r in rhos]
    x = list(range(len(rhos)))
    pk = y.index(max(y))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, y, "-", color=MUTED, lw=1.3, zorder=1)
    for i, r in enumerate(rhos):
        c = SERIES[1] if i == pk else SERIES[0]
        ax.plot([i], [y[i]], "o", ms=9 if i == pk else 6, color=c, zorder=3)
    axis(ax, rhos)
    ax.set_ylabel(r"$h^{\mathrm{rule}}(0)$   (censoring-free)")
    ax.set_ylim(0, max(y) * 1.25)
    rho_c_line(ax, rhos, max(y) * 1.20)
    label_line(ax, x[pk], y[pk], "peak 0.0032", color=SERIES[1], dx=0, dy=12)
    label_line(ax, x[0], y[0], "0.0012", color=SERIES[0], dx=2, dy=-16)
    label_line(ax, x[-1], y[-1], "0.0021", color=SERIES[0], dx=6, dy=0)
    save(fig, "hump.png")


def fig_first_vs_later(acc, tot, rhos):
    """The first deduction carries the rule errors; the tail is clean. Three
    series, fixed order, labelled at the right-hand end."""
    h0 = [acc[r]["E"] / acc[r]["F"] for r in rhos]
    h1, hp = [], []
    for r in rhos:
        F1 = tot[r]["F"] - acc[r]["F"]
        E1 = tot[r]["E"] - acc[r]["E"]
        h1.append(E1 / F1 if F1 > 0 else float("nan"))
        hp.append(tot[r]["E"] / tot[r]["F"])
    x = list(range(len(rhos)))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, h0, "-o", ms=5, color=SERIES[0], lw=1.6, zorder=3)
    ax.plot(x, hp, "-o", ms=5, color=SERIES[1], lw=1.6, zorder=3)
    ax.plot(x, h1, "-o", ms=5, color=SERIES[2], lw=1.6, zorder=3)
    axis(ax, rhos)
    ax.set_yscale("log")
    ax.set_ylabel(r"$h^{\mathrm{rule}}$")
    ax.set_xlim(-0.5, len(rhos) + 1.6)
    # A log axis over less than a decade and a half labels one tick by
    # default, which is unreadable. Name them.
    ticks = [2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3]
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0.0002", "0.0003", "0.0005", "0.0010", "0.0020",
                        "0.0030"], fontsize=8)
    ax.set_yticks([], minor=True)
    ax.set_ylim(1.8e-4, 4.2e-3)
    label_line(ax, x[-1], h0[-1], r"$j=0$ only", color=SERIES[0], dx=6, dy=0)
    label_line(ax, x[-1], hp[-1], "pooled over all $j$", color=SERIES[1],
               dx=6, dy=0)
    label_line(ax, x[-1], h1[-1], r"$j\geq 1$ (censored)", color=SERIES[2],
               dx=6, dy=-10)
    save(fig, "first_vs_later.png")


def fig_weight(acc, tot, rhos):
    """Why pooling hid the rise: the share of forced emissions at j = 0 falls
    threefold across the high arm, because low-rho traces are long."""
    w = [100.0 * acc[r]["F"] / tot[r]["F"] for r in rhos]
    x = list(range(len(rhos)))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x, w, "-o", ms=6, color=SERIES[0], lw=1.6, zorder=3)
    axis(ax, rhos)
    ax.set_ylabel(r"share of forced emissions at $j=0$  (%)")
    ax.set_ylim(0, max(w) * 1.22)
    rho_c_line(ax, rhos, max(w) * 1.18)
    label_line(ax, x[0], w[0], "33.8%", color=SERIES[0], dx=-2, dy=14)
    label_line(ax, x[-1], w[-1], "12.3%", color=SERIES[0], dx=6, dy=0)
    save(fig, "weight.png")


def fig_named(acc, rhos):
    """The selection j = 0 does NOT remove: h_rule conditions on the model
    having named a forced cell, and that share is density-dependent."""
    y = [100.0 * acc[r]["F"] / acc[r]["at_risk"] for r in rhos]
    x = list(range(len(rhos)))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x, y, "-o", ms=6, color=SERIES[0], lw=1.6, zorder=3)
    axis(ax, rhos)
    ax.set_ylabel(r"named a forced cell at $j=0$  (%)")
    ax.set_ylim(60, 102)
    rho_c_line(ax, rhos, 100.0)
    label_line(ax, x[0], y[0], "96.3%", color=SERIES[0], dx=-2, dy=14)
    label_line(ax, x[-1], y[-1], "75.9%", color=SERIES[0], dx=6, dy=0)
    save(fig, "named.png")


def fig_forest(acc, rhos):
    """Both arms excluding 1, as rate ratios against the peak. The peak sits
    at 1 by construction and carries no interval."""
    pk = max(rhos, key=lambda r: acc[r]["E"] / acc[r]["F"])
    gp = group(acc, [pk])
    rows = [("four added bins\n" + r"$\rho \leq 0.45$", ADDED),
            (r"$\rho \geq 0.80$ pooled", HIGH_RHOS),
            (r"$\rho = 0.95$ densest", [0.95])]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.axvline(1.0, color=MUTED, lw=1.2, ls="--", zorder=1)
    for i, (nm, rs) in enumerate(rows):
        R, lo, hi, _se = ratio_poisson(group(acc, rs), gp)
        ax.plot([lo, hi], [i, i], "-", color=SERIES[0], lw=2.2, zorder=3)
        ax.plot([R], [i], "o", ms=8, color=SERIES[0], zorder=4)
        ax.text(hi + 0.03, i, "%.2f" % R, va="center", fontsize=9,
                color=SERIES[0])
    style_axes(ax, xlabel=r"rate ratio against the peak at $\rho = 0.75$")
    ax.set_yticks(list(range(len(rows))))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0.15, 1.15)
    ax.text(1.02, -0.45, "peak", fontsize=9, color=MUTED)
    save(fig, "forest.png")


def main():
    seq, _n, _blank = sequences()
    acc, _per_run = slice_at(seq, None)
    tot = collections.defaultdict(lambda: {"E": 0.0, "F": 0.0})
    for (_run, rho), byj in seq.items():
        for r in byj.values():
            if r["at_risk"] <= 0:
                continue
            tot[rho]["E"] += r["err_in_F"]
            tot[rho]["F"] += r["in_F"]
    rhos = sorted(acc, reverse=True)
    assert len(rhos) == 14, len(rhos)
    fig_hump(acc, rhos)
    fig_first_vs_later(acc, tot, rhos)
    fig_weight(acc, tot, rhos)
    fig_named(acc, rhos)
    fig_forest(acc, rhos)


if __name__ == "__main__":
    main()
