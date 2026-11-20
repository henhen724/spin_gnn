"""Figures for stage WIDTH's deck (/spin-gnn/round-width/).

Reads `width_control.json` (the post-hoc density-controlled numbers) and the
raw per-(run, rho) CSVs.  Figure env only -- py-matplotlib unloads torch, so
this never runs in the same shell as the probe.

One claim per figure, matching the deck's page titles.  Every figure here was
read back as a PNG before the deck shipped.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line, series_colors

SCAN = "/scratch/users/hshunt/spin_gnn/runs/stagewidth_2026-11-20"
OUT = "/home/users/hshunt/slides/decks/spin-gnn/round-width/figures"
NOT_CEILING = ("L08_s0", "L08_s1")
DENSITIES = [0.75, 0.60, 0.45, 0.35]
MIN_TOK = 200


def load_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(SCAN, "width_*of*.csv"))):
        with open(path) as fh:
            rows.extend(list(csv.DictReader(fh)))
    return rows


def by_rho_w(rows, n_nodes, only_pairs=None):
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        if int(r["n_nodes"]) != n_nodes or not r["pair_key"]:
            continue
        if only_pairs and r["pair_key"] not in only_pairs:
            continue
        for w, (c, n) in json.loads(r["by_w"]).items():
            cell = out[(round(float(r["rho"]), 4), int(w))]
            cell[0] += c
            cell[1] += n
    return out


def curve(pool, rho, ws):
    xs, ys = [], []
    for w in ws:
        v = pool.get((rho, w))
        if v and v[1] >= MIN_TOK:
            xs.append(w)
            ys.append(v[0] / v[1])
    return xs, ys


def fig_slope_reversal(ctl):
    """THE headline: the width slope is positive at N=48 and negative at N=96.

    A single panel with a zero line, because the claim is about which side of
    zero each N sits on -- not about the magnitudes, whose ratio is meaningless
    across a sign change.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    rhos = sorted(set(ctl["slopes"]["n48_L08"]) & set(ctl["slopes"]["n96"]),
                  key=float)
    xs = [float(r) for r in rhos]
    s48 = [ctl["slopes"]["n48_L08"][r] for r in rhos]
    s96 = [ctl["slopes"]["n96"][r] for r in rhos]
    ax.axhline(0.0, color="#444444", lw=1.1, zorder=1)
    ax.plot(xs, s48, "o-", color=SERIES[0], lw=2.0, ms=6, zorder=3)
    ax.plot(xs, s96, "s-", color=SERIES[1], lw=2.0, ms=6, zorder=3)
    # The axis is INVERTED (rho decreasing left to right), so the right-hand
    # end of each line is the SMALLEST rho -- xs[0], not xs[-1].  Labelling
    # xs[-1] put both labels at the left edge, on top of their own markers.
    label_line(ax, xs[0], s48[0], "N = 48", color=SERIES[0], dy=4)
    label_line(ax, xs[0], s96[0], "N = 96", color=SERIES[1], dy=-4)
    ax.set_xlabel("hint density  $\\rho$")
    ax.set_ylabel("slope of accuracy per unit round width")
    ax.invert_xaxis()
    ax.set_xlim(0.82, 0.22)
    ax.set_ylim(-0.040, 0.036)
    ax.text(0.80, 0.030, "wider rounds EASIER", color=SERIES[0], fontsize=9.5)
    ax.text(0.80, -0.036, "wider rounds HARDER", color=SERIES[1], fontsize=9.5)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "slope_reversal.png"), dpi=150)
    plt.close(fig)


def fig_width_curves(rows):
    """The curves the slopes are fitted to, faceted by density.

    Small multiples rather than one axis: four densities x two N is eight
    series, and the deck's cap is three per plot.  Shared y so the panels are
    comparable, which is the whole point.
    """
    p48 = by_rho_w(rows, 48, only_pairs=NOT_CEILING)
    p96 = by_rho_w(rows, 96)
    ws = list(range(1, 7))
    fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.5), sharey=True)
    for ax, rho in zip(axes, DENSITIES):
        x48, y48 = curve(p48, rho, ws)
        x96, y96 = curve(p96, rho, ws)
        ax.plot(x48, y48, "o-", color=SERIES[0], lw=1.9, ms=5)
        ax.plot(x96, y96, "s-", color=SERIES[1], lw=1.9, ms=5)
        ax.set_title("$\\rho = %.2f$" % rho, fontsize=11)
        ax.set_xlabel("round width $w$")
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        style_axes(ax)
    axes[0].set_ylabel("teacher-forced accuracy")
    axes[0].set_ylim(0.62, 1.01)
    axes[-1].text(3.1, 0.955, "N = 48", color=SERIES[0], fontsize=10)
    axes[-1].text(3.1, 0.685, "N = 96", color=SERIES[1], fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "width_curves.png"), dpi=150)
    plt.close(fig)


def fig_mediator(ctl):
    """Matching on width -- absolute or fractional -- does not close the gap.

    Both matchings on ONE axis in the same units (accuracy gap), because the
    claim is that they land on top of each other.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    a = defaultdict(list)
    for r in ctl["abs_rows"]:
        a[r["w"]].append(r["gap"])
    f = defaultdict(list)
    for r in ctl["frac_rows"]:
        f[r["w48"]].append(r["gap"])
    ws = sorted(set(a) & set(f))
    ya = [float(np.median(a[w])) for w in ws]
    yf = [float(np.median(f[w])) for w in ws]
    ax.axhline(0.0, color="#444444", lw=1.1, zorder=1)
    ax.plot(ws, ya, "o-", color=SERIES[0], lw=2.0, ms=6, zorder=3)
    ax.plot(ws, yf, "s--", color=SERIES[1], lw=2.0, ms=6, zorder=3)
    label_line(ax, ws[-1], ya[-1], "matched $w$", color=SERIES[0], dy=5)
    label_line(ax, ws[-1], yf[-1], "matched $w/n_{\\rm blank}$",
               color=SERIES[1], dy=-9)
    ax.set_xlabel("round width $w$ at N = 48")
    ax.set_ylabel("accuracy gap,  N=48 $-$ N=96")
    ax.set_xlim(0.6, 9.9)
    ax.set_ylim(-0.02, 0.30)
    ax.text(1.0, 0.012, "a mediator would hold this line at zero",
            color="#444444", fontsize=9.5)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mediator.png"), dpi=150)
    plt.close(fig)


def fig_confound(rows):
    """Why the pooled curve could not be trusted: width is coupled to density.

    The mean rho inside each width bin.  If width were independent of density
    these would be flat lines at the grid mean.
    """
    mix = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if not r["pair_key"]:
            continue
        for w, (_c, n) in json.loads(r["by_w"]).items():
            mix[(int(r["n_nodes"]), int(w))][float(r["rho"])] += n
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for n_nodes, colour, mark in ((48, SERIES[0], "o"), (96, SERIES[1], "s")):
        xs, ys = [], []
        for w in range(1, 7):
            d = mix.get((n_nodes, w))
            if not d or sum(d.values()) < MIN_TOK:
                continue
            tot = sum(d.values())
            xs.append(w)
            ys.append(sum(k * v for k, v in d.items()) / tot)
        ax.plot(xs, ys, mark + "-", color=colour, lw=2.0, ms=6)
        # At w=6 the N=48 line sits BELOW the N=96 one (0.461 vs 0.479), so the
        # offsets have to push apart from there, not toward each other -- the
        # first draft overprinted the two labels exactly.
        label_line(ax, xs[-1], ys[-1], "N = %d" % n_nodes, color=colour,
                   dy=-11 if n_nodes == 48 else 9)
    # The bin that set W5's spread.  Named on the figure because the whole
    # point of this page is that one bin, not the trend.
    ax.annotate("w=5 at N=48:\n29% of its tokens\nare $\\rho=0.90$",
                xy=(5, 0.584), xytext=(3.05, 0.60),
                fontsize=9.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.set_xlabel("round width $w$")
    ax.set_ylabel("mean hint density $\\rho$ of the tokens in the bin")
    ax.set_xlim(0.6, 7.2)
    ax.set_ylim(0.40, 0.65)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confound.png"), dpi=150)
    plt.close(fig)


def fig_width_vs_position(rows):
    """W3: width is a real axis and within-round position is not."""
    pw = defaultdict(lambda: [0, 0])
    pj = defaultdict(lambda: [0, 0])
    for r in rows:
        if int(r["n_nodes"]) != 96 or not r["pair_key"]:
            continue
        for k, (c, n) in json.loads(r["by_w"]).items():
            pw[int(k)][0] += c
            pw[int(k)][1] += n
        for k, (c, n) in json.loads(r["by_j"]).items():
            pj[int(k)][0] += c
            pj[int(k)][1] += n
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    xw = [k for k in range(1, 7) if pw[k][1] >= MIN_TOK]
    yw = [pw[k][0] / pw[k][1] for k in xw]
    xj = [k for k in range(0, 6) if pj[k][1] >= MIN_TOK]
    yj = [pj[k][0] / pj[k][1] for k in xj]
    ax.plot([k for k in xw], yw, "o-", color=SERIES[0], lw=2.1, ms=6)
    ax.plot([k + 1 for k in xj], yj, "s-", color=SERIES[1], lw=2.1, ms=6)
    label_line(ax, xw[-1], yw[-1], "by round width $w$", color=SERIES[0],
               dy=-10)
    label_line(ax, xj[-1] + 1, yj[-1], "by position $j$ in the round",
               color=SERIES[1], dy=8)
    ax.set_xlabel("round width $w$   /   position $j+1$ within the round")
    ax.set_ylabel("teacher-forced accuracy,  N = 96")
    ax.set_ylim(0.70, 0.85)
    ax.set_xlim(0.7, 8.4)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "width_vs_position.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(SCAN, "width_control.json")) as fh:
        ctl = json.load(fh)
    rows = load_rows()
    fig_slope_reversal(ctl)
    fig_width_curves(rows)
    fig_mediator(ctl)
    fig_confound(rows)
    fig_width_vs_position(rows)
    for f in sorted(os.listdir(OUT)):
        print("  %s" % f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
