"""Figures for the round-width N-ladder pages, 48 to 1020.

Run in the FIGURE environment, never the torch one:

    module purge; module load devel math viz python/3.12.1
    module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
    export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig

Eight N values exceed the validated 3-slot palette, and `series_colors(8)`
refuses rather than cycling, so anything that wants one curve per N is
facetted into small multiples. Pages that want a single trend put N on the
x-axis instead, where there is no series problem at all.
"""
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line, series_colors

SCRATCH = os.environ.get("SCRATCH", "/scratch/users/hshunt")
CSV = os.path.join(SCRATCH, "spin_gnn/runs/widthladder_2026-11-20/width_ladder.csv")
OUT = "/home/users/hshunt/slides/decks/spin-gnn/round-width/figures"
DPI = 150

TRAINED = (48, 96, 192)          # sizes with a model; 192 still training
THREE = ["0.90", "0.60", "0.35"]  # the <=3 densities the by-rho page shows


def load():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        r["n_nodes"] = int(r["n_nodes"])
        for k in ("mean_width", "sd_width", "median_width", "p90_width",
                  "mean_T", "mean_achieved_rho", "max_rho_gap"):
            r[k] = float(r[k])
        r["max_width"] = int(r["max_width"])
        r["hist"] = {int(w): c for w, c in json.loads(r["hist"])}
        r["rho_by_w"] = {int(w): s for w, s in json.loads(r["rho_by_w"])}
    return rows


def by(rows, rho):
    """{N: row} for one density label."""
    return {r["n_nodes"]: r for r in rows if r["rho"] == rho}


def loglog_slope(xs, ys):
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


# ---------------------------------------------------------------- figures

def fig_width_vs_n(rows):
    """Mean round width against N on the full training density mix. One curve,
    so no palette question; the claim is the slope."""
    d = by(rows, "pool")
    Ns = sorted(d)
    ys = [d[n]["mean_width"] for n in Ns]
    slope = loglog_slope(Ns, ys)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(Ns, ys, "-o", color=SERIES[0], linewidth=2.0, markersize=5.5)
    # mark which sizes actually have a model behind them
    tx = [n for n in Ns if n in TRAINED]
    ty = [d[n]["mean_width"] for n in tx]
    ax.plot(tx, ty, "o", markersize=11, markerfacecolor="none",
            markeredgecolor=SERIES[1], markeredgewidth=1.8)
    label_line(ax, tx[-1], ty[-1], "models exist\nup to here", color=SERIES[1],
               dx=-6, dy=-34)
    label_line(ax, Ns[-1], ys[-1], "slope %.2f on log-log" % slope,
               color=SERIES[0], dx=8, dy=0)
    style_axes(ax, xlabel="graph size $N$", ylabel="mean round width $w$",
               logx=True, logy=True)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlim(Ns[0] * 0.88, Ns[-1] * 1.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ladder_width_vs_n.png"), dpi=DPI)
    plt.close(fig)
    return slope, Ns, ys


def fig_width_by_rho(rows):
    """Same trend at three pinned densities -- exactly the 3-slot budget."""
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    cols = series_colors(len(THREE))
    out = {}
    for col, rho in zip(cols, THREE):
        d = by(rows, rho)
        Ns = sorted(d)
        ys = [d[n]["mean_width"] for n in Ns]
        out[rho] = (Ns, ys, loglog_slope(Ns, ys))
        ax.plot(Ns, ys, "-o", color=col, linewidth=2.0, markersize=5)
        label_line(ax, Ns[-1], ys[-1],
                   r"$\rho=%s$  (slope %.2f)" % (rho, out[rho][2]),
                   color=col, dx=8, dy=0)
    style_axes(ax, xlabel="graph size $N$", ylabel="mean round width $w$",
               logx=True, logy=True)
    Ns = sorted(by(rows, THREE[0]))
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlim(Ns[0] * 0.88, Ns[-1] * 3.1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ladder_width_by_rho.png"), dpi=DPI)
    plt.close(fig)
    return out


def fig_confound_ladder(rows):
    """Slide 23's figure at all eight sizes: mean hint density of the rounds
    that land in each width bin, under the full training mix."""
    d = by(rows, "pool")
    Ns = sorted(d)
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6), sharey=True)
    for ax, n in zip(axes.ravel(), Ns):
        r = d[n]
        ws = sorted(r["hist"])
        tot = sum(r["hist"].values())
        # drop the thin tail so the panel is not 80% noise
        keep, run = [], 0
        for w in ws:
            keep.append(w)
            run += r["hist"][w]
            if run >= 0.99 * tot:
                break
        ys = [r["rho_by_w"][w] / r["hist"][w] for w in keep]
        col = SERIES[0] if n in TRAINED else MUTED
        ax.plot(keep, ys, "-", color=col, linewidth=1.8)
        ax.set_title("$N=%d$%s" % (n, "" if n in TRAINED else ""),
                     fontsize=13, color=col)
        style_axes(ax, xlabel="round width $w$")
        ax.set_ylim(0.30, 1.0)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"mean hint density $\rho$")
    fig.suptitle("blue = a model exists at this size; grey = instances only",
                 fontsize=12, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "ladder_confound.png"), dpi=DPI)
    plt.close(fig)
    return {n: (sorted(d[n]["hist"]),) for n in Ns}


def fig_width_dist(rows):
    """Where the rounds actually are: fraction of rounds at each width."""
    d = by(rows, "pool")
    Ns = sorted(d)
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6))
    for ax, n in zip(axes.ravel(), Ns):
        r = d[n]
        tot = sum(r["hist"].values())
        ws = sorted(r["hist"])
        ys = [r["hist"][w] / tot for w in ws]
        col = SERIES[0] if n in TRAINED else MUTED
        ax.fill_between(ws, ys, color=col, alpha=0.30, linewidth=0)
        ax.plot(ws, ys, "-", color=col, linewidth=1.5)
        ax.axvline(r["mean_width"], color=SERIES[1], linewidth=1.4,
                   linestyle="--")
        ax.set_title("$N=%d$   mean %.1f   max %d"
                     % (n, r["mean_width"], r["max_width"]),
                     fontsize=11.5, color=col)
        style_axes(ax, xlabel="round width $w$")
        ax.set_xlim(0, max(6, r["p90_width"] * 1.6))
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of rounds")
    fig.suptitle("dashed orange = mean; blue = a model exists at this size",
                 fontsize=12, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "ladder_width_dist.png"), dpi=DPI)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    print("loaded %d rows, N = %s"
          % (len(rows), sorted({r["n_nodes"] for r in rows})))
    slope, Ns, ys = fig_width_vs_n(rows)
    print("\npooled mix: mean width vs N, log-log slope %.4f" % slope)
    for n, y in zip(Ns, ys):
        r = by(rows, "pool")[n]
        print("  N=%4d  mean w %7.3f  sd %6.3f  median %4.1f  p90 %5.1f  "
              "max %3d  T %7.2f  ach rho %.3f"
              % (n, y, r["sd_width"], r["median_width"], r["p90_width"],
                 r["max_width"], r["mean_T"], r["mean_achieved_rho"]))
    out = fig_width_by_rho(rows)
    print("\npinned densities: log-log slope of mean width vs N")
    for rho in THREE:
        print("  rho=%s  slope %.4f   w: %s"
              % (rho, out[rho][2],
                 " ".join("%.2f" % v for v in out[rho][1])))
    fig_confound_ladder(rows)
    fig_width_dist(rows)
    print("\nconfound: mean rho in the narrowest and widest kept bin")
    d = by(rows, "pool")
    for n in sorted(d):
        r = d[n]
        ws = sorted(r["hist"])
        tot = sum(r["hist"].values())
        keep, run = [], 0
        for w in ws:
            keep.append(w); run += r["hist"][w]
            if run >= 0.99 * tot:
                break
        lo = r["rho_by_w"][keep[0]] / r["hist"][keep[0]]
        hi = r["rho_by_w"][keep[-1]] / r["hist"][keep[-1]]
        print("  N=%4d  w=%d rho %.3f  ->  w=%d rho %.3f   (delta %+.3f)"
              % (n, keep[0], lo, keep[-1], hi, hi - lo))
    print("\nwrote 4 figures to %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
