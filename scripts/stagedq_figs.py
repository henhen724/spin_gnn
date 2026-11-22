"""Figures for the stage DQ deck (/spin-gnn/depth-lever/).

Figure env only -- py-matplotlib unloads py-pytorch on this cluster, so this
reads CSVs and imports neither torch nor the project package.

  module purge; module load devel math viz python/3.12.1
  module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig
  python3 scripts/stagedq_figs.py

Figures are rendered INTO THE PROJECT REPO and copied to ~/slides at deck
time, which is the documented workflow and also keeps a figures/ directory
from appearing under decks/spin-gnn/ before a deck.md does.

TWO GROUPS, AND THE FIRST ONE IS ALREADY BUILDABLE.

  EXPLANATION figures come from runs/stagedq_ensemble_2026-11-21/, which is a
  property of the INSTANCE ENSEMBLE and needed no model, so they can be drawn
  and checked while the cells are still training. They are what makes
  rho_c ~ 0.25 and the depth lever legible to a reader who has not seen the
  code.

  RESULTS figures come from the array and the section split. Each one SKIPS
  with a printed line if its CSV is not on disk yet, so this script is safe
  to run at any point and tells you what is still missing.

RHO RUNS RIGHT-TO-LEFT ON EVERY DENSITY AXIS. Hard instances are at LOW rho,
the curves fan out there, and the house rule is to label each curve at its
right-hand end with no legend box. Inverting the axis puts the separated ends
on the right and makes left-to-right read easy-to-hard.

NO BANNED q=4 CONSTANT IS DRAWN. The free line 0.740, s*, the val_nll floors
and (5/16) log N appear nowhere. The d5q4 panel in the collapse figure is
there because REGISTERED.md rests the rho_c ~ 0.25 reading on that diagnostic
RECOVERING the published 0.465 -- showing the calibration arm is what makes
the measurement calibrated rather than fitted, and it is labelled as such.
Depth is the one quantity the registration names as cross-ensemble
comparable, so it is the only axis on which the two families share a figure.
"""
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line  # noqa: E402

SCRATCH = os.environ.get("SCRATCH", "/scratch/users/hshunt")
ENS = "/home/users/hshunt/spin_gnn/runs/stagedq_ensemble_2026-11-21"
SCAN = os.path.join(SCRATCH, "spin_gnn/sudoku_dq_2026-11-21")
SPLIT = os.path.join(SCRATCH, "spin_gnn/runs/stagedq_2026-11-21")
OUTDIR = "/home/users/hshunt/spin_gnn/runs/stagedq_2026-11-21/figures"
DPI = 150

LEVER, REF = "d4q3_fall", "d5q4_fall"
RHO_C_LEVER, RHO_C_REF = 0.25, 0.465
CHANCE = math.log(3)                    # ln 3 = 1.098612
D1_HN, D1_HT = 0.10, 0.55
CONF, N_PER_BIN = 0.95, 400


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def rows(path, casts):
    if not os.path.exists(path):
        return None
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k, fn in casts.items():
                if r.get(k) not in (None, ""):
                    r[k] = fn(r[k])
            out.append(r)
    return out


def hard_left(ax):
    """Rho decreasing left to right: easy on the left, hard on the right."""
    ax.invert_xaxis()


def curve(pairs):
    """Sorted (x, y) with non-finite y dropped.

    The ensemble pre-check writes NaN at any density BELOW an ensemble's hint
    floor, because greedy removal cannot reach it -- d5q4 is NaN under 0.30
    while d4q3 goes to 0.14. Plotting the raw column puts the right-hand
    label at NaN, which matplotlib silently declines to draw: the calibration
    curve would have arrived in the deck with no name on it. Filtering here
    and labelling the last SURVIVING point is the fix, and the gap that is
    left is a real property of the ensemble, annotated where it shows.
    """
    pts = sorted((x, y) for x, y in pairs
                 if y is not None and math.isfinite(y))
    return (list(zip(*pts)) if pts else (None, None))


# --------------------------------------------------------- explanation figs
def fig_depth_vs_rho():
    data = rows(os.path.join(ENS, "t_vs_rho.csv"),
                {"n_nodes": int, "rho": float, "T_mean": float})
    if data is None:
        print("skip depth_vs_rho: no t_vs_rho.csv")
        return
    want = [96, 192, 384]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for col, n in zip(SERIES, want):
        xs, ys = curve((r["rho"], r["T_mean"]) for r in data
                       if r["ensemble"] == LEVER and r["n_nodes"] == n)
        if xs is None:
            continue
        ax.plot(xs, ys, "o-", color=col, lw=2.2, ms=5, rasterized=True)
        label_line(ax, xs[0], ys[0], "N = %d" % n, color=col)
    ax.axvline(RHO_C_LEVER, color=MUTED, lw=1.4, ls="--")
    ax.text(RHO_C_LEVER, 0.97, r"$\rho_c \approx %.2f$  " % RHO_C_LEVER,
            transform=ax.get_xaxis_transform(), color=MUTED,
            ha="right", va="top", fontsize=11)
    style_axes(ax, xlabel=r"hint density $\rho$   (harder $\rightarrow$)",
               ylabel="DD rounds to finish, $T$")
    hard_left(ax)
    save(fig, "depth_vs_rho.png")


def fig_collapse():
    data = rows(os.path.join(ENS, "t_vs_rho.csv"),
                {"n_nodes": int, "rho": float, "T_mean": float})
    if data is None:
        print("skip collapse: no t_vs_rho.csv")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    for ax, ens, rc, title in (
            (axes[0], REF, RHO_C_REF, "$d = 5$, $q = 4$   (previous runs)"),
            (axes[1], LEVER, RHO_C_LEVER, "$d = 4$, $q = 3$   (this run)")):
        stop = None
        for col, n in zip(SERIES, (96, 192, 384)):
            xs, ys = curve((r["rho"], r["T_mean"] / math.log(n)) for r in data
                           if r["ensemble"] == ens and r["n_nodes"] == n)
            if xs is None:
                continue
            stop = xs[0] if stop is None else min(stop, xs[0])
            ax.plot(xs, ys, "o-", color=col, lw=2.0, ms=4, rasterized=True)
            label_line(ax, xs[0], ys[0], "N = %d" % n, color=col)
        ax.axvline(rc, color=MUTED, lw=1.4, ls="--")
        ax.set_title(title, color=MUTED, fontsize=11)
        style_axes(ax, xlabel=r"hint density $\rho$   (harder $\rightarrow$)")
        hard_left(ax)
        # ha="right" against an INVERTED axis puts the text on the high-rho
        # side of the line, which is the empty side in both panels.
        ax.text(rc, 0.97, r"$\rho_c = %.3g$  " % rc,
                transform=ax.get_xaxis_transform(), color=MUTED,
                fontsize=11, ha="right", va="top")
        # Say why a curve stops. The d5q4 family has no instances below its
        # own hint floor, so its panel ends early; that is the ensemble, not
        # missing data, and an unexplained stop reads as the latter.
        if stop is not None and stop > 0.2:
            ax.text(0.99, 0.03, "curves stop at this\nensemble's hint floor",
                    transform=ax.transAxes, color=MUTED,
                    fontsize=10, ha="right", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", pad=2.0))
    style_axes(axes[0], ylabel=r"$T / \log N$")
    save(fig, "collapse.png")


def fig_depth_vs_n():
    data = rows(os.path.join(ENS, "depth_vs_n.csv"),
                {"n_nodes": int, "T_rhomin_mean": float, "seq_len": float})
    if data is None:
        print("skip depth_vs_n: no depth_vs_n.csv")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for col, ens, lab in ((SERIES[0], LEVER, "$d{=}4$, $q{=}3$ (this run)"),
                          (MUTED, REF, "$d{=}5$, $q{=}4$ (previous)")):
        xs, ys = curve((r["n_nodes"], r["T_rhomin_mean"]) for r in data
                       if r["ensemble"] == ens)
        if xs is None:
            continue
        ax.plot(xs, ys, "o-", color=col, lw=2.2, ms=6, rasterized=True)
        label_line(ax, xs[-1], ys[-1], lab, color=col)
    style_axes(ax, xlabel="nodes $N$", ylabel="DD rounds at the hint floor",
               logx=True)
    ticks = [48, 96, 192, 384]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xticks([], minor=True)
    # The two families do NOT share a floor, so each point sits at its own
    # ensemble's lowest reachable density. Saying so is the difference
    # between a lever and an unfair comparison.
    ax.text(0.02, 0.96, "each point at that ensemble's own\n"
                        "lowest reachable hint density",
            transform=ax.transAxes, color=MUTED, fontsize=10,
            ha="left", va="top")
    save(fig, "depth_vs_n.png")


# ------------------------------------------------------------- results figs
def _by_cell(pattern, casts, key):
    out = defaultdict(list)
    for p in sorted(glob.glob(pattern)):
        for r in rows(p, casts) or []:
            out[key(r)].append(r)
    return out


def fig_d5_trainability():
    hits = sorted(glob.glob(os.path.join(SCAN, "n*_d4_q3_L12_s*",
                                         "*", "history.json")))
    if not hits:
        print("skip d5_trainability: no history.json yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    panels = {96: axes[0], 120: axes[1]}
    for p in hits:
        cell = os.path.basename(os.path.dirname(os.path.dirname(p)))
        n = int(cell.split("_")[0][1:])
        seed = int(cell.rsplit("_s", 1)[1])
        ax = panels.get(n)
        if ax is None:
            continue
        h = json.load(open(p))
        xs, ys = curve((r["epoch"], r["nll"]) for r in h)
        if xs is None:
            continue
        col = SERIES[seed % len(SERIES)]
        ax.plot(xs, ys, "-", color=col, lw=2.2, rasterized=True)
        label_line(ax, xs[-1], ys[-1], "seed %d" % seed, color=col)
    for n, ax in panels.items():
        ax.set_title("N = %d" % n, color=MUTED, fontsize=11)
        style_axes(ax, xlabel="epoch")
    style_axes(axes[0], ylabel="validation NLL (nats / token)")
    save(fig, "d5_trainability.png")


def fig_d1_reason_color():
    cells = _by_cell(os.path.join(SPLIT, "dqsplit_*.csv"),
                     {"n_nodes": int, "rho": float, "mean_nll": float,
                      "accuracy": float},
                     lambda r: (int(r["n_nodes"]),
                                int(r["pair_key"].rsplit("_s", 1)[1])))
    if not cells:
        print("skip d1_reason_color: no dqsplit_*.csv yet")
        return
    for n in sorted({k[0] for k in cells}):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for seed in sorted(s for nn, s in cells if nn == n):
            xs, ys = curve((r["rho"], r["mean_nll"])
                           for r in cells[(n, seed)]
                           if r["section"] == "reason_color")
            if xs is None:
                continue
            col = SERIES[seed % len(SERIES)]
            ax.plot(xs, ys, "o-", color=col, lw=2.2, ms=6, rasterized=True)
            label_line(ax, xs[0], ys[0], "seed %d" % seed, color=col)
        for y, txt in ((CHANCE, r"chance, $\ln 3$"),
                       (D1_HT, "H-T bar"), (D1_HN, "H-N bar")):
            ax.axhline(y, color=MUTED, lw=1.3, ls="--")
            ax.text(ax.get_xlim()[0], y, " " + txt, color=MUTED,
                    va="bottom", ha="left", fontsize=10)
        ax.set_ylim(bottom=-0.04)
        style_axes(ax, xlabel=r"hint density $\rho$   (harder $\rightarrow$)",
                   ylabel="colour-token NLL (nats)")
        hard_left(ax)
        save(fig, "d1_reason_color_n%d.png" % n)


def fig_d4():
    casts = {"n_nodes": int, "seed": int, "rho": float, "realized_rho": float,
             "solve_rate": float, "dd_mean_T": float, "n_instances": int}
    cells = _by_cell(os.path.join(SCAN, "rho_curve_n*_s*_*.csv"), casts,
                     lambda r: (r["n_nodes"], r["seed"]))
    if not cells:
        print("skip d4: no rho_curve CSVs yet")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    panels = {n: axes[i] for i, n in enumerate(sorted({k[0] for k in cells}))}
    for (n, seed), rs in sorted(cells.items()):
        ax = panels[n]
        xs, ys = curve((r["rho"], r["solve_rate"]) for r in rs)
        if xs is None:
            continue
        col = SERIES[seed % len(SERIES)]
        ax.plot(xs, ys, "o-", color=col, lw=2.2, ms=5, rasterized=True)
        label_line(ax, xs[0], ys[0], "seed %d" % seed, color=col)
    for n, ax in panels.items():
        ax.axvline(RHO_C_LEVER, color=MUTED, lw=1.4, ls="--")
        ax.set_title("N = %d" % n, color=MUTED, fontsize=11)
        style_axes(ax, xlabel=r"hint density $\rho$   (harder $\rightarrow$)")
        hard_left(ax)
    style_axes(axes[0], ylabel="whole-instance solve rate")
    save(fig, "d4_solve_vs_rho.png")

    # a_eff against DD depth, with the Clopper-Pearson bound drawn as a
    # DOWNWARD ARROW wherever a bin solved nothing -- a zero bin bounds the
    # per-deduction rate from above and must never be plotted as a point.
    p_up = 1.0 - (1.0 - CONF) ** (1.0 / N_PER_BIN)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for (n, seed), rs in sorted(cells.items()):
        col = SERIES[sorted({k[0] for k in cells}).index(n) % len(SERIES)]
        solid, bounds = [], []
        for r in rs:
            nb = r["n_nodes"] - round(r["n_nodes"] * r["realized_rho"])
            if nb <= 0:
                continue
            if r["solve_rate"] > 0:
                solid.append((r["dd_mean_T"], r["solve_rate"] ** (1.0 / nb)))
            else:
                bounds.append((r["dd_mean_T"], p_up ** (1.0 / nb)))
        xs, ys = curve(solid)
        if xs is not None:
            ax.plot(xs, ys, "o-", color=col, lw=2.0, ms=5, rasterized=True)
            label_line(ax, xs[-1], ys[-1], "N = %d" % n, color=col)
        for x, y in bounds:
            ax.annotate("", xy=(x, y - 0.012), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4))
    ax.text(0.02, 0.04, "arrows: 0 of %d solved, so $a_{\\mathrm{eff}}$ is "
                        "bounded above, not measured" % N_PER_BIN,
            transform=ax.transAxes, color=MUTED, fontsize=10)
    style_axes(ax, xlabel="DD rounds in the instance, $T$",
               ylabel=r"per-deduction rate $a_{\mathrm{eff}}$")
    save(fig, "d4_aeff_vs_depth.png")


def main():
    fig_depth_vs_rho()
    fig_collapse()
    fig_depth_vs_n()
    fig_d5_trainability()
    fig_d1_reason_color()
    fig_d4()
    return 0


if __name__ == "__main__":
    sys.exit(main())
