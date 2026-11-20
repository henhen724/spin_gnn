"""Figures for stage SCALE's deck (/spin-gnn/depth-vs-size/).

Figure env only -- reads stage SCALE's CSVs and imports neither torch nor the
project package:

  module purge; module load devel math viz python/3.12.1
  module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig
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

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, label_line, series_colors  # noqa: E402

SCAN = "/scratch/users/hshunt/spin_gnn/runs/stagescale_2026-11-20"
OUT = "/home/users/hshunt/slides/decks/spin-gnn/depth-vs-size/figures"
PAIRS = ["L08_s0", "L12_s0", "L08_s1"]
RHO_MAIN = 0.60
MIN_TOK = 200
LOG2 = math.log(2)


def load():
    rows = []
    for path in sorted(glob.glob(os.path.join(SCAN, "scale_*of*.csv"))):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                r["rho"] = float(r["rho"])
                r["n_nodes"] = int(r["n_nodes"])
                for k in ("nll_sum", "total_nll", "mean_T"):
                    r[k] = float(r[k])
                for k in ("nll", "accuracy"):
                    r[k] = float(r[k]) if r[k] != "" else None
                for k in ("n_tokens", "n_correct"):
                    r[k] = int(r[k])
                rows.append(r)
    return rows


def depth_table(rows):
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["section"] != "reason_node" or not r["pair_key"]:
            continue
        for t, (c, n) in json.loads(r["by_T"]).items():
            cell = out[(r["n_nodes"], int(t))]
            cell[0] += c
            cell[1] += n
    return out


def rho_mix(rows):
    """(N, T) -> mean rho of the instances contributing that band."""
    acc = defaultdict(lambda: [0.0, 0])
    for r in rows:
        if r["section"] != "reason_node" or not r["pair_key"]:
            continue
        for t, (_c, n) in json.loads(r["by_T"]).items():
            cell = acc[(r["n_nodes"], int(t))]
            cell[0] += r["rho"] * n
            cell[1] += n
    return {k: v[0] / v[1] for k, v in acc.items() if v[1]}


def pair_runs(rows):
    out = defaultdict(dict)
    for r in rows:
        if r["pair_key"]:
            out[r["pair_key"]][r["n_nodes"]] = r["run_id"]
    return out


def sec(rows, run, rho, section):
    for r in rows:
        if r["run_id"] == run and r["rho"] == rho and r["section"] == section:
            return r
    return None


# ---------------------------------------------------------------- figures ---
def fig_depth(rows):
    """THE headline: accuracy vs DD depth at both N.  The registered
    discriminator C2 predicted these two curves would lie on top of each other
    once depth was matched; they do not, and they diverge."""
    dt = depth_table(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    # Each N's own flat region, not a shared window: N=96 takes longer to get
    # there (T~14 vs T~9), which is part of the claim.
    plateau = {48: (12, 21), 96: (16, 30)}
    for i, n_nodes in enumerate((48, 96)):
        ts = sorted(t for (nn, t) in dt if nn == n_nodes
                    and dt[(nn, t)][1] >= MIN_TOK)
        acc = [dt[(n_nodes, t)][0] / dt[(n_nodes, t)][1] for t in ts]
        ax.plot(ts, acc, "o-", color=SERIES[i], ms=4, rasterized=True)
        label_line(ax, ts[-1], acc[-1], "N = %d" % n_nodes, color=SERIES[i])
        lo, hi = plateau[n_nodes]
        c = sum(dt[(n_nodes, t)][0] for t in range(lo, hi + 1))
        n = sum(dt[(n_nodes, t)][1] for t in range(lo, hi + 1))
        ax.hlines(c / n, lo, hi, color=SERIES[i], ls="--", lw=1.4)
        ax.annotate("plateau %.4f" % (c / n), (lo, c / n),
                    textcoords="offset points", xytext=(2, -16),
                    fontsize=11, color=SERIES[i])
    ax.set_xlabel("direct-deduction depth $T$ (rounds to solve the instance)")
    ax.set_ylabel("teacher-forced accuracy,\n`reason_node` tokens")
    ax.set_ylim(0.70, 1.02)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "depth_acc.png"), dpi=150)
    plt.close(fig)
    print("depth_acc: N=48 %.4f->%.4f, N=96 %.4f->%.4f"
          % (dt[(48, 1)][0] / dt[(48, 1)][1], dt[(48, 21)][0] / dt[(48, 21)][1],
             dt[(96, 1)][0] / dt[(96, 1)][1], dt[(96, 21)][0] / dt[(96, 21)][1]))


def fig_gap(rows):
    """The same data as one number per band: the gap, and that it GROWS."""
    dt = depth_table(rows)
    ts, gaps = [], []
    for t in sorted({t for (_n, t) in dt}):
        a, b = dt.get((48, t)), dt.get((96, t))
        if not a or not b or min(a[1], b[1]) < MIN_TOK:
            continue
        ts.append(t)
        gaps.append(a[0] / a[1] - b[0] / b[1])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ts, gaps, "o-", color=SERIES[0], ms=4, rasterized=True)
    # The registered falsifier, drawn so the reader can see which side it fell.
    ax.axhline(0.05, color=MUTED, ls=":", lw=1.2)
    ax.axhline(0.15, color=MUTED, ls="--", lw=1.2)
    ax.text(1.0, 0.057, "C2 passes below this", color=MUTED, fontsize=11,
            va="bottom")
    ax.text(1.0, 0.157, "C2 falsified above this", color=MUTED, fontsize=11,
            va="bottom")
    ax.set_xlabel("direct-deduction depth $T$")
    ax.set_ylabel("accuracy gap, N=48 minus N=96")
    ax.set_ylim(-0.01, 0.20)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "depth_gap.png"), dpi=150)
    plt.close(fig)
    print("depth_gap: %d bands, %.4f -> %.4f" % (len(ts), gaps[0], gaps[-1]))


def fig_confound(rows):
    """The confound, and that it runs the safe way.  At matched depth N=96's
    instances carry MORE hints, so they are EASIER by the density account, and
    the measured gap is understated rather than manufactured."""
    mix = rho_mix(rows)
    dt = depth_table(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, n_nodes in enumerate((48, 96)):
        ts = sorted(t for (nn, t) in dt if nn == n_nodes
                    and dt[(nn, t)][1] >= MIN_TOK)
        mr = [mix[(n_nodes, t)] for t in ts]
        ax.plot(ts, mr, "o-", color=SERIES[i], ms=4, rasterized=True)
        # Both curves converge on rho=0.35 (the lowest density run), so the
        # N=48 label lands on top of the N=96 line unless it is lifted.
        label_line(ax, ts[-1], mr[-1], "N = %d" % n_nodes, color=SERIES[i],
                   dy=14 if n_nodes == 48 else 0)
    ax.set_xlabel("direct-deduction depth $T$")
    ax.set_ylabel("mean hint density $\\rho$ of the\ninstances in that band")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confound_rho.png"), dpi=150)
    plt.close(fig)
    print("confound_rho: T=8 %.3f vs %.3f; T=21 %.3f vs %.3f"
          % (mix[(48, 8)], mix[(96, 8)], mix[(48, 21)], mix[(96, 21)]))


def fig_slots(rows):
    """C4 and C6 as the CHANGE itself, on ONE axis.

    The first draft was two panels, one per slot type.  Unshared axes made the
    copy panel -- whose whole range is 0.019 nats -- render as three steeply
    descending lines, which reads as a large effect and is the exact opposite of
    the claim.  Plotting the change puts both slot types on the same nats scale,
    so `+0.76` and `-0.01` are visually what they are, and keeps the page to one
    claim.
    """
    pr = pair_runs(rows)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    rowy = {"graph_nbr_novel": 1.0, "graph_nbr_copy": 0.0}
    for key, colour, lab in (("graph_nbr_novel", SERIES[0], "novel slot"),
                             ("graph_nbr_copy", SERIES[1], "copy slot")):
        for i, p in enumerate(PAIRS):
            d = (sec(rows, pr[p][96], RHO_MAIN, key)["nll"]
                 - sec(rows, pr[p][48], RHO_MAIN, key)["nll"])
            y = rowy[key] + (i - 1) * 0.13
            ax.plot([d], [y], "o", color=colour, ms=8, rasterized=True)
            ax.annotate("%+.4f  %s" % (d, p), (d, y),
                        textcoords="offset points", xytext=(11, 0),
                        fontsize=11, color=colour, va="center")
    ax.axvline(0.0, color=MUTED, ls=":", lw=1.2)
    ax.axvline(LOG2, color=MUTED, ls="--", lw=1.2)
    ax.text(LOG2, 1.42, "$\\log 2 = 0.6931$", color=MUTED, fontsize=11,
            ha="center")
    ax.text(0.0, -0.42, "no change", color=MUTED, fontsize=11, ha="center")
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["copy slot", "novel slot"])
    ax.set_ylim(-0.55, 1.6)
    ax.set_xlim(-0.25, 1.30)
    ax.set_xlabel("change in NLL from $N=48$ to $N=96$ (nats per token)")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "slot_scaling.png"), dpi=150)
    plt.close(fig)
    for key in ("graph_nbr_novel", "graph_nbr_copy"):
        d = [sec(rows, pr[p][96], RHO_MAIN, key)["nll"]
             - sec(rows, pr[p][48], RHO_MAIN, key)["nll"] for p in PAIRS]
        print("slot_scaling %-16s changes %s (log2 = %.4f)"
              % (key, ["%+.4f" % x for x in d], LOG2))


def fig_seed(rows):
    """The spread that is not N and not depth: seed.  At N=48 two seeds of the
    same cell are indistinguishable; at N=96 they are 0.17 apart."""
    pr = pair_runs(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, (s0, s1, lab) in enumerate((("L08_s0", "L08_s1", "L = 8"),)):
        for n_nodes, marker in ((48, "o"), (96, "s")):
            a = sec(rows, pr[s0][n_nodes], RHO_MAIN, "reason_node")["accuracy"]
            b = sec(rows, pr[s1][n_nodes], RHO_MAIN, "reason_node")["accuracy"]
            ax.plot([n_nodes, n_nodes], [a, b], "-", color=SERIES[0], lw=2.5)
            ax.plot([n_nodes, n_nodes], [a, b], marker, color=SERIES[0], ms=7)
            ax.annotate("seeds %.4f / %.4f\nspread %.4f" % (a, b, abs(a - b)),
                        (n_nodes, min(a, b)), textcoords="offset points",
                        xytext=(10, -2), fontsize=11, color=SERIES[0])
    ax.set_xlim(36, 132)
    ax.set_xticks([48, 96])
    ax.set_ylim(0.715, 0.955)   # headroom so the N=48 annotation clears the frame
    ax.set_xlabel("number of cells $N$")
    ax.set_ylabel("teacher-forced accuracy,\n`reason_node` tokens")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "seed_spread.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    print("%d rows" % len(rows))
    fig_depth(rows)
    fig_gap(rows)
    fig_confound(rows)
    fig_slots(rows)
    fig_seed(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
