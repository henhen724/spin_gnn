"""Figures for stage ENTROPY's deck (/spin-gnn/entropy/).

Figure env only -- py-matplotlib unloads torch, so this never runs in the same
shell as the probe.  One claim per figure, matching the deck's page titles.
Every figure here was read back as a PNG before the deck shipped.
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

SCAN = "/scratch/users/hshunt/spin_gnn/runs/stageentropy_2026-11-20"
OUT = "/home/users/hshunt/slides/decks/spin-gnn/entropy/figures"
BEST = "n48_L12_s0"
H_EDGES = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00]


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(SCAN, "entropy_*of*.csv"))):
        with open(p) as fh:
            rows.extend(list(csv.DictReader(fh)))
    return rows


def ixd(rows):
    return {(r["run_id"], r["rho"], r["section"]): r for r in rows}


def pooled_H(ix, rhos, cell, section):
    num = den = 0.0
    for r in rhos:
        k = (cell, r, section)
        if k not in ix:
            continue
        n = int(ix[k]["n_tokens"])
        num += float(ix[k]["mean_H"]) * n
        den += n
    return num / den if den else float("nan")


def fig_concentration(ix, rhos):
    """THE answer: 83% of deduction tokens carry under 0.01 nats."""
    agg = [0] * (len(H_EDGES) + 1)
    for r in rhos:
        k = (BEST, r, "reason_node")
        if k in ix and ix[k]["h_bins"]:
            for i, v in enumerate(json.loads(ix[k]["h_bins"])):
                agg[i] += v
    tot = sum(agg)
    cum, ys = 0, []
    for i in range(len(H_EDGES)):
        cum += agg[i]
        ys.append(100.0 * cum / tot)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    xs = np.arange(len(H_EDGES))
    # Accent the 0.01 bar: the claim is about the FIRST bin, not the trend.
    cols = series_colors(len(H_EDGES), highlight=0)
    ax.bar(xs, ys, color=cols, width=0.68)
    for x, y in zip(xs, ys):
        ax.text(x, y + 1.6, "%.1f%%" % y, ha="center", fontsize=9.5,
                color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels(["%.2f" % e for e in H_EDGES])
    ax.set_xlabel("per-token entropy threshold $H$  (nats)")
    ax.set_ylabel("% of deduction tokens below it")
    ax.set_ylim(0, 112)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "concentration.png"), dpi=150)
    plt.close(fig)


def fig_sections(ix, rhos, cells):
    """Deterministic sections go to zero; random ones do not move."""
    secs = ["reason_color", "reason_node", "round_mark", "term",
            "graph_nbr_copy", "hint_color", "graph_lead", "graph_nbr_novel",
            "hint_node"]
    vals = [pooled_H(ix, rhos, BEST, s) for s in secs]
    det = [s in ("reason_color", "reason_node", "round_mark", "term")
           for s in secs]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ys = np.arange(len(secs))[::-1]
    cols = [SERIES[0] if d else MUTED for d in det]
    ax.barh(ys, vals, color=cols, height=0.66)
    for y, v in zip(ys, vals):
        ax.text(v + 0.06, y, "%.3f" % v, va="center", fontsize=9.5,
                color=MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels(secs, fontsize=9.5)
    ax.set_xlabel("mean predictive entropy $H$  (nats)")
    ax.set_xlim(0, 3.9)
    ax.text(1.5, ys[0] - 0.1, "predicted zero floor (forced by the prefix)",
            color=SERIES[0], fontsize=9.5)
    ax.text(1.5, ys[-1] + 0.42, "predicted positive floor (a random draw)",
            color=MUTED, fontsize=9.5)
    # graph_lead was pre-registered as random and measured zero: graph rows are
    # emitted in sorted node order, so the row's lead node is forced too.
    ax.text(0.62, ys[6], "<-- predicted random, measured zero",
            fontsize=9.5, color=SERIES[2], va="center")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sections.png"), dpi=150)
    plt.close(fig)


def fig_vs_density(ix, rhos):
    """Certainty is a function of how deep the deduction has to go."""
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    show = [(BEST, "N = 48, L12"), ("n96_L12_s0", "N = 96, L12"),
            ("n48_L04_s0", "N = 48, L04")]
    xs = [float(r) for r in rhos]
    for (cell, lab), col in zip(show, SERIES[:3]):
        ys = [float(ix[(cell, r, "reason_node")]["mean_H"])
              for r in rhos if (cell, r, "reason_node") in ix]
        ax.plot(xs[:len(ys)], ys, "o-", color=col, lw=2.0, ms=6)
        # x-axis is inverted, so the visual right-hand end is xs[0].
        label_line(ax, xs[0], ys[0], lab, color=col, dx=11, dy=5)
    ax.set_yscale("log")
    ax.set_xlabel("hint density  $\\rho$")
    ax.set_ylabel("$H$ on deduction tokens  (nats, log scale)")
    ax.invert_xaxis()
    ax.set_xlim(0.95, 0.24)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vs_density.png"), dpi=150)
    plt.close(fig)


def fig_vs_accuracy(ix, rhos, cells):
    """Entropy is a near-perfect proxy for competence: Spearman -0.97."""
    xs, ys, labs = [], [], []
    for c in cells:
        num = den = 0.0
        for r in rhos:
            k = (c, r, "reason_node")
            if k not in ix:
                continue
            n = int(ix[k]["n_tokens"])
            num += float(ix[k]["accuracy"]) * n
            den += n
        xs.append(num / den)
        ys.append((pooled_H(ix, rhos, c, "reason_node")
                   + pooled_H(ix, rhos, c, "reason_color")) / 2)
        labs.append(c)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    cols = [SERIES[0] if l.startswith("n48") else SERIES[1] for l in labs]
    for x, y, c in zip(xs, ys, cols):
        ax.plot([x], [y], "o", color=c, ms=8)
    ax.set_yscale("log")
    ax.set_xlabel("teacher-forced accuracy on deduction cells")
    ax.set_ylabel("$H$ on deduction tokens  (nats, log scale)")
    ax.text(0.745, 0.60, "N = 48", color=SERIES[0], fontsize=10)
    ax.text(0.745, 0.40, "N = 96", color=SERIES[1], fontsize=10)
    ax.text(0.80, 0.016, "Spearman $= -0.97$", color=MUTED, fontsize=10)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vs_accuracy.png"), dpi=150)
    plt.close(fig)


def fig_n_penalty(ix, rhos, rows):
    """Doubling N raises deduction entropy at every matched cell."""
    pairs = defaultdict(dict)
    for r in rows:
        if r["pair_key"]:
            pairs[r["pair_key"]][int(r["n_nodes"])] = r["run_id"]
    pk = sorted(p for p in pairs if 48 in pairs[p] and 96 in pairs[p])
    a = [pooled_H(ix, rhos, pairs[p][48], "reason_node") for p in pk]
    b = [pooled_H(ix, rhos, pairs[p][96], "reason_node") for p in pk]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(pk))
    ax.bar(x - 0.19, a, width=0.36, color=SERIES[0])
    ax.bar(x + 0.19, b, width=0.36, color=SERIES[1])
    for xi, (u, v) in enumerate(zip(a, b)):
        ax.text(xi - 0.19, u + 0.015, "%.3f" % u, ha="center", fontsize=9.5,
                color=MUTED)
        ax.text(xi + 0.19, v + 0.015, "%.3f" % v, ha="center", fontsize=9.5,
                color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(pk)
    ax.set_xlabel("matched cell (same depth, recipe, token budget, seed)")
    ax.set_ylabel("$H$ on deduction tokens  (nats)")
    ax.set_ylim(0, max(b) * 1.26)
    ax.text(-0.36, max(b) * 1.14, "N = 48", color=SERIES[0], fontsize=10)
    ax.text(-0.36, max(b) * 1.05, "N = 96", color=SERIES[1], fontsize=10)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "n_penalty.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    ix = ixd(rows)
    rhos = sorted({r["rho"] for r in rows}, key=float, reverse=True)
    cells = sorted({r["run_id"] for r in rows})
    fig_concentration(ix, rhos)
    fig_sections(ix, rhos, cells)
    fig_vs_density(ix, rhos)
    fig_vs_accuracy(ix, rhos, cells)
    fig_n_penalty(ix, rhos, rows)
    for f in sorted(os.listdir(OUT)):
        print("  %s" % f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
