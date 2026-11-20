"""Figures for the stage SECTION deck (/spin-gnn/deduction-share/).

Figure env only -- py-matplotlib unloads py-pytorch on this cluster, so this
reads stage SECTION's CSVs and imports neither torch nor the project package.

  module purge; module load devel math viz python/3.12.1
  module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
  export MPLCONFIGDIR=/scratch/users/hshunt/mplconfig
  python3 scripts/stagesection_figs.py
"""
import csv
import glob
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/users/hshunt/.local/lib/claude-slides")
from slides import SERIES, MUTED, style_axes, series_colors  # noqa: E402

SCAN = "/scratch/users/hshunt/spin_gnn/runs/stagesection_2026-11-20"
OUTDIR = "/home/users/hshunt/slides/decks/spin-gnn/deduction-share/figures"
GATE = 0.740
N_NODES = 48
LOGN = math.log(N_NODES)
IDEAL_NOVEL = 3.317          # token_budget_probe.py, residual-restricted
DPI = 150


def load():
    rows = []
    for path in sorted(glob.glob(os.path.join(SCAN, "section_*of*.csv"))):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                for k in ("rho50_pub", "rho", "nll_sum", "total_nll",
                          "total_accuracy"):
                    r[k] = float(r[k])
                for k in ("nll", "accuracy"):
                    r[k] = float(r[k]) if r[k] != "" else None
                for k in ("n_params", "gate_pass", "n_tokens", "n_correct"):
                    r[k] = int(r[k])
                rows.append(r)
    return rows


def grouped(rows):
    g = defaultdict(dict)
    for r in rows:
        g[(r["run_id"], r["rho"])][r["section"]] = r
    return g


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_loss_share(g, runs):
    """Where the loss actually goes, at the hardest density scored."""
    rho = 0.60
    order = ["graph_nbr_novel", "graph_nbr_copy", "hint_color", "hint_node",
             "reason_node", "round_mark"]
    label = {"graph_nbr_novel": "graph: novel edge",
             "graph_nbr_copy": "graph: repeated edge",
             "hint_color": "hints: colour",
             "hint_node": "hints: which cell",
             "reason_node": "DEDUCTION: which cell",
             "round_mark": "round / end markers"}
    shares = []
    for s in order:
        num = median([g[(r, rho)][s]["nll_sum"] for r in runs])
        den = median([sum(v["nll_sum"] for v in g[(r, rho)].values()) for r in runs])
        shares.append(num / den)
    # reason_color is 0.0000 nats -- stated in the body, not drawn as a sliver.
    cols = series_colors(len(order), highlight=4)
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    y = np.arange(len(order))[::-1]
    ax.barh(y, shares, color=cols, rasterized=True, height=0.68)
    for yy, s in zip(y, shares):
        ax.text(s + 0.012, yy, "%.1f%%" % (100 * s), va="center",
                fontsize=12, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([label[s] for s in order], fontsize=12)
    ax.set_xlabel("share of total validation NLL")
    ax.set_xlim(0, max(shares) * 1.18)
    style_axes(ax)
    save(fig, "loss_share.png")


def fig_deduce_share(g, runs, rhos):
    xs, med, hi = [], [], []
    for rho in rhos:
        v = []
        for r in runs:
            sec = g[(r, rho)]
            tot = sum(x["nll_sum"] for x in sec.values())
            ded = sec["reason_node"]["nll_sum"] + sec["reason_color"]["nll_sum"]
            v.append(ded / tot)
        xs.append(rho)
        med.append(median(v))
        hi.append(max(v))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(xs, med, "o-", color=SERIES[0], lw=2.2, ms=7, rasterized=True)
    ax.plot(xs, hi, "o--", color=MUTED, lw=1.6, ms=5, rasterized=True)
    ax.text(xs[-1] + 0.006, med[-1], "median of 79", color=SERIES[0],
            fontsize=12, va="center")
    ax.text(xs[-1] + 0.006, hi[-1], "worst of 79", color=MUTED,
            fontsize=12, va="center")
    ax.set_xlabel(r"hint density $\rho$")
    ax.set_ylabel("deduction share of total NLL")
    ax.set_ylim(0, max(hi) * 1.25)
    ax.set_xlim(min(xs) - 0.03, max(xs) + 0.10)
    ax.axhline(0, color=MUTED, lw=0.8)
    style_axes(ax)
    save(fig, "deduce_share.png")


def fig_gate_sep(g, runs):
    """The claim: the gate separates on the deduction tokens, not on the total."""
    rho = 0.60
    # SHARED y-axis on purpose.  Per-panel autoscaling zooms the right panel
    # onto a 0.03-wide window and makes a +0.0122 gap look as clean as the
    # +0.1936 one, which is the opposite of the claim.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharey=True)
    for ax, key, title in (
            (axes[0], "accuracy", "deduction tokens\n(reason_node accuracy)"),
            (axes[1], "total_accuracy", "all tokens\n(published metric)")):
        p = [g[(r, rho)]["reason_node"][key] for r in runs
             if g[(r, rho)]["reason_node"]["gate_pass"]]
        f = [g[(r, rho)]["reason_node"][key] for r in runs
             if not g[(r, rho)]["reason_node"]["gate_pass"]]
        rng = np.random.default_rng(0)
        for i, (vals, col) in enumerate(((f, MUTED), (p, SERIES[0]))):
            ax.scatter(np.full(len(vals), i) + rng.uniform(-0.1, 0.1, len(vals)),
                       vals, s=26, color=col, alpha=0.75, rasterized=True)
            ax.plot([i - 0.26, i + 0.26], [median(vals)] * 2, color=col, lw=2.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["fails gate\n(n=56)", "passes gate\n(n=23)"],
                           fontsize=11)
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(title, fontsize=12.5, color=MUTED)
        ax.text(0.03, 0.93, "gap %+.4f" % (median(p) - median(f)),
                transform=ax.transAxes, ha="left", fontsize=13,
                color=SERIES[0] if key == "accuracy" else MUTED)
        ax.set_ylim(0.44, 1.02)
        style_axes(ax)
    axes[0].set_ylabel("teacher-forced accuracy")
    fig.tight_layout()
    save(fig, "gate_sep.png")


def fig_rank(g, runs):
    rho = 0.60
    x = [g[(r, rho)]["reason_node"]["accuracy"] for r in runs]
    y = [g[(r, rho)]["reason_node"]["rho50_pub"] for r in runs]
    gp = [g[(r, rho)]["reason_node"]["gate_pass"] for r in runs]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for flag, col, lab in ((0, MUTED, "fails gate"), (1, SERIES[0], "passes gate")):
        xs = [a for a, f in zip(x, gp) if f == flag]
        ys = [b for b, f in zip(y, gp) if f == flag]
        ax.scatter(xs, ys, s=34, color=col, alpha=0.8, rasterized=True)
    ax.axhline(GATE, color=SERIES[1], lw=1.4, ls="--")
    ax.text(min(x) + 0.01, GATE - 0.012, r"gate $\rho_{50}=0.740$",
            color=SERIES[1], fontsize=11.5)
    # Axes coords, not data coords: at data (0.55, 0.80) the "fails gate" label
    # ran off the left spine and was clipped.
    ax.text(0.98, 0.95, "passes gate", transform=ax.transAxes, ha="right",
            color=SERIES[0], fontsize=12)
    ax.text(0.03, 0.14, "fails gate", transform=ax.transAxes, ha="left",
            color=MUTED, fontsize=12)
    ax.set_xlabel(r"reason_node accuracy at $\rho=0.60$")
    ax.set_ylabel(r"$\rho_{50}$  (lower is better)")
    ax.invert_yaxis()
    style_axes(ax)
    save(fig, "rank_scatter.png")


def colour_exclusion(rho, n_nodes=N_NODES, q=4):
    """Nats for a novel slot if the ONLY structure used is `a neighbour cannot
    repeat my colour`, with colours known on a rho fraction of cells.

    Not a bound -- an estimate of the SLOPE in rho.  The level is far above the
    measurement because the model also uses degree saturation, the already-named
    set and the fall structure; only the rho-dependence is being compared.
    """
    cand = n_nodes - 1            # every cell but this row's own
    same = (n_nodes / q) - 1      # others sharing my colour: 11 at N=48, q=4
    # My colour is known with prob rho; only then can same-coloured candidates
    # be struck out, and only the rho fraction of those whose colour is hinted.
    return rho * math.log(cand - rho * same) + (1 - rho) * math.log(cand)


def fig_novel(g, runs, rhos):
    """The novel-edge slot is near-incompressible AND it tracks hint density.

    The earlier version drew IDEAL_NOVEL = 3.317 as an `information-theoretic
    bound`, which was false: the measurement crosses BELOW it at rho = 0.90.
    3.317 is hint-independent, and the true novel-slot entropy falls with rho
    because hints forbid same-colour neighbours.  Drawn as the estimate it is.
    """
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    med = [median([g[(r, rho)]["graph_nbr_novel"]["nll"] for r in runs])
           for rho in rhos]
    best = [min(g[(r, rho)]["graph_nbr_novel"]["nll"] for r in runs) for rho in rhos]
    excl = [colour_exclusion(rho) for rho in rhos]
    ax.plot(rhos, med, "o-", color=SERIES[0], lw=2.2, ms=7, rasterized=True)
    ax.plot(rhos, best, "o-", color=SERIES[1], lw=2.0, ms=6, rasterized=True)
    ax.plot(rhos, excl, "s--", color=MUTED, lw=1.6, ms=5, rasterized=True)
    ax.axhline(LOGN, color=MUTED, lw=1.0, ls=":")
    ax.text(rhos[-1] + 0.008, med[-1], "median model", color=SERIES[0],
            fontsize=12, va="center")
    ax.text(rhos[-1] + 0.008, best[-1] - 0.02, "best model", color=SERIES[1],
            fontsize=12, va="center")
    ax.text(rhos[-1] + 0.008, excl[-1], "colour exclusion\n(estimate)",
            color=MUTED, fontsize=11.5, va="center")
    ax.text(min(rhos) - 0.03, LOGN + 0.03,
            r"no information: $\log 48 = 3.871$", color=MUTED, fontsize=11.5)
    ax.set_xlabel(r"hint density $\rho$")
    ax.set_ylabel("novel-edge slot NLL (nats)")
    ax.set_xlim(min(rhos) - 0.04, max(rhos) + 0.19)
    ax.set_ylim(3.0, 3.95)
    style_axes(ax)
    save(fig, "novel_bound.png")
    print("novel medians %s" % ["%.4f" % v for v in med])
    print("novel best    %s" % ["%.4f" % v for v in best])
    print("colour-excl   %s" % ["%.4f" % v for v in excl])
    print("measured slope %.4f over rho, estimate slope %.4f"
          % (med[-1] - med[0], excl[-1] - excl[0]))


def main():
    rows = load()
    g = grouped(rows)
    runs = sorted({r["run_id"] for r in rows})
    rhos = sorted({r["rho"] for r in rows})
    print("%d runs, rhos %s" % (len(runs), rhos))
    fig_loss_share(g, runs)
    fig_deduce_share(g, runs, rhos)
    fig_gate_sep(g, runs)
    fig_rank(g, runs)
    fig_novel(g, runs, rhos)


if __name__ == "__main__":
    main()
