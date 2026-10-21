"""Correction deck: the frontier result reverses, and depth turns out to cost.

The deck sent on 2026-10-21 said the model never represents DD's frontier
predicate. On a threshold-free metric that is wrong, and this deck says so
first rather than burying it. Built from 48861799 (probe rescore, AUROC +
average precision, seeds 0/1/2) and 48863464 (solve_rate binned by DD round
count T at fixed hint density).

Usage:
    python scripts/build_frontier_correction_slides.py --out slides.pdf
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

SCRATCH = os.environ.get("SCRATCH", "")
PROBES = f"{SCRATCH}/spin_gnn/sudoku_series_probes_auc_2026-10-21/series_probes_auc.csv"
DEPTH = f"{SCRATCH}/spin_gnn/sudoku_depth_at_fixed_rho_2026-10-21/depth_at_fixed_rho.csv"
SERIES = f"{SCRATCH}/spin_gnn/sudoku_checkpoint_series_2026-10-19/series.csv"

# From 48861799's label-only baseline step. These are computed from DD's own
# ground truth with no model involved, so they are what the model's number has
# to be read against.
BASELINES = {
    "not_known (1 - known_mask)": (0.7341, 0.7341),
    "known_probe (known bits only)": (0.5882, 0.7672),
    "degree_aug (known + n_known_nbrs)": (0.6478, 0.7834),
    "shuffled (chance control)": (0.5000, 0.5096),
}
MODEL_BAL, MODEL_AUC = 0.679, 0.8636
FIG = (11, 8.5)


def _title(fig, text, sub=None):
    fig.text(0.5, 0.93, text, ha="center", size=21, weight="bold")
    if sub:
        fig.text(0.5, 0.875, sub, ha="center", size=13, color="#444444")


def load():
    p = pd.read_csv(PROBES)
    p["epoch"] = p.checkpoint.str.extract(r"epoch0*(\d+)").astype(int)
    try:
        s = pd.read_csv(SERIES)
    except FileNotFoundError:
        s = None
    return p, pd.read_csv(DEPTH), s


def page_headline(pdf):
    fig = plt.figure(figsize=FIG)
    _title(fig, "Correction: the model DOES represent DD's frontier",
           "It just never develops. Previous deck (2026-10-21) said the signal was absent.")
    body = [
        ("What changed", "#b00020"),
        ("   The frontier probe was scored with balanced accuracy, which thresholds a", "k"),
        ("   BCE-trained probe's logits at 0. On a 15%-positive target that threshold is", "k"),
        ("   badly placed, so the metric reported a floor. Rescored on AUROC:", "k"),
        ("", "k"),
        ("       frontier_mask peak:   0.679  ->  0.863        (seed std <= 0.0014)", "#b00020"),
        ("       best label-only baseline, with the graph:     0.783", "k"),
        ("       layer-0 control:      0.571  ->  0.676        (lift +0.186)", "k"),
        ("", "k"),
        ("   The model carries more frontier information than DD's own labels plus node", "k"),
        ("   degree can supply. The earlier 'weak evidence' reading was an artifact.", "k"),
        ("", "k"),
        ("What survives", "#1b5e20"),
        ("   0.863 is reached by EPOCH 5, when solve_rate is 0.007, and is 0.864 at epoch", "k"),
        ("   900, when solve_rate is ~0.74. Present before competence, flat through it.", "k"),
        ("   next_color is still the only target that tracks solve_rate (0.327 -> 0.675).", "k"),
        ("", "k"),
        ("What is new", "#0d47a1"),
        ("   Training RELOCATES the computation instead of strengthening it: peak layer", "k"),
        ("   8 -> 5, early layers rise (L1 0.708 -> 0.776), peak value unchanged.", "k"),
        ("   And depth does cost solve_rate at fixed density (page 5) - against my own", "k"),
        ("   pre-registered prediction of 'flat'.", "k"),
    ]
    y = 0.78
    for line, color in body:
        weight = "bold" if color != "k" and not line.startswith("  ") else "normal"
        fig.text(0.08, y, line, size=12.5, family="monospace", color=color, weight=weight)
        y -= 0.034
    pdf.savefig(fig)
    plt.close(fig)


def page_metrics(pdf):
    fig = plt.figure(figsize=FIG)
    _title(fig, "Same probes, same splits, two metrics",
           "Only the scoring changed. known_probe moving 0.588 -> 0.767 is the demonstration.")
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.66])
    ax.axis("off")
    rows = [["predictor of frontier_mask", "balanced acc", "AUROC", "what it knows"]]
    rows.append(["MODEL, best layer", f"{MODEL_BAL:.3f}", f"{MODEL_AUC:.3f}", "residual stream"])
    whats = ["bookkeeping, as a 0/1 rule", "bookkeeping, fit as a probe",
             "bookkeeping + graph degree", "nothing (permuted)"]
    for (name, (bal, auc)), what in zip(BASELINES.items(), whats):
        rows.append([name, f"{bal:.3f}", f"{auc:.3f}", what])
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2.4)
    for j in range(4):
        tbl[0, j].set_facecolor("#dddddd")
        tbl[0, j].set_text_props(weight="bold")
    for j in range(4):
        tbl[1, j].set_facecolor("#ffe0e0")
        tbl[1, j].set_text_props(weight="bold")
    fig.text(0.08, 0.085,
             "A 0/1 rule's AUROC equals its balanced accuracy by construction (not_known, row 2),\n"
             "which is what keeps the hand baselines comparable across the metric change.\n"
             "Average precision agrees: model 0.47 vs degree_aug 0.46 vs base rate 0.149.",
             size=11.5, family="monospace", color="#333333")
    pdf.savefig(fig)
    plt.close(fig)


def page_development(pdf, probes, series):
    fig = plt.figure(figsize=FIG)
    _title(fig, "Present before competence, flat through it",
           "Peak-layer AUROC per snapshot, mean of 3 seeds. Shaded band = seed min/max.")
    ax = fig.add_axes([0.09, 0.19, 0.82, 0.61])
    ml = probes[probes["kind"] == "multilabel"]
    pk = ml.groupby(["target", "epoch", "seed"]).auroc.max().reset_index()
    for tgt, color in [("known_mask", "#2ca02c"), ("frontier_mask", "#d62728"),
                       ("remaining_mask", "#9467bd")]:
        g = pk[pk.target == tgt].groupby("epoch").auroc
        m, lo, hi = g.mean(), g.min(), g.max()
        ax.plot(m.index, m.values, "o-", color=color, lw=2.2, ms=5, label=tgt, rasterized=True)
        ax.fill_between(m.index, lo.values, hi.values, color=color, alpha=0.25, rasterized=True)
    cat = probes[probes["kind"] == "categorical"]
    cpk = cat.groupby(["target", "epoch"]).accuracy.max().reset_index()
    g = cpk[cpk.target == "next_color"]
    ax.plot(g.epoch, g.accuracy, "s--", color="#ff7f0e", lw=2.2, ms=5,
            label="next_color (top-1 acc)", rasterized=True)
    if series is not None and "solve_rate" in series:
        ax2 = ax.twinx()
        ax2.plot(series.epoch, series.solve_rate, "k:", lw=2.5, label="solve_rate", rasterized=True)
        ax2.set_ylabel("solve_rate", size=13)
        ax2.legend(loc="center left", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("epoch (log)", size=13)
    ax.set_ylabel("peak-layer AUROC  /  top-1 accuracy", size=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)
    ax.tick_params(labelsize=11)
    fig.text(0.09, 0.055,
             "frontier_mask is at 0.863 by epoch 5 (solve_rate 0.007) and 0.864 at epoch 900.\n"
             "The seed band is thinner than the line: this flatness is not a noise floor.",
             size=11.5, family="monospace", color="#333333")
    pdf.savefig(fig)
    plt.close(fig)


def page_layers(pdf, probes):
    fig = plt.figure(figsize=FIG)
    _title(fig, "Training moves the computation earlier, not higher",
           "frontier_mask AUROC by layer. The peak value barely changes; its position does.")
    ax = fig.add_axes([0.09, 0.19, 0.82, 0.61])
    f = probes[probes.target == "frontier_mask"]
    t = f.pivot_table(index="layer", columns="epoch", values="auroc", aggfunc="mean")
    for ep, color in [(1, "#cccccc"), (5, "#fdae61"), (50, "#74add1"), (900, "#b2182b")]:
        if ep in t:
            ax.plot(t.index, t[ep], "o-", color=color, lw=2.4, ms=5,
                    label=f"epoch {ep}", rasterized=True)
    ax.axhline(0.783, color="#1b5e20", ls="--", lw=1.8,
               label="best label-only baseline (degree_aug)")
    ax.set_xlabel("layer  (0 = token+position embedding, the control)", size=13)
    ax.set_ylabel("frontier_mask AUROC", size=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)
    ax.tick_params(labelsize=11)
    fig.text(0.09, 0.055,
             "Layer 0 sits at 0.676 at every epoch, so the +0.186 lift is computed, not read\n"
             "off the tokenization. Peak layer 8 (ep 5) -> 5 (ep 900); L1 0.708 -> 0.776.",
             size=11.5, family="monospace", color="#333333")
    pdf.savefig(fig)
    plt.close(fig)


def page_depth(pdf, depth):
    fig = plt.figure(figsize=FIG)
    _title(fig, "Depth costs solve_rate even at fixed hint density",
           "Against my pre-registered prediction of flat. rho pinned; bins differ only in T.")
    ax = fig.add_axes([0.08, 0.40, 0.40, 0.42])
    ax2 = fig.add_axes([0.57, 0.40, 0.40, 0.42])
    for ax_, ck in zip((ax, ax2), sorted(depth.checkpoint.unique())):
        for rho, color, style in [(0.75, "#1b5e20", "o-"), (0.60, "#0d47a1", "s-"),
                                  (0.50, "#999999", "^--")]:
            g = depth[(depth.checkpoint == ck) & (depth.rho_requested == rho)]
            if len(g):
                ax_.plot(g["T"], g.solve_rate, style, color=color, lw=2.2, ms=6,
                         label=f"rho={rho:.2f}", rasterized=True)
        ax_.set_title(ck.split("_")[0], size=13)
        ax_.set_xlabel("DD rounds T (reasoning depth)", size=12)
        ax_.set_ylabel("solve_rate", size=12)
        ax_.grid(alpha=0.3)
        ax_.legend(fontsize=10)
        ax_.tick_params(labelsize=10)
    fig.text(0.08, 0.30,
             "Why this is not the old rho sweep in disguise:", size=13, weight="bold")
    for i, line in enumerate([
        "  rho=0.75  density IDENTICAL in every bin (spread 0.0000)   r(solve,T) = -0.71 d0128 / -0.98 d0256",
        "  rho=0.60  realized spread 0.0012 over 24 nodes             r(solve,T) = -0.91 d0128 / -0.90 d0256",
        "  rho=0.50  spread 0.0123 AND r(realized_rho,T) = -0.82   <- DISCOUNT: the confound moved inside",
        "",
        "  Six of six (rho, checkpoint) cells negative. Single bin gaps are only ~1.5-2.5 sigma at",
        "  n = 58-698, so the consistency across cells is the evidence, not any one comparison.",
    ]):
        fig.text(0.08, 0.255 - i * 0.032, line, size=11, family="monospace", color="#333333")
    pdf.savefig(fig)
    plt.close(fig)


def page_whatnow(pdf):
    fig = plt.figure(figsize=FIG)
    _title(fig, "Where this leaves the question",
           "Two results that do not sit comfortably together, which is the useful part.")
    lines = [
        ("The tension", "#b00020"),
        ("   The frontier is represented from epoch 5 and never changes (page 3), yet", "k"),
        ("   solve_rate falls with reasoning depth at pinned density (page 5). A model", "k"),
        ("   with a static, already-good frontier representation should not care how", "k"),
        ("   many DD rounds an instance needs - unless the cost is somewhere else.", "k"),
        ("", "k"),
        ("Candidate, and how to test it", "#0d47a1"),
        ("   The cost is in CHAINING COLORS, not in locating the frontier. next_color is", "k"),
        ("   the one target that develops with competence (0.327 -> 0.675), so a depth", "k"),
        ("   cost should show up as next_color accuracy falling with position-within-T,", "k"),
        ("   while frontier_mask AUROC stays flat across the same positions.", "k"),
        ("", "k"),
        ("Training-side status, same eval throughout", "#1b5e20"),
        ("   best agg_solve_rate  0.847  d0256 @300ep        (unchanged, still leads)", "k"),
        ("   epochs 300 -> 900    0.817  predicted not to help; it hurt", "k"),
        ("                               but raised solve_rate_k0 0.24 -> 0.87", "k"),
        ("   n_train 5000->20000  running", "k"),
        ("   d_model 256 -> 384   running", "k"),
        ("", "k"),
        ("Caveats", "#b00020"),
        ("   One training trajectory; the 3 seeds vary the PROBE, not the training run.", "k"),
        ("   Depth bins come from one instance pool per density, one seed each.", "k"),
    ]
    y = 0.79
    for line, color in lines:
        weight = "bold" if color != "k" and not line.startswith("  ") else "normal"
        fig.text(0.08, y, line, size=12.5, family="monospace", color=color, weight=weight)
        y -= 0.034
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    probes, depth, series = load()
    plt.rcParams["savefig.dpi"] = 150
    with PdfPages(args.out) as pdf:
        page_headline(pdf)
        page_metrics(pdf)
        page_development(pdf, probes, series)
        page_layers(pdf, probes)
        page_depth(pdf, depth)
        page_whatnow(pdf)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.3f} MB)")


if __name__ == "__main__":
    main()
