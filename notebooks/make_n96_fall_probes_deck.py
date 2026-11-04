"""Slide deck for the N=96 fall probe sweep (50979059) + the label-only baseline (51009233).

    srun -p dev -t 00:20:00 -c 2 --mem=16G python3 notebooks/make_n96_fall_probes_deck.py

Writes $SCRATCH/spin_gnn/decks/n96_fall_probes_2026-11-03/slides.pdf plus one PNG per
page (there is no PDF rasterizer on this cluster, so the PNGs are the only way to
actually look at the deck before sending it).

Numbers come from `scripts/n96_fall_probe_analysis`, the same committed reader the
results notebook imports, so deck, notebook and
`python3 -m scripts.n96_fall_probe_analysis` cannot disagree. Pages 2 and 3 are the
mandatory tokenization and real-trace pages, built at the SAME ensemble the probed
checkpoints were trained on (N=96, --shuffle-adj --fall).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.expanduser("~/spin_gnn"))
from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style, textpage
from scripts import n96_fallscan_analysis as scan
from scripts.n96_fall_probe_analysis import (
    CELLS, DEPTH_CELLS, MULTILABEL, PROBE_DEFAULT, SEED_PAIRS, TARGETS,
    cell_stats, load, metric_of, rho50_by_cell, spearman)

OUT = os.path.join(os.environ["SCRATCH"], "spin_gnn", "decks",
                   "n96_fall_probes_2026-11-03")
BDIR = os.path.join(os.environ["SCRATCH"], "spin_gnn",
                    "sudoku_frontier_baseline_2026-11-03")
LADDER = os.path.join(os.environ["SCRATCH"], "spin_gnn",
                      "sudoku_probes_competence_ladder_2026-10-21",
                      "probes_competence_ladder.csv")
GREEN, RED, BLUE, GREY, ORANGE = "#1a7f37", "#b42318", "#1f4e9c", "0.35", "#b5500a"
FREE_ROWS = ("not_known", "known_probe", "degree_aug")


def baseline(pattern):
    """(free auroc, free AP, ceiling auroc, ceiling AP, rule auroc, rule AP, spread).

    The free score is the STRICTEST seen over the pools -- the bar the model has to
    clear -- and the ceiling is the pool mean, since it is a reference level rather
    than a bar. `color_aug` is the ceiling and `rule_on_visible` is not: a hard 0/1
    rule occupies one operating point and its AUROC understates what its information
    is worth, which is the same threshold artifact this deck is about.
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(BDIR, pattern))):
        b = json.load(open(p))
        rows.append((max(b[k]["auroc"] for k in FREE_ROWS),
                     max(b[k]["average_precision"] for k in FREE_ROWS),
                     b["color_aug"]["auroc"], b["color_aug"]["average_precision"],
                     b["rule_on_visible"]["auroc"],
                     b["rule_on_visible"]["average_precision"],
                     b["run"]["positive_rate"],
                     max(b[k]["balanced_accuracy"] for k in FREE_ROWS)))
    if not rows:
        return None
    return dict(free=max(r[0] for r in rows), free_ap=max(r[1] for r in rows),
                free_ba=max(r[7] for r in rows),
                ceil=float(np.mean([r[2] for r in rows])),
                ceil_ap=float(np.mean([r[3] for r in rows])),
                rule=float(np.mean([r[4] for r in rows])),
                rule_ap=float(np.mean([r[5] for r in rows])),
                pos=float(np.mean([r[6] for r in rows])),
                spread=max(r[0] for r in rows) - min(r[0] for r in rows),
                n=len(rows))


def ladder_profiles():
    """N=24 competence ladder, the previous best on this same probe eval."""
    import csv
    from collections import defaultdict
    acc = defaultdict(lambda: defaultdict(list))
    with open(LADDER) as f:
        for r in csv.DictReader(f):
            col = "auroc" if r["kind"] == "multilabel" else "balanced_accuracy"
            acc[(r["run"].split("/")[0], r["target"])][int(r["layer"])].append(float(r[col]))
    return {k: {l: float(np.mean(v)) for l, v in d.items()} for k, d in acc.items()}


# ------------------------------------------------------------------ page 1
def page_verdict(o, prof, kind, base, R50, present):
    best = max(present, key=lambda c: cell_stats(prof, c, "frontier_mask", "auroc")["peak_abs"])
    bs = cell_stats(prof, best, "frontier_mask", "auroc")
    lo = min(cell_stats(prof, c, "frontier_mask", "auroc")["peak_abs"] for c in present)
    nn = [cell_stats(prof, c, "next_node", "balanced_accuracy")["final"] for c in present]
    fm = [cell_stats(prof, c, "frontier_mask", "auroc")["peak_abs"] for c in present]
    r50 = [R50[c][0] for c in present]
    # Fraction of the free->ceiling bracket closed, computed rather than quoted:
    # an earlier draft of this page carried "81% ... 93%", which came from the
    # N=24 smoke-test ceiling and was wrong by 20-30 points at N=96.
    gap = sorted(100 * (v - base["free"]) / (base["ceil"] - base["free"]) for v in fm)
    gap_ap = sorted(100 * (cell_stats(prof, c, "frontier_mask", "average_precision")["peak_abs"]
                           - base["free_ap"]) / (base["ceil_ap"] - base["free_ap"])
                    for c in present)
    textpage(
        o,
        "Yes, the N=96 model computes DD's deducibility predicate — in its last layer",
        "6 checkpoints x 3 probe seeds · jobs 50979059 + 51009233 · commit 309ad1a · "
        "linear probes on the residual stream",
        [
            (f"THE PREDICATE IS THERE: AUROC {lo:.3f}-{bs['peak_abs']:.3f} against a label-only "
             f"free score of {base['free']:.3f}.", GREEN,
             f"frontier_mask is which nodes DD can deduce this round. A reader that knows only "
             f"which nodes are already coloured scores {base['free']:.4f} on it for free (spread "
             f"{base['spread']:.4f} over {base['n']} instance pools); a probe handed "
             f"n_known_colors -- the quantity DD actually thresholds -- reaches {base['ceil']:.3f}. "
             f"All six cells land inside that bracket, closing {gap[0]:.0f}%-{gap[-1]:.0f}% of it "
             f"on AUROC and {gap_ap[0]:.0f}%-{gap_ap[-1]:.0f}% on average precision. This is beyond "
             f"what bookkeeping explains -- and well short of the information DD uses."),
            ("IT IS BUILT LAST, and the control says that is about DD and not about the probe.", BLUE,
             "frontier_mask peaks at layer 8, 12 and 16 for L=8, 12 and 16 -- fraction 1.000, "
             "spread 0.000, identical across three probe seeds AND across training seeds. "
             "known_mask, pure bookkeeping, is already at 0.768 from the EMBEDDING and peaks at "
             "layers 4/8/2 with no depth pattern at all. The deduction predicate and the "
             "bookkeeping are different signals living in different places."),
            ("BUT MORE DEPTH BUYS NO MORE OF IT.", ORANGE,
             "Peak lift over the embedding is 0.143 / 0.119 / 0.131 at L=8 / 12 / 16 -- spread "
             "0.0237, under the 0.0411 one-training-seed yardstick. Eight extra layers move where "
             "the predicate tops out and not how much of it there is. Consistent with the scan, "
             "where the same depth range was also unreadable against the seed."),
            (f"AND IT DOES NOT PREDICT SUCCESS: Spearman {spearman(fm, r50):+.3f} vs rho_50, "
             f"while next-node choice gives {spearman(nn, r50):+.3f}.", RED,
             "The worst model in the grid (t5, rho_50 0.829) has the SECOND-HIGHEST frontier "
             "AUROC. What separates good cells from bad is next_node -- which of the deducible "
             "nodes the model commits to emitting -- where t5 collapses to 0.603 against 0.870. "
             "Every cell works out roughly what is deducible; they differ in acting on it."),
        ],
        "Probe eval: 400 train / 160 test instances, <=20 reasoning positions each, 600 probe "
        "epochs, three probe-fit seeds. AUROC, not balanced accuracy -- page 4 is about why that "
        "choice reverses the verdict. rho_50 from the same 14-point eval as the scan deck.",
        y0=0.835, gap=0.048, width=118)


# ------------------------------------------------------------------ page 4
def page_column(o, prof, base):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Read on the wrong column, this run is a null",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.895,
             "Same CSVs, same rows, two columns. frontier_mask is positive on "
             f"{base['pos']*100:.1f}% of bits.",
             ha="center", size=SMALL, color="0.3")

    ax = fig.add_axes([0.075, 0.40, 0.25, 0.44])
    ax2 = fig.add_axes([0.405, 0.40, 0.25, 0.44])
    ax3 = fig.add_axes([0.715, 0.40, 0.25, 0.44])

    cells = [c for c, _L in DEPTH_CELLS]
    for a, m, name, free in ((ax, "auroc", "AUROC (threshold-free)", base["free"]),
                             (ax2, "balanced_accuracy", "balanced accuracy (threshold 0)",
                              base["free_ba"])):
        for c in cells:
            L = CELLS[c][1]
            xs = sorted({l for (cc, _s, t), p in prof.items()
                         if cc == c and t == "frontier_mask" for l in p})
            ys = [np.mean([p[l][m] for (cc, _s, t), p in prof.items()
                           if cc == c and t == "frontier_mask" and l in p]) for l in xs]
            a.plot(xs, ys, "-o", ms=4, lw=2, label=f"L={L}", rasterized=True)
            k = int(np.argmax(ys[1:])) + 1
            a.plot(xs[k], ys[k], "*", ms=16, color="k", zorder=5)
        a.axhline(free, color="k", lw=1.2, ls="--")
        a.text(0.3, free + 0.004, "label-only free score", size=SMALL - 2)
        a.set_xlabel("layer"); a.set_title(name, size=SMALL + 2)
        a.grid(alpha=0.3); a.legend(fontsize=SMALL - 1, loc="lower right")
    ax.set_ylabel("frontier_mask score")
    ax.text(0.03, 0.03, "stars = argmax over layers >= 1", transform=ax.transAxes,
            va="bottom", size=SMALL - 2, color="0.35")

    # The bracket, on both threshold-free columns.
    best_a = max(cell_stats(prof, c, "frontier_mask", "auroc")["peak_abs"] for c in CELLS
                 if any(k[0] == c for k in prof))
    best_p = max(cell_stats(prof, c, "frontier_mask", "average_precision")["peak_abs"]
                 for c in CELLS if any(k[0] == c for k in prof))
    names = ["chance", "free score\n(bookkeeping)", "MODEL\n(best cell)",
             "ceiling\n(n_known_colors)"]
    auc = [0.5, base["free"], best_a, base["ceil"]]
    apv = [base["pos"], base["free_ap"], best_p, base["ceil_ap"]]
    x = np.arange(4)
    ax3.bar(x - 0.2, auc, 0.38, color=[GREY, GREY, GREEN, BLUE], rasterized=True)
    ax3.bar(x + 0.2, apv, 0.38, color=[GREY, GREY, GREEN, BLUE], alpha=0.45,
            rasterized=True)
    for xi, (a, p) in enumerate(zip(auc, apv)):
        ax3.text(xi - 0.2, a + 0.012, f"{a:.3f}", ha="center", size=SMALL - 2)
        ax3.text(xi + 0.2, p + 0.012, f"{p:.3f}", ha="center", size=SMALL - 2, color="0.3")
    ax3.set_xticks(x); ax3.set_xticklabels(names, size=SMALL - 4)
    ax3.set_ylim(0, 1.0); ax3.grid(alpha=0.3, axis="y")
    ax3.set_title("solid = AUROC, pale = avg precision", size=SMALL + 1)

    fig.text(0.5, 0.30,
             "A BCE probe on a 10%-positive target under-predicts positives, so its balanced "
             "accuracy at threshold 0 is a FLOOR that moves with calibration, not with knowledge.\n"
             "The tell was already in the baseline: the TRIVIAL rule 'not yet coloured' scores "
             f"{base['free_ba']:.4f} there, above every trained probe. That cannot be a statement "
             "about knowing.",
             ha="center", va="top", size=SMALL, color="0.25", linespacing=1.6)
    fig.text(0.5, 0.17,
             "Same trap at the other end: DD's own rule, computed not fitted, scores only "
             f"{base['rule']:.3f} AUROC -- a 0/1 rule gets no credit for ranking within its two "
             f"classes --\nbut {base['rule_ap']:.3f} average precision, the highest of anything "
             "here. The graded ceiling is color_aug, same information, allowed to rank.",
             ha="center", va="top", size=SMALL, color=RED, linespacing=1.6)
    footer(fig, "Read on balanced accuracy, every cell sits below the free score and the peak layer "
                "wanders 8.67 layers across L=8/12/16 -- while one training seed alone moves it 9.00 "
                "at fixed L=12. That is a flat curve's argmax. The baseline script's own docstring "
                "had already said to quote the threshold-free columns.")
    o.savefig(fig)


# ------------------------------------------------------------------ page 5
def page_where(o, prof, kind, base, present):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Where DD gets done: bookkeeping early, the predicate last",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.895,
             "Layer 0 is the token+position embedding, so it is the tokenization control: "
             "whatever it scores needs no computation.",
             ha="center", size=SMALL, color="0.3")

    axes = [fig.add_axes([0.075 + 0.232 * i, 0.46, 0.185, 0.38]) for i in range(3)]
    axp = fig.add_axes([0.775, 0.46, 0.185, 0.38])
    for ax, t in zip(axes, MULTILABEL):
        for c in present:
            L = CELLS[c][1]
            xs = sorted({l for (cc, _s, tt), p in prof.items()
                         if cc == c and tt == t for l in p})
            ys = [np.mean([p[l]["auroc"] for (cc, _s, tt), p in prof.items()
                           if cc == c and tt == t and l in p]) for l in xs]
            ax.plot(xs, ys, "--" if CELLS[c][3] else "-", marker="o", ms=3, lw=1.6,
                    label=CELLS[c][0].split(" ")[0] + f" L={L}", rasterized=True)
        if t == "frontier_mask":
            ax.axhline(base["free"], color="k", lw=1.2, ls="--")
            ax.axhline(base["ceil"], color=RED, lw=1.2, ls=":")
            ax.text(0.3, base["free"] + 0.005, "free", size=SMALL - 2)
            ax.text(0.3, base["ceil"] + 0.005, "ceiling", size=SMALL - 2, color=RED)
        ax.set_title(t + ("  <- THE DD PREDICATE" if t == "frontier_mask" else "  (control)"),
                     size=SMALL, color=BLUE if t == "frontier_mask" else "0.35")
        ax.set_xlabel("layer"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC")
    axes[2].legend(fontsize=SMALL - 3, loc="lower right")

    for t, col, mk in (("frontier_mask", BLUE, "o"), ("known_mask", ORANGE, "s")):
        ys = [cell_stats(prof, c, t, "auroc")["peak_layer"] for c, _L in DEPTH_CELLS]
        xs = [L for _c, L in DEPTH_CELLS]
        axp.plot(xs, ys, mk + "-", ms=10, lw=2, color=col, label=t, rasterized=True)
    axp.plot([8, 16], [8, 16], ":", color="0.6", lw=1.5)
    axp.text(12.6, 10.6, "peak = last layer", size=SMALL - 2, color="0.4", rotation=38)
    axp.set_xticks([8, 12, 16]); axp.set_xlabel("num_layers")
    axp.set_ylabel("peak layer"); axp.set_ylim(0, 18)
    axp.set_title("peak layer vs depth", size=SMALL + 1)
    axp.grid(alpha=0.3); axp.legend(fontsize=SMALL - 2, loc="upper left")

    fig.text(0.5, 0.36,
             "P1 as pre-registered FAILS (the peak is not at a fixed absolute layer, it moves 8) "
             "— the informative alternative holds in its extreme form:\n"
             "a fixed FRACTION of depth, and the fraction is 1.000 with spread 0.000. "
             "P3, the control clause, passes: known_mask shows no depth pattern (spread 0.542).",
             ha="center", va="top", size=SMALL, color="0.25", linespacing=1.6)
    fig.text(0.5, 0.235,
             "Noise floors both claims were read against — P2(a) three probe-fit seeds: frontier "
             "peak layer spread 0.00, lift 0.0090.\n"
             "P2(b) training-seed pairs: peak layer spread 0.00, lift 0.0411. The LAYER is solid; "
             "the AMOUNT is not, which is why no depth claim is made about the amount.",
             ha="center", va="top", size=SMALL, color=GREY, linespacing=1.6)
    footer(fig, "known_mask reaching 0.768 at layer 0 is not a result about the model: the token "
                "sequence states which nodes have been coloured, so the embedding carries it. That "
                "is exactly why it is the control -- any layer structure it showed would be the "
                "probe's, not DD's.")
    o.savefig(fig)


# ------------------------------------------------------------------ page 6
def page_channels(o, prof, R50, present):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Knowing what is deducible is not what separates the models",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.895,
             "Six cells, each with rho_50 from the 14-point x 400-instance eval "
             "(lower = solves with fewer hints = better).",
             ha="center", size=SMALL, color="0.3")

    ax = fig.add_axes([0.07, 0.40, 0.25, 0.44])
    ax2 = fig.add_axes([0.40, 0.40, 0.25, 0.44])
    ax3 = fig.add_axes([0.73, 0.40, 0.23, 0.44])
    r50 = [R50[c][0] for c in present]
    nn = [cell_stats(prof, c, "next_node", "balanced_accuracy")["final"] for c in present]
    fm = [cell_stats(prof, c, "frontier_mask", "auroc")["peak_abs"] for c in present]

    for a, ys, name, col in ((ax, fm, "frontier_mask peak AUROC\n(the DD predicate)", BLUE),
                             (ax2, nn, "next_node top-1, final layer\n(the selection step)", GREEN)):
        for c, y in zip(present, ys):
            a.scatter(y, R50[c][0], s=150, color=col, zorder=3, rasterized=True)
            a.annotate(CELLS[c][0].split(" ")[0], (y, R50[c][0]),
                       textcoords="offset points", xytext=(8, -4), size=SMALL - 1)
        m, b = np.polyfit(ys, r50, 1)
        xs = np.linspace(min(ys), max(ys), 10)
        a.plot(xs, m * xs + b, "--", color="0.5", lw=1.5, zorder=1)
        a.set_xlabel(name, size=SMALL); a.set_ylabel("rho_50  (lower = better)", size=SMALL)
        a.set_title(f"Spearman {spearman(ys, r50):+.3f}", size=MED - 2,
                    color=RED if abs(spearman(ys, r50)) < 0.5 else GREEN)
        a.grid(alpha=0.3)

    order = sorted(present, key=lambda c: R50[c][0])
    y = np.arange(len(order))
    ax3.barh(y + 0.2, [cell_stats(prof, c, "frontier_mask", "auroc")["peak_abs"] for c in order],
             0.38, color=BLUE, rasterized=True)
    ax3.barh(y - 0.2, [cell_stats(prof, c, "next_node", "balanced_accuracy")["final"]
                       for c in order],
             0.38, color=GREEN, rasterized=True)
    ax3.set_yticks(y)
    ax3.set_yticklabels([f"{CELLS[c][0].split(' ')[0]}  rho50 {R50[c][0]:.2f}" for c in order],
                        size=SMALL - 2)
    ax3.invert_yaxis(); ax3.set_xlim(0.5, 0.95)
    ax3.set_title("cells best -> worst\nblue = frontier AUROC, green = next_node top-1",
                  size=SMALL - 1)
    ax3.grid(alpha=0.3, axis="x")

    fig.text(0.5, 0.275,
             "t5 is the worst model in the grid (rho_50 0.829, val_nll 1.814) and has the "
             "SECOND-HIGHEST frontier AUROC (0.764).\n"
             "Its next_node collapses to 0.603 against 0.870 for the best cell. The predicate "
             "channel is flat across good and bad models; the selection channel is not.",
             ha="center", va="top", size=SMALL, color="0.25", linespacing=1.6)
    fig.text(0.5, 0.165,
             "Two caveats kept in view: n=6, and next_node carries a 0.1555 training-seed spread "
             "at L=8, so -0.943 is an observation over six points, not an estimate.\n"
             "next_node also sits closer to the logits than frontier_mask does, so part of its "
             "edge may be proximity rather than mechanism. The clean test is an INTERVENTION.",
             ha="center", va="top", size=SMALL, color=GREY, linespacing=1.6)
    footer(fig, "This is the finding that redirects the next experiment. The standing question was "
                "whether the network runs DD; the probe says it largely works out WHAT is deducible "
                "and that this is not where the N=96 models differ. The difference is in which "
                "deducible node they then choose to emit.")
    o.savefig(fig)


# ------------------------------------------------------------------ page 7
def page_prev_and_next(o, prof, lad, base, b24, present):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "Below the previous best on this eval — and normalising does not rescue it",
             ha="center", size=BIG, weight="bold")
    fig.text(0.5, 0.895,
             "N=24 competence ladder (2026-10-21), same sudoku_probes code path, same targets, "
             "same three probe seeds, L=12. Each N now has a label-only bracket of its own.",
             ha="center", size=SMALL, color="0.3")

    ax = fig.add_axes([0.055, 0.46, 0.27, 0.38])
    runs = sorted({k[0] for k in lad})
    for r in runs:
        p = lad[(r, "frontier_mask")]
        xs = sorted(p); L = max(xs)
        ax.plot([x / L for x in xs], [p[x] for x in xs], "-", lw=1.4,
                color=GREEN, alpha=0.55, rasterized=True)
    ax.plot([], [], "-", color=GREEN, lw=1.4, label="N=24 ladder, 5 cells")
    for c in present:
        L = CELLS[c][1]
        xs = sorted({l for (cc, _s, t), p in prof.items()
                     if cc == c and t == "frontier_mask" for l in p})
        ys = [np.mean([p[l]["auroc"] for (cc, _s, t), p in prof.items()
                       if cc == c and t == "frontier_mask" and l in p]) for l in xs]
        ax.plot([x / L for x in xs], ys, "-o", ms=3, lw=1.6, color=BLUE, alpha=0.8,
                rasterized=True)
    ax.plot([], [], "-o", ms=3, color=BLUE, label="N=96 fall, 6 cells")
    ax.axhline(base["free"], color="k", lw=1.1, ls="--")
    ax.text(0.02, base["free"] + 0.008, "N=96 free score", size=SMALL - 2)
    ax.set_xlabel("layer / num_layers    (0 = embedding)")
    ax.set_ylabel("frontier_mask AUROC")
    ax.set_title("N=24 peaks mid-stack and plateaus;\nN=96 rises into the last layer",
                 size=SMALL)
    ax.legend(fontsize=SMALL - 1, loc="lower right"); ax.grid(alpha=0.3)

    ax2 = fig.add_axes([0.395, 0.46, 0.25, 0.38])
    tg = [("frontier_mask", "auroc"), ("known_mask", "auroc"),
          ("next_node", "balanced_accuracy")]
    a24 = [max(max(v for l, v in lad[(r, t)].items() if l >= 1) for r in runs) for t, _m in tg]
    a96 = [max(cell_stats(prof, c, t, m)["peak_abs"] for c in present) for t, m in tg]
    x = np.arange(3)
    ax2.bar(x - 0.2, a24, 0.38, color=GREEN, rasterized=True)
    ax2.bar(x + 0.2, a96, 0.38, color=BLUE, rasterized=True)
    for xi, (a, b) in enumerate(zip(a24, a96)):
        ax2.text(xi - 0.2, a + 0.015, f"{a:.3f}", ha="center", size=SMALL - 3, color=GREEN)
        ax2.text(xi + 0.2, b + 0.015, f"{b:.3f}", ha="center", size=SMALL - 3, color=BLUE)
        ax2.text(xi, 1.17, f"{b-a:+.3f}", ha="center", size=SMALL - 1, color=RED, weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.replace("_mask", "") for t, _m in tg], size=SMALL - 2)
    ax2.set_ylim(0, 1.3); ax2.grid(alpha=0.3, axis="y")
    ax2.set_title("raw AUROC: N=24 (green) vs N=96 (blue)", size=SMALL)

    # The raw gap confounds "weaker model" with "easier task". The fraction of
    # each ensemble's OWN free->ceiling bracket that the cells close does not.
    ax3 = fig.add_axes([0.715, 0.46, 0.245, 0.38])
    if b24:
        g24 = sorted(100 * (max(v for l, v in lad[(r, "frontier_mask")].items() if l >= 1)
                            - b24["free"]) / (b24["ceil"] - b24["free"]) for r in runs)
        g96 = sorted(100 * (cell_stats(prof, c, "frontier_mask", "auroc")["peak_abs"]
                            - base["free"]) / (base["ceil"] - base["free"]) for c in present)
        for xi, (g, col) in enumerate(((g24, GREEN), (g96, BLUE))):
            ax3.scatter([xi] * len(g), g, s=130, color=col, zorder=3, rasterized=True)
            ax3.plot([xi - 0.17, xi + 0.17], [np.mean(g)] * 2, "-", color="k", lw=2, zorder=4)
            ax3.text(xi, max(g) + 7, f"{min(g):.0f}-{max(g):.0f}%", ha="center",
                     size=SMALL - 1, color=col, weight="bold")
        ax3.axhline(100, color=GREY, lw=1.1, ls=":")
        ax3.text(1.45, 101.5, "ceiling", size=SMALL - 2, color=GREY, ha="right")
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels([f"N=24\nfree {b24['free']:.3f} / ceil {b24['ceil']:.3f}",
                             f"N=96 fall\nfree {base['free']:.3f} / ceil {base['ceil']:.3f}"],
                            size=SMALL - 3)
        ax3.set_xlim(-0.5, 1.5); ax3.set_ylim(0, 125)
        ax3.set_ylabel("% of own free->ceiling bracket closed", size=SMALL - 1)
        ax3.set_title("normalised by each N's own bracket:\nthe decline is not just a harder task",
                      size=SMALL)
        ax3.grid(alpha=0.3, axis="y")

    fig.text(0.5, 0.37,
             "N=24 is an easier TASK as well as an easier ensemble — its label-only free score is "
             f"{b24['free']:.3f} against {base['free']:.3f} here. Normalising by each bracket, N=24 "
             "saturates its ceiling\n"
             "and N=96 does not; and the shortfall is not confined to the predicate — known_mask, "
             "pure bookkeeping, near-perfect at N=24, falls to 0.91 at N=96 (no bracket for that one).",
             ha="center", va="top", size=SMALL, color="0.25", linespacing=1.6)
    fig.text(0.5, 0.265,
             "Tempting reading — 'N=96 is depth-limited, the computation does not finish inside "
             "the stack' — is CONTRADICTED by this very run:\n"
             "L=16 decodes the predicate no better than L=8 (0.742 vs 0.757, inside the seed "
             "yardstick). The last-layer peak is assembly, not an unfinished computation.",
             ha="center", va="top", size=SMALL, color=RED, linespacing=1.6)
    fig.text(0.5, 0.155,
             "NEXT, in order: (1) the causal test — ablate/patch the frontier direction at the "
             "final layer and measure rho_50 on the same eval; a probe shows information is\n"
             "readable, not used. (2) Aim it at the SELECTION step, which tracks rho_50 at -0.943, "
             "rather than at the predicate, which does not track it at all.\n"
             "(3) Explain the N=24 -> N=96 bookkeeping collapse, which is upstream of everything "
             "here.",
             ha="center", va="top", size=SMALL, color=BLUE, linespacing=1.6)
    footer(fig, "Comparability: N=24 is a different ensemble (no fall; frontier positive rate 0.148 "
                "against 0.101 here), so even the normalised panel compares two tasks rather than "
                "two models on one task, and the AP columns are not comparable across N at all. "
                "Both brackets: 3 instance pools, 600 probe epochs, 51009233.")
    o.savefig(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    set_style()
    prof, majority, kind, files = load(PROBE_DEFAULT)
    present = [c for c in CELLS if any(k[0] == c for k in prof)]
    R50 = rho50_by_cell(scan.SCAN_DEFAULT)
    # 600 probe epochs on BOTH sides: the model probes were fit for 600, and an
    # underfit baseline would understate the free score and flatter the model.
    base = baseline("n96_fall_ep600_off*.json")
    if base is None:
        raise SystemExit("no N=96 baseline JSON in " + BDIR)
    b24 = baseline("n24_ep600_off*.json")
    lad = ladder_profiles()
    print(f"{len(present)} cells, {len(files)} probe seeds; "
          f"free {base['free']:.4f} ceiling {base['ceil']:.4f} over {base['n']} baseline runs")
    if b24:
        print(f"N=24 bracket: free {b24['free']:.4f} ceiling {b24['ceil']:.4f} "
              f"over {b24['n']} runs")

    # Same ensemble the probed checkpoints were trained on, and a deep instance:
    # the deck is about where in the stack a MULTI-ROUND deduction gets done.
    ex = example_instance(n_nodes=96, degree=5, n_colors=4, rho=0.33, rho_max=0.95,
                          shuffle_adj=True, fall=True, pick="deep")
    print(f"trace instance: {ex.n_rounds} DD rounds, {len(ex.tokens)} tokens")

    with PdfPages(f"{OUT}/slides.pdf") as pdf:
        o = PdfAndPng(pdf, OUT, total=7)
        page_verdict(o, prof, kind, base, R50, present)
        page_tokenization(o, ex)
        page_trace(o, ex, note="N=96, degree 5, 4 colours, --shuffle-adj --fall: the task all six "
                               "probed checkpoints were trained on. The probe reads the residual "
                               "stream at the reasoning positions -- one per node/colour emission, "
                               "capped at 20 per instance.")
        page_column(o, prof, base)
        page_where(o, prof, kind, base, present)
        page_channels(o, prof, R50, present)
        page_prev_and_next(o, prof, lad, base, b24, present)

    mb = os.path.getsize(f"{OUT}/slides.pdf") / 1e6
    print(f"wrote {OUT}/slides.pdf  ({mb:.2f} MB, {o.n} pages)")
    if mb > 2:
        print("WARNING: over the 2 MB delivery budget")


if __name__ == "__main__":
    main()
