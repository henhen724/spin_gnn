"""Results deck for 49655447: what are these networks actually fitting?

Finished 2026-10-26, ExitCode 0:0, 00:18:42. Eval-only, no training.

PROVENANCE. spin_gnn/ package at commit e25dcf3 for the run itself (the leak fix
in 5103e83 came after and does not affect these numbers). seed=0 for every
checkpoint scored; --seed 0 for the permutations. Part A uses n=600 per side,
part B n=400, held-out instances at seed_start=50000000000 in both.

The three part-B checkpoints:
  N=24 record  $OAK/.../sudoku_dd_d0256_L12_ntrain80000_ep300_sharedpool0.9960_job48912548
  N=96 cell B  sudoku_n96_plateau_2026-10-26/d0256_h04_L12_ff01024_lr1e-03_pre  (80000)
  N=48 best    sudoku_ff_depth_tradeoff_2026-10-25_n48/d0256_h04_L08_ff00256_lr1e-03_pre
Part A instead spans the data budget: N=48 L=12 @20000, N=96 control @20000,
N=96 cell B @80000.

COMPARISON TO PREVIOUS BEST ON THE SAME EVAL. There is no previous measurement
of either quantity in this project -- no run has ever been scored on its own
training instances, and no run has ever been scored under a permuted adjacency.
The reference these numbers displace is the SOLVE_RATE LADDER itself (0.9960 at
N=24, 0.9573 at N=48, 0.9273 at N=96), which page 6 addresses directly.

Run as: python -m notebooks.make_leak_deck
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

ROOT = os.environ.get("SCRATCH", "") + "/spin_gnn"
SRC = os.path.join(ROOT, "sudoku_memorization_2026-10-26")
OUT = os.path.join(ROOT, "notebooks", "sudoku_leak_2026-10-26")
PDF = os.path.join(OUT, "encoding_leak.pdf")

GREY, BLUE, RED, GREEN = "0.45", "#1f77b4", "#d62728", "#2ca02c"
ORDER = ["identity", "within_block", "shuffle", "byid"]
# Short labels: the full names overrun a 4-tick axis at MED.
OLAB = {"identity": "identity\n(as trained)", "within_block": "within_block\n(control)",
        "shuffle": "shuffle", "byid": "byid"}


def load(name):
    with open(os.path.join(SRC, name)) as f:
        return list(csv.DictReader(f))


def fnum(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


def table(ax, header, rows, colw, fontsize=SMALL, row_colors=None, dy=0.115):
    # dy is explicit because page 3 carries 9 rows: at the 0.115 default they run
    # off the axes and overprint the footer.
    ax.axis("off")
    x0 = 0.0
    for j, h in enumerate(header):
        ax.text(x0 + sum(colw[:j]), 0.92, h, size=fontsize, weight="bold", va="top")
    for i, row in enumerate(rows):
        y = 0.92 - dy * (i + 1)
        c = (row_colors or {}).get(i, "black")
        for j, cell in enumerate(row):
            ax.text(x0 + sum(colw[:j]), y, cell, size=fontsize, va="top", color=c,
                    weight="bold" if c != "black" else "normal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


ordr = load("ordering_counterfactual.csv")
memr = load("memorization_gap.csv")

# Part B rows keyed by (N, L, ff) -> {ordering: row}. The N=24 record and the two
# others all carry the same tag shape, so the run_dir is what distinguishes them.
models = []
for r in ordr:
    key = (int(r["n_nodes"]), int(r["num_layers"]), int(r["dim_feedforward"]))
    if key not in [m[0] for m in models]:
        models.append((key, {}))
    dict(models)[key][r["ordering"]] = r
models = [(k, v) for k, v in models]
LABEL = {(24, 12, 1024): "N=24 record\nL=12 ff=1024  0.9960",
         (96, 12, 1024): "N=96 cell B\nL=12 ff=1024  0.9273",
         (48, 8, 256): "N=48 best\nL=8 ff=256  0.9573"}

set_style()
os.makedirs(OUT, exist_ok=True)
with PdfPages(PDF) as _pdf:
    pdf = PdfAndPng(_pdf, OUT)

    # ---------------------------------------------------------------- page 1
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.80, "The networks were not solving graph colouring",
             ha="center", size=BIG + 7, weight="bold")
    fig.text(0.5, 0.685, "They were reading the planted colours off the token order",
             ha="center", size=BIG - 1, color=RED)
    fig.text(0.5, 0.52,
             "Permuting each node's neighbour list -- same edges, same colouring, same\n"
             "hints, same DD trace, all asserted equal -- drops every checkpoint in\n"
             "the project from 0.94-1.00 to 0.00-0.15.",
             ha="center", size=MED + 5, linespacing=1.7)
    fig.text(0.5, 0.30, "job 49655447   ·   eval-only, 18 min   ·   2026-10-26",
             ha="center", size=MED, color=GREY)
    fig.text(0.5, 0.21,
             "The in-distribution control (permute only within a colour group) ties "
             "identity on all three models,\nso the collapse is the colour code, not distribution shift.",
             ha="center", size=SMALL + 2, color=GREY, linespacing=1.6)
    pdf.savefig(fig)

    # ---------------------------------------------------------------- page 2
    fig, ax = plt.subplots(figsize=(16, 9))
    width = 0.2
    xs = np.arange(len(ORDER))
    cols = {"identity": BLUE, "within_block": GREEN, "shuffle": RED, "byid": "#7f2704"}
    for i, (key, byo) in enumerate(models):
        vals = [fnum(byo[o], "heldout_solve_rate") for o in ORDER]
        los = [v - fnum(byo[o], "heldout_wilson_lo") for v, o in zip(vals, ORDER)]
        his = [fnum(byo[o], "heldout_wilson_hi") - v for v, o in zip(vals, ORDER)]
        ax.bar(xs + (i - 1) * width, vals, width, yerr=[los, his], capsize=3,
               color=[cols[o] for o in ORDER], edgecolor="black", linewidth=0.8,
               alpha=0.95 if i == 0 else (0.75 if i == 1 else 0.55))
        for x, v in zip(xs + (i - 1) * width, vals):
            ax.text(x, v + 0.025, f"{v:.3f}", ha="center", size=SMALL - 2, rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels([OLAB[o] for o in ORDER], size=MED - 1)
    ax.set_ylabel("solve_rate (n=400 held-out)")
    ax.set_ylim(0, 1.18)
    ax.set_title("Same problems, different neighbour order: three models, four orderings",
                 size=MED + 2, pad=14)
    ax.axhline(0.90, color=GREY, ls="--", lw=1.5)
    # Bar alpha encodes the model; a colour legend would collide with the ordering
    # colours, so the models are spelled out in text at the top instead.
    ax.text(0.015, 0.965, "bar triplets, left to right:  " +
            "   |   ".join(LABEL[k].replace("\n", "  ") for k, _ in models),
            transform=ax.transAxes, size=SMALL - 1, color=GREY)
    footer(fig, "Each ordering preserves the edge set per node, the colouring, the hint set and the "
                "DD trace -- the trace is recomputed on the reordered adjacency and asserted equal, so "
                "these are the same instances asked the same question. Error bars are Wilson 95%.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    pdf.savefig(fig)

    # ---------------------------------------------------------------- page 3
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Paired, on the same 400 instances", size=BIG, weight="bold", y=0.95)
    axt = fig.add_axes([0.05, 0.17, 0.90, 0.70])
    rows, colors = [], {}
    for i, (key, byo) in enumerate(models):
        ident = fnum(byo["identity"], "heldout_solve_rate")
        for o in ORDER[1:]:
            r = byo[o]
            rows.append([LABEL[key].split("\n")[0] if o == "within_block" else "",
                         o, f"{ident:.4f}", f"{fnum(r, 'heldout_solve_rate'):.4f}",
                         f"{fnum(r, 'gap'):+.4f}", f"{float(r['p']):.3g}",
                         f"{fnum(r, 'well_formed_rate'):.3f}"])
            if o == "within_block":
                colors[len(rows) - 1] = GREEN
            else:
                colors[len(rows) - 1] = RED
    table(axt, ["model", "ordering", "identity", "reordered", "delta", "paired p", "well-formed"],
          rows, [0.17, 0.16, 0.13, 0.14, 0.13, 0.15, 0.12], fontsize=SMALL,
          row_colors=colors, dy=0.092)
    footer(fig, "GREEN = within_block, the design's own sanity check: it preserves the colour code and "
                "changes only the order, and it ties identity on all three models. That is what licenses "
                "reading the RED rows as the colour code rather than as distribution shift. McNemar exact "
                "on discordant pairs. Well-formed is the fraction of outputs that parse at all: its "
                "collapse at N=96 (0.185) means the model stops producing a valid trace, not merely a "
                "wrong one.")
    pdf.savefig(fig)

    # ---------------------------------------------------------------- page 4
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Where the leak comes from, and the fix", size=BIG, weight="bold", y=0.96)
    ax = fig.add_axes([0.05, 0.50, 0.90, 0.38])
    ax.axis("off")
    ax.text(0, 0.97,
            "planted_regular_graph (sudoku_dd.py:49) appends edges in class-pair block order",
            size=MED, va="top", weight="bold")
    ax.text(0, 0.74, "for a in range(q):\n    for b in range(a + 1, q):   # (0,1), (0,2), (0,3), (1,2), ...\n"
                     "        edge_blocks.append(...)\n...\nfor x, y in edges:   # adj filled in that order\n"
                     "    adj[x, fill[x]] = y",
            size=SMALL + 1, va="top", family="monospace", color=BLUE, linespacing=1.5)
    ax.text(0, 0.16,
            "The node relabelling that follows shuffles node IDs but not the order edges were appended,\n"
            "so adj[v] arrives sorted by the neighbour's TRUE COLOUR -- and encode_instance emits it raw.",
            size=MED - 1, va="top", linespacing=1.6)
    axt = fig.add_axes([0.05, 0.10, 0.90, 0.34])
    table(axt, ["measurement", "N=24", "N=48", "N=96"],
          [["adj[v] sorted by neighbour colour", "100.0%", "100.0%", "100.0%"],
           ["the one absent colour IS v's own", "67.1%", "61.3%", "--"],
           ["round events in ascending node index", "100.0%", "100.0%", "100.0%"],
           ["within-colour-block order sorted by id", "--", "38.9%", "36.2%"],
           ["AFTER THE FIX (shuffle_adj=True)", "5.7%", "9.1%", "--"]],
          [0.42, 0.19, 0.19, 0.20], fontsize=SMALL + 1,
          row_colors={0: RED, 4: GREEN})
    footer(fig, "Within-colour-block order is already arbitrary at chance (36-39% sorted), which is why "
                "within_block is in-distribution and makes a valid control. The fix permutes adj[v] in "
                "build_instances from a dedicated rng stream: colour-sorted 100.0% -> 5.7%/9.1% (chance "
                "for d=5 over q=4), the pre-fix generator stream stays bit-identical, and the edge set, "
                "colouring, hints and DD trace are unchanged instance by instance.")
    pdf.savefig(fig)

    # ---------------------------------------------------------------- page 5
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("The other question: are they memorizing their training set?",
                 size=BIG, weight="bold", y=0.96)
    ax = fig.add_axes([0.10, 0.40, 0.38, 0.46])
    labs, gaps, ps = [], [], []
    for r in memr:
        nt = r["n_train_instances"] or "20000"
        labs.append(f"N={r['n_nodes']}\n{int(nt) // 1000}k")
        gaps.append(fnum(r, "gap"))
        ps.append(float(r["p"]))
    xs = np.arange(len(labs))
    ax.bar(xs, gaps, 0.55, color=[RED if p < 0.05 else GREY for p in ps],
           edgecolor="black", linewidth=0.8)
    for x, g, p in zip(xs, gaps, ps):
        ax.text(x, g + (0.004 if g > 0 else -0.009), f"{g:+.4f}\np={p:.2g}",
                ha="center", va="bottom" if g > 0 else "top", size=SMALL - 1)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, size=SMALL + 1)
    # The full "solve_rate(own training) - solve_rate(held-out)" is clipped off
    # the left edge of the slide even with the axes inset; the footer says it.
    ax.set_ylabel("train - held-out  solve_rate", size=MED - 1)
    ax.set_ylim(-0.045, 0.105)
    ax.set_title("measured directly, n=600 per side", size=MED, pad=10)
    axt = fig.add_axes([0.53, 0.38, 0.43, 0.50])
    table(axt, ["run", "train", "held-out", "gap", "p"],
          [["N=48 L=12 @20000", "0.9967", "0.9383", "+0.0583", "1.2e-08"],
           ["N=96 ctrl @20000", "0.8967", "0.8250", "+0.0717", "3.4e-04"],
           ["N=96 cell B @80000", "0.9267", "0.9400", "-0.0133", "0.355"]],
          [0.36, 0.16, 0.18, 0.16, 0.14], fontsize=SMALL + 1,
          row_colors={0: RED, 1: RED, 2: GREEN})
    axt.text(0, 0.42, "Yes at 20000, and data closes it: no gap at 80000.",
             size=MED - 1, weight="bold", color=GREEN)
    axt.text(0, 0.32, "Pre-registered in advance and CONFIRMED -- the training set is\n"
                      "built once, before the epoch loop, so a 150-epoch run sees the\n"
                      "same instances 150 times and can memorize them.",
             size=SMALL, va="top", linespacing=1.6)
    axt.text(0, 0.08, "WITHDRAWN: the train-val NLL gaps I had been reading this off.\n"
                      "train_nll is averaged over the epoch WHILE weights update, so it\n"
                      "carries a +0.3263 bias at epoch 10 (N=96 control, decaying). The\n"
                      "sign of a +-0.03 gap means nothing, and finding 15's MECHANISM\n"
                      "rested on +0.0239. Its +0.1007 result is unaffected.",
             size=SMALL, va="top", linespacing=1.6, color=RED)
    footer(fig, "Training index i is reproducible as build_instances(cfg, 1, i), so a prefix of the "
                "training set is recoverable exactly and the model can be scored on the instances it "
                "actually saw. No run in this project had ever been scored that way. Unpaired "
                "two-proportion z-test: the train sample and the held-out sample are different instances.")
    pdf.savefig(fig)

    # ---------------------------------------------------------------- page 6
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("What this costs, and what is running now", size=BIG, weight="bold", y=0.96)
    axt = fig.add_axes([0.05, 0.42, 0.90, 0.45])
    table(axt, ["result", "survives?", "why"],
          [["solve_rate ladder 0.9960 / 0.9573 / 0.9273", "AS DECODING", "real on this generator, not deduction"],
           ["finding 2: the model is not running DD", "STRENGTHENED", "now with a mechanism for why it need not"],
           ["L*(T), L*(N), the depth-vs-N series", "RE-MEASURE", "depth to decode, not depth to deduce"],
           ["finding 13 / N=48 thin-deep beats wide", "RE-MEASURE", "a claim about decoding capacity"],
           ["finding 15: data is the binding lever (+0.1007)", "RESULT HOLDS", "mechanism withdrawn, see page 5"],
           ["memorization closes with data", "NEW, CLEAN", "measured on instances, not on NLL"]],
          [0.38, 0.17, 0.45], fontsize=SMALL + 1,
          row_colors={0: RED, 2: RED, 3: RED, 1: GREEN, 5: GREEN})
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.24])
    ax.axis("off")
    ax.text(0, 0.95, "49659502 -- the first run in this project measuring deduction",
            size=MED + 1, weight="bold", va="top", color=BLUE)
    ax.text(0, 0.62,
            "One change from the record model: --shuffle-adj. N=24, L=12, ff=1024, 80000 instances, 300 epochs.\n"
            "Pre-registered: below 0.9960 at p<0.05 but above 0.50 -- a real cost, not a collapse; order-insensitive\n"
            "across all colour-free orderings; and re-instating the leak (bycolor) does NOT help it. ~6 h.",
            size=SMALL + 1, va="top", linespacing=1.7)
    footer(fig, "RE-MEASURE does not mean wrong: those runs really do score what they score, on instances "
                "drawn from this generator. It means the quantity they measure is the depth and width "
                "needed to decode a positional code, not to execute T rounds of deduction, so the "
                "reasoning-depth reading of them does not go through. Which of them to redo is a scoping "
                "call -- 49659502 is one cheap cell that makes that call informed rather than a guess.")
    pdf.savefig(fig)

print(f"wrote {PDF} ({os.path.getsize(PDF) / 1e6:.2f} MB, {pdf.n} pages)")
