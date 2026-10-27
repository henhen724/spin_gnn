"""Slide deck for job 49744985: the layer-0 position-kind split.

Run as `python -m notebooks.make_kindsplit_deck` from the repo root.

Carries the tokenization and reasoning-trace pages (notebooks/deck_tokenization)
per hshunt's 2026-10-26 standing request -- and on this deck they are not
decoration: the whole result is about which token sits at which position.
"""
import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from spin_gnn.sudoku_activations import reasoning_positions
from notebooks.deck_tokenization import MONO, example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

SCRATCH = os.environ["SCRATCH"]
RUN = f"{SCRATCH}/spin_gnn/sudoku_sae_kindsplit_2026-10-27"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sudoku_sae_kindsplit_2026-10-27")

BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "0.45"
LAYERS = [0, 2, 4, 6, 8, 10, 12]
LF, LK = "leak-free (0.854)", "leaked record (0.996)"


def load(arm):
    df = pd.read_csv(f"{RUN}/{arm}_l0_64/sae_summary_by_kind.csv")
    df["fam"] = df.family_counts.map(ast.literal_eval)
    for f in ("justifies_next", "nbr_of_next", "frontier", "next_node"):
        df[f"n_{f}"] = df.fam.map(lambda d, f=f: d.get(f, 0))
    return df.set_index(["layer", "position_kind"])


def series(df, kind, col):
    return [df.loc[(L, kind), col] for L in LAYERS]


def table(ax, header, rows, colw, fontsize=SMALL, row_colors=None, dy=0.115,
          rule_dy=0.09):
    ax.axis("off")
    y = 0.92
    for i, txt in enumerate(header):
        ax.text(sum(colw[:i]), y, txt, size=fontsize, weight="bold", va="top")
    ax.plot([0, sum(colw)], [y - rule_dy, y - rule_dy], color="0.6", lw=1.0)
    for r, row in enumerate(rows):
        y -= dy
        c = (row_colors or {}).get(r, "black")
        for i, txt in enumerate(row):
            ax.text(sum(colw[:i]), y, txt, size=fontsize, va="top", color=c)
    ax.set_xlim(0, sum(colw)); ax.set_ylim(0, 1)


def page_title(out, lf, lk):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.80, "The layer-0 features were free.", size=BIG + 7, weight="bold")
    fig.text(0.06, 0.70, "The real structure splits by position kind.",
             size=BIG + 7, weight="bold")
    fig.text(0.06, 0.565,
             "Every one of the 23 layer-0 next_node features reported last time fires at pre_color\n"
             "positions, where the current token IS the node, at F1 exactly 1.000. At pre_node\n"
             "positions, where the next node is not yet known, there are ZERO -- in BOTH models.",
             size=MED + 1, va="top")
    fig.text(0.06, 0.345,
             "So job 49729375's pre-registration (a) was UNTESTABLE as written, not false: pooling the\n"
             "two kinds of position mixed a token-identity match into the count. Splitting them also\n"
             "sharpens the main result -- each kind of position carries the DD family it needs, but\n"
             "NOT at the time I predicted. My pre-registration (d) here is falsified.",
             size=SMALL + 1, va="top", color="0.25")
    fig.text(0.06, 0.13,
             f"Same two checkpoints, seed, layers and sparsity target as 49729375, plus --split-by-kind.\n"
             f"The pooled sae_summary.csv came back byte-identical to 49729375 for both arms, so "
             f"finding 19 reproduces exactly.",
             size=SMALL, va="top", color=GREY)
    footer(fig, "job 49744985, COMPLETED 01:00:56, ExitCode 0:0 | commit f24bb40 | "
                "scripts/slurm/sudoku2_sae_kindsplit.sbatch")
    out.savefig(fig)


def page_kinds(out, lf, lk):
    """The mechanism: the real token stream, with the two kinds of position marked."""
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "Two kinds of position, and why one of them is free",
             size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "A position is where the model is about to emit a token: pos is the index with "
             "tokens[pos + 1] as the target.", size=SMALL)

    toks = np.asarray(ex.tokens)
    begin = int(np.flatnonzero(toks == ex.vocab.BEGIN_REASONING)[0])
    kinds = {p["pos"]: p["kind"] for p in reasoning_positions(ex.tokens, ex.vocab)}
    # From BEGIN_ROUND, not BEGIN_REASONING: the two markers are adjacent and
    # the first name is wide enough to run under the second column.
    show = list(range(begin + 1, min(begin + 8, len(toks))))

    ax = fig.add_axes([0.06, 0.52, 0.90, 0.28]); ax.axis("off")
    ax.text(0, 1.0, "THE REASONING SECTION OF THE INSTANCE ON THE PREVIOUS PAGE",
            size=SMALL, weight="bold", va="top")
    dx = 1.0 / len(show)
    for j, p in enumerate(show):
        x = j * dx
        ax.text(x, 0.74, ex.names[p], size=SMALL - 1, va="top", **MONO)
        ax.text(x, 0.56, f"pos {p}", size=SMALL - 3, va="top", color=GREY, **MONO)
        if p in kinds:
            col = BLUE if kinds[p] == "pre_node" else RED
            ax.text(x, 0.38, kinds[p], size=SMALL - 2, va="top", color=col,
                    weight="bold", **MONO)
            ax.text(x, 0.20, f"-> {ex.names[p + 1]}", size=SMALL - 3, va="top",
                    color=col, **MONO)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    left = [("pre_node  (pos = i - 1)", BLUE),
            ("tokens[pos] is BEGIN_ROUND or the PREVIOUS", "black"),
            ("deduction's colour. Nothing at the embedding", "black"),
            ("identifies the node that comes next: the model", "black"),
            ("has to work out which node is deducible.", "black")]
    right = [("pre_color  (pos = i)", RED),
             ("tokens[pos] IS the node whose colour is the", "black"),
             ("target. next_node[v] is therefore matched by a", "black"),
             ("pure token-identity feature, at layer 0, before", "black"),
             ("any attention, with no computation and no leak.", "black")]
    for x, col in ((0.06, left), (0.53, right)):
        for i, (line, c) in enumerate(col):
            fig.text(x, 0.44 - 0.045 * i, line, size=SMALL,
                     weight="bold" if i == 0 else "normal", color=c, va="top")

    fig.text(0.06, 0.19, "AND THAT IS EXACTLY WHAT LAYER 0 CONTAINS, IN BOTH MODELS:",
             size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.145,
             f"leak-free   pre_color {int(lf.loc[(0,'pre_color'),'n_next_node'])} next_node features at best F1 "
             f"{lf.loc[(0,'pre_color'),'best_next_node_f1']:.3f}"
             f"      pre_node {int(lf.loc[(0,'pre_node'),'n_next_node'])} features, best F1 "
             f"{lf.loc[(0,'pre_node'),'best_next_node_f1']:.3f}\n"
             f"leaked      pre_color {int(lk.loc[(0,'pre_color'),'n_next_node'])} next_node features at best F1 "
             f"{lk.loc[(0,'pre_color'),'best_next_node_f1']:.3f}"
             f"      pre_node {int(lk.loc[(0,'pre_node'),'n_next_node'])} features, best F1 "
             f"{lk.loc[(0,'pre_node'),'best_next_node_f1']:.3f}",
             size=SMALL, va="top", **MONO)
    footer(fig, "spin_gnn.sudoku_activations.reasoning_positions; F1 = 1.000 exactly is the "
                "signature of a token-identity feature")
    out.savefig(fig)


def page_layer0(out, lf, lk):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "Layer 0, decomposed", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "49729375 reported 23 vs 24 interpretable next_node features at layer 0 and I recorded\n"
             "pre-registration (a) as falsified. Here is where all of them were.",
             size=SMALL, va="top")

    rows = []
    for arm, df in (("leak-free", lf), ("leaked", lk)):
        for kind in ("pre_node", "pre_color"):
            r = df.loc[(0, kind)]
            rows.append([arm, kind, f"{int(r.n_positions)}", f"{int(r.n_alive)}",
                         f"{int(r.n_interpretable)}", f"{int(r.n_next_node)}",
                         f"{r.best_next_node_f1:.3f}", f"{r.median_best_f1_null:.3f}"])
    ax = fig.add_axes([0.06, 0.50, 0.90, 0.28])
    table(ax, ["model", "positions", "n", "alive", "interp.", "next_node feats",
               "best F1", "median null"],
          rows, [0.13, 0.13, 0.10, 0.09, 0.11, 0.17, 0.13, 0.14], fontsize=SMALL,
          row_colors={0: BLUE, 1: RED, 2: BLUE, 3: RED}, dy=0.17, rule_dy=0.13)

    fig.text(0.06, 0.42, "WHAT THIS SETTLES", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.375,
             "(a) CONFIRMED. The leak-free model has zero next_node-family features above null at\n"
             "     pre_node positions and best_next_node_f1 = 0.000, below its own median null (0.092).\n"
             "(b) CONFIRMED. At pre_color it has 23, at F1 exactly 1.000 -- the token-identity match.\n"
             "(c) CONFIRMED, in the branch I predicted. The leaked model decomposes identically\n"
             "     (0 / 24), so the 23-vs-24 count compared two freebies. 49729375's pre-reg (a) was\n"
             "     UNTESTABLE as written and is reclassified from FALSIFIED to UNTESTABLE.",
             size=SMALL, va="top")
    fig.text(0.06, 0.145,
             "Layer 0 is the token+position embedding, and the only layer in either arm that misses the\n"
             "sparsity target (L0 5.9 / 7.0 against 64): a 2-sparse embedding is what the dictionary finds\n"
             "because that is what is there. Both arms keep ~10 interpretable pre_node features, none next_node.",
             size=SMALL - 1, va="top", color=GREY)
    footer(fig, "sae_summary_by_kind.csv, layer 0, target L0=64, 26600 positions per mask")
    out.savefig(fig)


def page_families(out, lf, lk):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "Each kind of position carries a different DD family",
             size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "Left: the edges that force the colour. Right: the frontier, which selects the node. "
             "Both are best F1 among alive features.", size=SMALL)

    for k, (col, title) in enumerate((("best_justifier_f1", "justifies_next[u]"),
                                      ("best_frontier_f1", "frontier[v]"))):
        ax = fig.add_axes([0.08 + 0.50 * k, 0.40, 0.37, 0.38])
        for df, arm, ls in ((lf, "leak-free", "-"), (lk, "leaked", "--")):
            for kind, c in (("pre_node", BLUE), ("pre_color", RED)):
                ax.plot(LAYERS, series(df, kind, col), ls, color=c, marker="o",
                        ms=4, lw=1.8 if arm == "leak-free" else 1.1,
                        alpha=1.0 if arm == "leak-free" else 0.55,
                        label=f"{arm} {kind}")
        ax.axhline(0.3, color="0.6", ls=":", lw=1.0)
        ax.set_xlabel("layer", size=SMALL); ax.set_ylabel("best F1", size=SMALL)
        ax.set_title(title, size=SMALL + 1)
        ax.set_ylim(0, 1.0); ax.set_xticks(LAYERS)
        ax.tick_params(labelsize=SMALL - 2)
        if k == 0:
            ax.legend(fontsize=SMALL - 4, loc="upper left")

    fig.text(0.06, 0.315,
             "THE EDGES ARE REAL AND THE CONTRAST HOLDS -- BUT ONLY AT pre_color.", size=SMALL,
             weight="bold", va="top")
    n_lf = sum(int(lf.loc[(L, "pre_color"), "n_justifies_next"]) for L in LAYERS)
    n_lk = sum(int(lk.loc[(L, "pre_color"), "n_justifies_next"]) for L in LAYERS)
    fig.text(0.06, 0.27,
             f"At pre_color the leak-free model has {n_lf} justifies_next features against the leaked "
             f"model's {n_lk}, and best F1\n"
             f"{lf.loc[(8,'pre_color'),'best_justifier_f1']:.3f} vs "
             f"{lk.loc[(8,'pre_color'),'best_justifier_f1']:.3f} at layer 8 -- finding 19's result, "
             f"intact and larger than the pooled version.\n"
             f"At pre_node BOTH arms have zero, at F1 0.15-0.25, under the cut. The frontier is the "
             f"mirror image: the\nleak-free model has more frontier features at pre_node than at "
             f"pre_color at every layer above 0 (6 of 6).",
             size=SMALL - 1, va="top")
    fig.text(0.06, 0.125,
             "Read positively, this is relevant edges at the relevant time -- the known neighbours "
             "matter when the COLOUR is\nbeing chosen, and the node is already fixed by then. "
             "I predicted the other time, and said so in advance.",
             size=SMALL - 1, va="top", color="0.25")
    footer(fig, "solid = leak-free, faint dashed = leaked; blue = pre_node, red = pre_color; "
                "dotted = the F1>=0.3 interpretability cut")
    out.savefig(fig)


def page_prereg(out, lf, lk):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "Pre-registrations, as resolved", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "Written into scripts/slurm/sudoku2_sae_kindsplit.sbatch before the job ran; "
             "commit f24bb40.", size=SMALL)

    lf0n, lf0c = lf.loc[(0, "pre_node")], lf.loc[(0, "pre_color")]
    rows = [
        ["(a)", "leak-free L0 pre_node has ZERO next_node features",
         "CONFIRMED", f"0 features, best F1 {lf0n.best_next_node_f1:.3f} vs null "
                      f"{lf0n.median_best_f1_null:.3f}"],
        ["(b)", "leak-free L0 pre_color has many, at high F1",
         "CONFIRMED", f"{int(lf0c.n_next_node)} features, best F1 "
                      f"{lf0c.best_next_node_f1:.3f} (exactly 1)"],
        ["(c)", "if both hold in both arms, 49729375's (a) was\n"
                "      untestable rather than false",
         "CONFIRMED", "leaked decomposes 0 / 24 identically;\n"
                      "      (a) reclassified"],
        ["(d)", ">=1 justifies_next feature survives at pre_node\n"
                "      (L8 or L12), and leak-free beats leaked there",
         "FALSIFIED", "0 features at pre_node at every layer, both\n"
                      "      arms; and leak-free is LOWER there\n"
                      "      (0.171 vs 0.241 at L12)"],
        ["(e)", "frontier F1 higher at pre_node than pre_color\n      (leak-free)",
         "SPLIT", "4 of 6 layers on F1, but 6 of 6 on feature counts\n"
                  "      (31/31/26/26/35/33 vs 13/14/19/15/14/2)"],
    ]
    ax = fig.add_axes([0.06, 0.34, 0.90, 0.47])
    table(ax, ["", "claim", "verdict", "evidence"], rows,
          [0.05, 0.42, 0.13, 0.40], fontsize=SMALL - 1,
          row_colors={0: GREEN, 1: GREEN, 2: GREEN, 3: RED, 4: "0.35"},
          dy=0.185, rule_dy=0.10)

    fig.text(0.06, 0.245,
             "(d) is the one that cost something, and it is worth being exact about what it cost.",
             size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.20,
             "I predicted the edge features would appear while the model is CHOOSING the node, on the\n"
             "theory that the justifying neighbours are what makes a node deducible. They appear only\n"
             "while it is choosing the COLOUR. Both are 'relevant edges at a relevant time', and the\n"
             "second is the one DD's firing rule actually needs -- but it is the weaker of the two\n"
             "claims, because at pre_color the node is already in the context window.",
             size=SMALL - 1, va="top")
    footer(fig, "my record on mechanism predictions in this project: 4 confirmed, "
                "1 falsified, 1 split, over the last two jobs")
    out.savefig(fig)


def page_caveats(out):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "What this changes, and what it does not", size=BIG, weight="bold")

    fig.text(0.06, 0.855, "CHANGED IN FINDING 19", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.815,
             "- Pre-reg (a) goes from FALSIFIED to UNTESTABLE: the 23-vs-24 layer-0 count compared\n"
             "  two token-identity freebies and could not have separated the arms.\n"
             "- The edge result is restated: the leak-free model's justifies_next features live at\n"
             "  pre_color positions only. The leak-free vs leaked contrast is unchanged and in fact\n"
             "  larger inside that mask than it was pooled.\n"
             "- The pooled sae_summary.csv reproduced 49729375 byte-for-byte in both arms.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.60, "NEW OPEN PROBLEM, FOUND HERE", size=SMALL, weight="bold",
             va="top", color=RED)
    fig.text(0.06, 0.56,
             "The leaked model has MORE next_node features at pre_node than the leak-free one (60 vs\n"
             "15 at L12, best F1 0.741 vs 0.545) -- it is better at knowing which node comes next\n"
             "before emitting it. Finding 16 measured that round events are emitted in ASCENDING NODE\n"
             "INDEX for 100% of rounds, and shuffle_adj permutes adj[v] only, not that. So a next_node\n"
             "feature at pre_node may track 'the index goes up' rather than deduction, in BOTH arms.\n"
             "Permuting within-round event order is a separate control, never run, and cheap.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.345, "CAVEATS", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.305,
             "- Each mask keeps 26600 of 53200 positions, so a feature at the 1e-3 aliveness floor\n"
             "  fires ~26 times instead of ~52. Counts compare WITHIN a mask, not across masks.\n"
             "- The dictionaries are the pooled ones, scored twice; this does not ask what a dictionary\n"
             "  fitted to pre_node positions alone would look like.\n"
             "- One seed, one checkpoint per arm. best_f1 is a max over 256 predicates; the rolled null\n"
             "  controls the argmax, not the interpretation.\n"
             "- None of this says either model runs DD, only which DD predicates are readable, and when.",
             size=SMALL - 1, va="top")
    footer(fig, "next: a second SAE seed, and the event-order control above")
    out.savefig(fig)


def main():
    global ex
    set_style()
    os.makedirs(OUT, exist_ok=True)
    lf, lk = load("leakfree"), load("leaked")
    ex = example_instance(shuffle_adj=True, pick="median")
    pdf_path = os.path.join(OUT, "sae_kindsplit.pdf")
    with PdfPages(pdf_path) as pdf:
        out = PdfAndPng(pdf, OUT)
        page_title(out, lf, lk)
        page_tokenization(out, ex)
        page_trace(out, ex)
        page_kinds(out, lf, lk)
        page_layer0(out, lf, lk)
        page_families(out, lf, lk)
        page_prereg(out, lf, lk)
        page_caveats(out)
    print(f"{out.n} pages -> {pdf_path} ({os.path.getsize(pdf_path)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
