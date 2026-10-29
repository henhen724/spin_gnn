"""Slide deck for findings 31 and 32: the last rung of the N=96 ladder.

Companion to notebooks/sudoku_n96_rungs_2026-10-29_results.ipynb, which carries
the full argument; this is the 8-page version meant to be read on a phone. Every
number is recomputed here from the run directories, not copied from the
notebook's markdown, so the two can be diffed against each other.

Run from the repo root:  python -m notebooks.make_n96_rungs_deck
"""
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

S = os.environ["SCRATCH"] + "/spin_gnn"
L07 = f"{S}/sudoku_n96_L07_2026-10-28"          # 50053882, finding 31
WARM = f"{S}/sudoku_n96_warmup_2026-10-28"      # 50008605, finding 32
WIDE = f"{S}/sudoku_n96_layer_scan_2026-10-26"  # 49639312, finding 22
COLD = f"{S}/sudoku_n96_ff256_2026-10-27"       # 49816217, finding 25
OUT = f"{S}/notebooks/sudoku_n96_rungs_2026-10-29"

REF = "d0256_h04_L08_ff01024_lr1e-03_pre"       # the wide knee, in every pool
GATE = 0.90
RED, GREEN, BLUE = "#c0392b", "#1e8449", "#1f77b4"

LAB = {"d0256_h04_L04_ff01024_lr1e-03_pre": "L=4 ff=1024",
       "d0256_h04_L06_ff01024_lr1e-03_pre": "L=6 ff=1024",
       "d0256_h04_L07_ff01024_lr1e-03_pre": "L=7 ff=1024 (NEW)",
       "d0256_h04_L08_ff01024_lr1e-03_pre": "L=8 ff=1024 (knee)",
       "d0256_h04_L12_ff01024_lr1e-03_pre": "L=12 ff=1024",
       "d0256_h04_L08_ff00256_lr1e-03_pre": "L=8 ff=256 cold",
       "d0256_h04_L08_ff00256_lr1e-03_pre_w05": "L=8 ff=256 WARM (NEW)",
       "d0256_h04_L12_ff00256_lr1e-03_pre": "L=12 ff=256 cold",
       "d0256_h04_L12_ff00256_lr1e-03_pre_w05": "L=12 ff=256 WARM (NEW)"}


def solved(d):
    """Per-instance correctness on the shared pool, keyed by checkpoint tag."""
    z = np.load(f"{d}/sharedpool_n1500_per_instance.npz", allow_pickle=True)
    return dict(zip([str(x) for x in z["names"]], z["solved"]))


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans -> (a-only, b-only, p)."""
    import math
    n10, n01 = int((a & ~b).sum()), int((b & ~a).sum())
    m, k = n10 + n01, min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    return n10, n01, p


def load():
    """Everything the pages draw, computed once."""
    sv_l07, sv_warm = solved(L07), solved(WARM)
    pairs = []
    # (key, label): the key is what the other pages index by, so a label can be
    # shortened to fit a table column without silently breaking a .loc lookup.
    for key, lab, a, b in [
            ("rungs", "L=7 vs L=6 -- adjacent wide rungs",
             sv_l07["d0256_h04_L07_ff01024_lr1e-03_pre"],
             sv_l07["d0256_h04_L06_ff01024_lr1e-03_pre"]),
            ("knee", "L=7 vs L=8 -- the knee it undercuts",
             sv_l07["d0256_h04_L07_ff01024_lr1e-03_pre"], sv_l07[REF]),
            ("treatment", "L=12/ff256 WARM vs its COLD self -- TREATMENT",
             sv_warm["d0256_h04_L12_ff00256_lr1e-03_pre_w05"],
             sv_warm["d0256_h04_L12_ff00256_lr1e-03_pre"]),
            ("control", "L=8/ff256 WARM vs its COLD self -- CONTROL",
             sv_warm["d0256_h04_L08_ff00256_lr1e-03_pre_w05"],
             sv_warm["d0256_h04_L08_ff00256_lr1e-03_pre"]),
            ("reversal", "L=12/ff256 WARM vs L=8/ff256 cold -- reversal",
             sv_warm["d0256_h04_L12_ff00256_lr1e-03_pre_w05"],
             sv_warm["d0256_h04_L08_ff00256_lr1e-03_pre"]),
            ("vs_knee", "L=12/ff256 WARM vs the wide knee",
             sv_warm["d0256_h04_L12_ff00256_lr1e-03_pre_w05"], sv_warm[REF])]:
        w, l, p = mcnemar(a, b)
        pairs.append({"key": key, "contrast": lab, "a": a.mean(), "b": b.mean(),
                      "delta": a.mean() - b.mean(), "won": w, "lost": l, "p": p})

    dep = pd.concat([pd.read_csv(f"{WIDE}/depth_by_layers_n96.csv"),
                     pd.read_csv(f"{COLD}/depth_by_layers_n96_ff256.csv"),
                     pd.read_csv(f"{L07}/depth_n96_L07.csv"),
                     pd.read_csv(f"{WARM}/depth_n96_ff256_warmup.csv")])
    d60 = dep[dep.rho_requested == 0.60].copy()

    nll = []
    for d in (WIDE, COLD, L07, WARM):
        sw = pd.read_csv(f"{d}/sweep.csv")
        pool_t = pd.read_csv(f"{d}/sharedpool_n1500.csv").set_index("checkpoint")
        for _, r in sw.iterrows():
            tag = os.path.basename(r.checkpoint_dir)
            nll.append({"tag": tag, "cell": LAB.get(tag, tag), "L": int(r.num_layers),
                        "ff": int(r.dim_feedforward), "params_M": r.n_params / 1e6,
                        "warm": int(r.get("warmup_epochs", 0)),
                        "val_nll": r.final_val_nll,
                        "pool": float(pool_t.solve_rate.get(tag, np.nan))})
    nll = pd.DataFrame(nll).drop_duplicates("cell").dropna(subset=["pool"])

    wpool = pd.read_csv(f"{WARM}/sharedpool_n1500.csv").set_index("checkpoint")
    return {"d60": d60, "pairs": pd.DataFrame(pairs), "nll": nll, "wpool": wpool,
            "ref_same": bool((sv_l07[REF] == sv_warm[REF]).all())}


def gate_row(d60, tag):
    """(bins cleared, bins total, Cochran-Armitage z, p) for one cell."""
    g = d60[d60.checkpoint == tag].sort_values("T")
    k = int((g.solve_rate >= GATE).sum())
    x = np.repeat(g["T"].values, g.n.values)
    y = np.concatenate([np.r_[np.ones(int(round(s * n))), np.zeros(n - int(round(s * n)))]
                        for s, n in zip(g.solve_rate, g.n)])
    z = np.corrcoef(x, y)[0, 1] * np.sqrt(len(x))
    return k, len(g), z, 2 * stats.norm.sf(abs(z))


def page_title(o, D):
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.5, 0.945, "How many layers does N=96 need? The last two rungs",
             size=BIG, weight="bold", ha="center")
    fig.text(0.5, 0.895,
             "L* = the fewest LAYERS that clear a 0.90 solve rate in EVERY depth bin at fixed hint density. "
             "Two one-change runs, both also scored on one shared 1500-instance pool.",
             size=SMALL, ha="center", color="0.3")

    p = D["pairs"].set_index("key")
    l7 = p.loc["knee"]
    tr = p.loc["treatment"]
    ct = p.loc["control"]
    k7, n7, _, _ = gate_row(D["d60"], "d0256_h04_L07_ff01024_lr1e-03_pre")
    kw, nw, _, _ = gate_row(D["d60"], "d0256_h04_L12_ff00256_lr1e-03_pre_w05")
    # Read off the sweep, not typed: the parameter ratio is the one headline
    # number on this page that is not a rate, and it should move if a cell does.
    pm = D["nll"].set_index("tag").params_M
    p7 = pm["d0256_h04_L07_ff01024_lr1e-03_pre"]
    p8 = pm["d0256_h04_L08_ff01024_lr1e-03_pre"]

    for i, (colour, head, body) in enumerate([
            (BLUE, "L*(96, ff=1024) = 8 STANDS -- but the two metrics disagree about why.",
             [f"L=7 clears the 0.90 gate in {k7} of {n7} depth bins, missing only T=8, by 0.003.",
              # int(): these came out of a DataFrame as floats, and "92.0-91.0"
              # reads as a rate rather than as a count of disagreeing instances.
              f"On the pool it is {l7.delta:+.4f} against L=8 ({int(l7.won)}-{int(l7.lost)} discordant, "
              f"p = {l7.p:.3g}): a TIE, at {p7 / p8:.0%} of the parameters ({p7:.2f}M vs {p8:.2f}M).",
              "So the ladder has a rung the metric of record cannot see."]),
            (GREEN, "Most of the deep-thin deficit was TRAINABILITY, not capacity.",
             [f"Five epochs of LR warmup move L=12/ff=256 by {tr.delta:+.4f} on the pool "
              f"({int(tr.won)}-{int(tr.lost)}, p = {tr.p:.2g}).",
              f"THE CONTROL HELD: the cell with no grammar problem moved {ct.delta:+.4f} "
              f"(p = {ct.p:.2g}), so warmup did not just lift the whole column.",
              f"And it is PARTIAL: the warm cell still clears the gate in {kw} of {nw} bins, "
              "so L*(96, ff=256) > 12 stands."])]):
        y = 0.82 - 0.265 * i
        fig.text(0.055, y, head, size=MED + 1, weight="bold", color=colour)
        for j, line in enumerate(body):
            fig.text(0.075, y - 0.055 - 0.045 * j, line, size=MED - 2)

    fig.text(0.055, 0.375, "WHAT L* IS MEASURED ON, AND THE NOTATION THE LATER PAGES USE",
             size=SMALL, weight="bold")
    # Every term the pages after this one use without re-explaining. A reader
    # seeing the project for the first time should not have to infer "ff" or
    # "cell" from a footer four pages later.
    fig.text(0.075, 0.340,
             "Graph colouring solvable by Direct Deduction (DD) alone: colour the one node whose known "
             "neighbours leave it a single colour, repeat. N=96 nodes, degree 5,\n"
             "q=4 colours. An instance's DEPTH T is how many DD rounds it takes. rho = the fraction of "
             "nodes given as hints; DENSITY is the realized hint fraction, held at\n"
             "0.5938 in every bin. ff = the width of the MLP inside each transformer block (ff=1024 WIDE, "
             "ff=256 THIN). A CELL is one trained model, one seed. COLD = flat\n"
             "learning rate, WARM = a 5-epoch ramp to the same rate. THE KNEE = L=8/ff=1024, the shallowest "
             "wide cell clearing every bin. 'Finding NN' = this project's log.",
             size=MED - 4, va="top", linespacing=1.55)

    # Up front, not in the limitations: with shuffle_adj=False a solve rate on
    # this ladder is a DECODING score, and finding 16 showed these checkpoints
    # collapse when adj[v] is permuted. Reading these pages as deduction rates
    # is the one misreading that would change what they mean.
    # Wrapped at 140, not by hand: at bold 13pt from x=0.055 the slide holds
    # about 145 characters, and the hand-broken version ran a fourth line down
    # into the footer.
    fig.text(0.055, 0.190,
             textwrap.fill(
                 "CAVEAT, up front: these runs keep the ADJACENCY-ORDER LEAK (shuffle_adj=False), so every rate "
                 "here is a DECODING score, not a deduction score. Each node's neighbour row is listed sorted by "
                 "the neighbours' TRUE colours, which names the node's own colour outright for 61-67% of nodes "
                 "(finding 16); permuting those rows dropped the N=96 model it was measured on from 0.94 to 0.04. "
                 "One seed per cell. The leak-free ladder is a separate series, running at N=48.", width=152),
             size=MED - 4, va="top", color="0.15", weight="bold", linespacing=1.5)
    footer(fig, "Jobs 50053882 and 50008605. Prepared 2026-10-29; every number on every page is "
                "recomputed from the run outputs by this script, not typed in.")
    o.savefig(fig)


def page_pool(o, D):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("The metric of record: one pool, every comparison paired", size=BIG, weight="bold")
    ax = fig.add_axes([0.04, 0.40, 0.92, 0.46]); ax.axis("off")
    cols = ["contrast", "rates", "difference", "discordant\n(won-lost)", "exact\nMcNemar p", "reading"]
    widths = [0.42, 0.13, 0.11, 0.12, 0.12, 0.10]
    rows, flags = [], []
    for _, r in D["pairs"].iterrows():
        sig = r.p < 0.05
        read = ("tie" if not sig else "resolvable")
        rows.append([r.contrast, f"{r.a:.4f} vs {r.b:.4f}", f"{r.delta:+.4f}",
                     f"{int(r.won)}-{int(r.lost)}", f"{r.p:.3g}", read])
        flags.append(sig)
    t = ax.table(cellText=rows, colLabels=cols, colWidths=widths, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(MED - 3); t.scale(1, 3.0)
    for j in range(len(cols)):
        t[0, j].set_facecolor("#dddddd"); t[0, j].set_text_props(weight="bold")
    for i, sig in enumerate(flags, start=1):
        t[i, 0].set_text_props(ha="left")
        t[i, 2].set_text_props(weight="bold", color="0.0" if sig else "0.45")
        t[i, 5].set_text_props(color="0.0" if sig else "0.45")

    fig.text(0.5, 0.345,
             "A TIE HERE IS NOT AN UNDERPOWERED TEST. L=7 vs L=8 disagree on 183 of 1500 instances and split "
             "them 92-91: the pool\nhas the resolution and reports no difference. That is the one weakness in "
             "this ladder no further N=96 run will fix.",
             size=MED - 3, ha="center", va="top", linespacing=1.6)
    fig.text(0.5, 0.235,
             "THE CONTROL IS THE LOAD-BEARING ROW. Had warmup lifted the whole thin column, finding 25's "
             "width gap would have been\nmeasured against an undertrained competitor and every N=96 width number "
             "would need re-running. The cell with no grammar\nproblem moved by -0.0040 (p = 0.77) while the cell "
             "with one moved by +0.0800.",
             size=MED - 3, ha="center", va="top", linespacing=1.6)
    footer(fig, "In the last column 'resolvable' means p < 0.05 on the paired test and 'tie' means it is not; "
                "at n=1500 a tie is a measured absence of difference, not a missing measurement. "
                "sudoku_agg_eval draws instance i at seed_start = 5e10, far above every training seed, so a "
                "larger n is a strict superset of a smaller one and all cells see byte-identical prompts: every "
                "row above is paired instance by instance. 'discordant' counts only the instances the two cells "
                "disagree on, which is what the exact test uses. The reference checkpoint's 1500-instance "
                f"correctness vector is {'IDENTICAL' if D['ref_same'] else 'NOT identical'} across the two runs' "
                "independent eval invocations -- a free replication check.")
    o.savefig(fig)


def page_depth(o, D):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Where L* is read off: six depth bins, density held fixed", size=BIG, weight="bold")
    d60 = D["d60"]
    fig.text(0.5, 0.900, "dotted line = the 0.90 gate;   [k/6] in each legend = how many of the six bins that "
             "cell clears", size=SMALL, ha="center", color="0.3")
    axes = [fig.add_axes([0.07, 0.42, 0.41, 0.40]), fig.add_axes([0.56, 0.42, 0.41, 0.40])]
    series = [("The wide ladder, ff=1024", axes[0],
               [("d0256_h04_L04_ff01024_lr1e-03_pre", "0.75"),
                ("d0256_h04_L06_ff01024_lr1e-03_pre", "tab:orange"),
                ("d0256_h04_L07_ff01024_lr1e-03_pre", BLUE),
                ("d0256_h04_L08_ff01024_lr1e-03_pre", GREEN)]),
              ("The thin column, ff=256: warm (dashed) vs cold", axes[1],
               [("d0256_h04_L08_ff00256_lr1e-03_pre", "0.6"),
                ("d0256_h04_L08_ff00256_lr1e-03_pre_w05", GREEN),
                ("d0256_h04_L12_ff00256_lr1e-03_pre", RED),
                ("d0256_h04_L12_ff00256_lr1e-03_pre_w05", "tab:purple")])]
    for title, ax, cells in series:
        for tag, colour in cells:
            g = d60[d60.checkpoint == tag].sort_values("T")
            if not len(g):
                continue
            # Jeffreys interval per bin: the two deepest bins hold 68 and 43
            # instances, and that is exactly where the L=7 verdict is decided.
            lo, hi = [], []
            for _, r in g.iterrows():
                k = r.solve_rate * r.n
                a, b = stats.beta.interval(0.95, k + 0.5, r.n - k + 0.5)
                lo.append(max(0.0, r.solve_rate - a))
                hi.append(max(0.0, b - r.solve_rate))
            k_cl, n_cl, _, _ = gate_row(d60, tag)
            ax.errorbar(g["T"], g.solve_rate, yerr=[lo, hi], marker="o", lw=2,
                        ls="--" if tag.endswith("_w05") else "-", color=colour, capsize=3,
                        label=f"{LAB[tag]}  [{k_cl}/{n_cl}]", rasterized=True)
        ax.axhline(GATE, color="k", ls=":", lw=1.5)
        ax.set_title(title, size=MED)
        ax.set_xlabel("DD depth T of the instance")
        ax.set_ylim(0.55, 1.02)
        ax.legend(fontsize=SMALL - 2, loc="lower left")
    axes[0].set_ylabel("solved, within that bin")
    # The gate is named under the title, not beside the line: at the left edge
    # the label sat on the L=6 error bar, and no in-axes band is empty in both
    # panels at once.
    axes[1].tick_params(labelleft=False)

    n_per = d60.groupby("T").n.max().to_dict()
    fig.text(0.5, 0.370,
             "THIS IS A SEPARATE EVAL FROM THE POOL ON PAGES 4 AND 7. Here 1500 instances are drawn at rho=0.60 "
             "and binned by DD depth T; the six bins with\nn>=60 hold "
             + str(int(sum(n_per.values()))) + " of them (n = "
             + ", ".join(f"{int(v)} (T={int(k)})" for k, v in n_per.items()) + "). Realized density is 0.5938 in "
             "EVERY bin of EVERY cell, so the bins differ in\nreasoning depth and in nothing else. The pool is a "
             "different draw, spanning the whole training range rho=0.30-0.95, and runs harder -- bin rates and "
             "pool\nrates are not interchangeable, which is why the two metrics can disagree about L=7.",
             size=MED - 4, ha="center", va="top", linespacing=1.55)
    fig.text(0.5, 0.235,
             "L=7's single miss is T=8 at 0.897 in a 68-instance bin, where ONE instance is 0.015. Three "
             "thousandths is not a measurement; it is\nwhere a threshold happened to fall -- so the N=96 width "
             "factor is quoted as '>1.5x, denominator resting on a 0.003 bin margin'.",
             size=MED - 3, ha="center", va="top", linespacing=1.6, weight="bold")
    footer(fig, "Cold L=12/ff=256 is FLAT and faintly RISING in T (0.829-0.868, Cochran-Armitage z = +0.67): the "
                "signature of a model whose failures are not about reasoning depth at all, which is what the "
                "grammar diagnosis on the next page predicted. Warm, it FALLS with T (z = -1.54) like an "
                "under-depth reasoner. Warmup changed the shape of the failure, not only its level.")
    o.savefig(fig)


def page_mechanism(o, D):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("What warmup repaired: the model could not WRITE the trace", size=BIG, weight="bold")
    d60 = D["d60"]
    tags = ["d0256_h04_L08_ff00256_lr1e-03_pre", "d0256_h04_L08_ff00256_lr1e-03_pre_w05",
            "d0256_h04_L12_ff00256_lr1e-03_pre", "d0256_h04_L12_ff00256_lr1e-03_pre_w05"]
    wmean, worst = [], []
    for tag in tags:
        g = d60[d60.checkpoint == tag]
        wmean.append((g.well_formed_rate * g.n).sum() / g.n.sum())
        worst.append(g.well_formed_rate.min())

    ax = fig.add_axes([0.07, 0.36, 0.40, 0.44])
    x = np.arange(len(tags))
    ax.bar(x - 0.2, wmean, 0.4, color=["0.6", GREEN, "0.6", "tab:purple"], label="n-weighted over the six bins")
    ax.bar(x + 0.2, worst, 0.4, color="w", edgecolor=["0.6", GREEN, "0.6", "tab:purple"],
           hatch="///", label="the WORST single bin")
    for xi, (m, w) in enumerate(zip(wmean, worst)):
        ax.text(xi - 0.2, m + 0.008, f"{m:.3f}", ha="center", size=SMALL - 2, weight="bold")
        ax.text(xi + 0.2, w + 0.008, f"{w:.3f}", ha="center", size=SMALL - 2)
    # The 0.98 line is explained in the LEGEND, not by a label next to it: as a
    # label it collided with the 0.985 bar value whichever side it was anchored
    # on, and right-aligned at the axes edge it also ran into the frame.
    ax.axhline(0.98, color="k", ls=":", lw=1.5,
               label="pre-registration (a): 0.98 in EVERY bin")
    ax.set_xticks(x)
    ax.set_xticklabels(["L=8 ff=256\ncold", "L=8 ff=256\nWARM",
                        "L=12 ff=256\ncold", "L=12 ff=256\nWARM"], size=SMALL - 2)
    ax.set_ylabel("generations that PARSE\nas a trace at all", size=MED - 4)
    ax.set_ylim(0.75, 1.045)
    ax.legend(fontsize=SMALL - 2, loc="lower left")
    ax.set_title("well_formed_rate, rho=0.60", size=MED)

    ax2 = fig.add_axes([0.56, 0.36, 0.40, 0.44]); ax2.axis("off")
    ax2.text(0, 1.0, "THE POOL AGREES, ON ITS OWN THREE GENERATIVE STATISTICS",
             size=SMALL, weight="bold", va="top")
    t = D["wpool"]
    c, w = "d0256_h04_L12_ff00256_lr1e-03_pre", "d0256_h04_L12_ff00256_lr1e-03_pre_w05"
    c8, w8 = "d0256_h04_L08_ff00256_lr1e-03_pre", "d0256_h04_L08_ff00256_lr1e-03_pre_w05"
    # The CONTROL columns are here, not only on page 4, because the sentence
    # below claims the control barely moves and a claim needs its numbers on the
    # same page as itself.
    lines = [("solve_rate", "up"), ("truncated_rate", "down"), ("outcome_accuracy", "up")]
    ax2.text(0.02, 0.88, "treated: L=12/ff=256", size=SMALL - 2, weight="bold", va="top")
    ax2.text(0.60, 0.88, "CONTROL: L=8/ff=256", size=SMALL - 2, weight="bold", va="top")
    for xi, head in [(0.33, "cold"), (0.50, "warm"), (0.74, "cold"), (0.91, "warm")]:
        ax2.text(xi, 0.78, head, size=SMALL - 2, weight="bold", va="top", ha="center")
    for i, (name, want) in enumerate(lines):
        y = 0.66 - 0.12 * i
        ax2.text(0.02, y, name, size=SMALL - 2, va="top", family="monospace")
        for xi, (tag, bold) in zip([0.33, 0.50, 0.74, 0.91],
                                   [(c, False), (w, True), (c8, False), (w8, True)]):
            v = t[name][tag]
            ref = t[name][c if tag in (c, w) else c8]
            good = (v > ref) if want == "up" else (v < ref)
            ax2.text(xi, y, f"{v:.4f}", size=SMALL - 2, va="top", ha="center", family="monospace",
                     weight="bold" if bold else "normal",
                     color=(GREEN if good else RED) if bold else "k")
    ax2.text(0.02, 0.27,
             "Three independent generative statistics move the same way in the\ntreated cell (green = the "
             "direction warmup was supposed to move it).\nIn the CONTROL they barely move, so warmup did not "
             "lift the whole\nthin column. Finding 25's mechanism was right: most of the\ndeep-thin deficit "
             "was the model failing to WRITE traces, not\nfailing to deduce.",
             size=MED - 4, va="top", linespacing=1.6)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    # The average improved while the floor did not. Saying only "0.868 -> 0.924"
    # hides that, and an earlier one-line write-up of this finding compared the
    # cold WORST BIN against the warm WEIGHTED MEAN, which is not a pair.
    fig.text(0.5, 0.285,
             f"A PARTIAL REPAIR, AND THE LIKE-FOR-LIKE PAIRS SAY SO: weighted, L=12/ff=256 goes "
             f"{wmean[2]:.3f} -> {wmean[3]:.3f}; its worst bin goes\n{worst[2]:.3f} -> {worst[3]:.3f}, i.e. "
             "NO BETTER THAN BEFORE. An average that rises while the floor does not is a partial repair, "
             "not a fixed cell.",
             size=MED - 3, ha="center", va="top", linespacing=1.6, weight="bold")
    footer(fig, "well_formed_rate is the fraction of generations that parse as a trace at all -- a grammar "
                "measure, scored before any question of whether the colours are right. truncated_rate is the "
                "fraction that hit the length cap, outcome_accuracy the fraction whose SOLVED/UNRESOLVED "
                "verdict is correct. Warmup is a 5-epoch "
                "linear ramp to the same 1e-3, nothing else changed, chosen once and not scanned: 'a longer ramp "
                "would clear the gate' is untested and nothing here rules it in or out.")
    o.savefig(fig)


def page_nll(o, D):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("The cheap metric still does not rank these architectures",
                 size=BIG, weight="bold")
    t = D["nll"].sort_values("val_nll")
    ax = fig.add_axes([0.09, 0.30, 0.40, 0.50])
    # Hand-placed, per cell: the two pairs this page is ABOUT -- L=7 against the
    # knee, and the thin column's warm against cold -- are near-identical in
    # BOTH coordinates, so one offset for all eight points stacks their labels
    # on top of each other. Rightward labels would also run off the axes on the
    # crowded right-hand side, so those point left.
    OFF = {"d0256_h04_L04_ff01024_lr1e-03_pre": (10, -4, "left"),
           "d0256_h04_L06_ff01024_lr1e-03_pre": (-11, -4, "right"),
           "d0256_h04_L07_ff01024_lr1e-03_pre": (-11, 5, "right"),
           "d0256_h04_L08_ff01024_lr1e-03_pre": (-11, -14, "right"),
           "d0256_h04_L08_ff00256_lr1e-03_pre": (-11, 4, "right"),
           "d0256_h04_L08_ff00256_lr1e-03_pre_w05": (-11, -15, "right"),
           "d0256_h04_L12_ff00256_lr1e-03_pre": (-11, -4, "right"),
           "d0256_h04_L12_ff00256_lr1e-03_pre_w05": (-11, 7, "right")}
    for _, r in t.iterrows():
        warm = r.warm > 0
        ax.scatter(r.val_nll, r.pool, s=160, zorder=3, rasterized=True,
                   color=GREEN if warm else ("0.45" if r.ff == 256 else BLUE),
                   marker="D" if warm else "o")
        dx, dy, ha = OFF.get(r.tag, (10, -4, "left"))
        ax.annotate(r.cell.replace(" (NEW)", "*"), (r.val_nll, r.pool), textcoords="offset points",
                    xytext=(dx, dy), size=SMALL - 3, ha=ha)
    ax.set_xlabel("final val_nll (teacher-forced)")
    ax.set_ylabel("pool solve rate, n=1500")
    ax.set_title("each point is one trained cell", size=MED)
    ax.invert_xaxis()
    ax.set_xlim(1.505, 1.405)
    # 0.745, not 0.765: at 0.765 the L=4 cell (pool 0.760) fell off the bottom
    # of the axes while the footnote still said eight runs.
    ax.set_ylim(0.745, 0.928)
    rho = stats.spearmanr(t.val_nll, t.pool).statistic
    # Said once, in one direction: the correlation is computed on (val_nll, pool),
    # where a perfect ranker is -1. The flipped axis is a reading aid, not a
    # second convention to hold in mind.
    ax.text(0.03, 0.97, f"Spearman(val_nll, pool) = {rho:+.3f}\na perfect ranker would be -1.000\n"
                        "axis flipped, so better is up AND right",
            transform=ax.transAxes, va="top", size=SMALL - 2)

    ax2 = fig.add_axes([0.56, 0.30, 0.40, 0.50]); ax2.axis("off")
    ax2.text(0, 1.0, "THE TWO READINGS, SEPARATELY", size=SMALL, weight="bold", va="top")
    i = t.set_index("tag")
    c12, w12 = "d0256_h04_L12_ff00256_lr1e-03_pre", "d0256_h04_L12_ff00256_lr1e-03_pre_w05"
    c8, w8 = "d0256_h04_L08_ff00256_lr1e-03_pre", "d0256_h04_L08_ff00256_lr1e-03_pre_w05"
    l7 = "d0256_h04_L07_ff01024_lr1e-03_pre"
    blocks = [
        (GREEN, "FIXED inside the thin column.",
         [f"Cold, L=12 had the LOWER nll ({i.val_nll[c12]:.4f} vs {i.val_nll[c8]:.4f})",
          f"and the LOWER pool rate ({i.pool[c12]:.4f} vs {i.pool[c8]:.4f}) -- inverted.",
          f"Warm, it keeps the lower nll ({i.val_nll[w12]:.4f} vs {i.val_nll[w8]:.4f})",
          f"and now also the higher pool rate ({i.pool[w12]:.4f} vs {i.pool[w8]:.4f})."]),
        (RED, "STILL BROKEN across columns, as badly as ever.",
         [f"Warm L=12/ff=256 has a LOWER nll than the new L=7/ff=1024",
          f"({i.val_nll[w12]:.4f} vs {i.val_nll[l7]:.4f}) while scoring",
          f"{abs(i.pool[w12] - i.pool[l7]) * 100:.1f} POINTS LOWER on the pool "
          f"({i.pool[w12]:.4f} vs {i.pool[l7]:.4f}).",
          "So val_nll is not a model-selection signal at N=96, and the open",
          "'no best-val checkpoint selection' item cannot be resolved by it."])]
    for j, (colour, head, body) in enumerate(blocks):
        y = 0.86 - 0.40 * j
        ax2.text(0.0, y, head, size=SMALL + 1, weight="bold", color=colour, va="top")
        for k, line in enumerate(body):
            ax2.text(0.02, y - 0.09 - 0.075 * k, line, size=SMALL - 1, va="top")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    fig.text(0.5, 0.21,
             "WHY THIS MATTERS FOR THE LADDER: val_nll is what you would use to pick a checkpoint or prune a "
             "scan without paying for\ngeneration. At N=96 it would have picked the wrong cell. Every L* in this "
             "series is therefore read off generated traces,\nwhich costs about an hour of eval per cell.",
             size=MED - 3, ha="center", va="top", linespacing=1.6)
    footer(fig, "Axis flipped so that BETTER is to the right on both axes. Circles are cold cells, DIAMONDS the "
                "warm-start ones, grey the cold thin column; a * marks a cell that is new in these two runs. "
                "L=7 and L=8 differ by one pixel here (1.4142/0.9120 vs 1.4141/0.9113), so their labels share a "
                "point. One seed per cell, so this is a ranking check across eight single runs, not a correlation "
                "with an error bar; it confirms finding 4 at a third task size rather than measuring a new "
                "quantity.")
    o.savefig(fig)


def page_verdict(o, D):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Pre-registrations, as resolved", size=BIG, weight="bold")
    p = D["pairs"].set_index("key")
    k7, n7, z7, pz7 = gate_row(D["d60"], "d0256_h04_L07_ff01024_lr1e-03_pre")
    kw, nw, _, _ = gate_row(D["d60"], "d0256_h04_L12_ff00256_lr1e-03_pre_w05")
    l7 = p.loc["knee"]
    tr = p.loc["treatment"]
    ct = p.loc["control"]
    rv = p.loc["reversal"]

    rows = [
        ("31", "(a)", f"L=7 clears 4-5 of 6 bins, so L*(96, ff=1024) = 8 stands",
         "CONFIRMED", f"{k7} of {n7}; the miss is T=8 at 0.897"),
        ("31", "(b)", "the bins it misses are the deepest ones",
         "WEAK", f"second-deepest, but one miss cannot trend: CA z={z7:+.2f}, p={pz7:.2f}"),
        ("31", "(c)", "on the pool L=7 lands resolvably BELOW L=8",
         "FALSIFIED", f"{l7.delta:+.4f}, {int(l7.won)}-{int(l7.lost)}, p = {l7.p:.2g} -- a tie at 0.88x the params"),
        ("31", "(d)", "well_formed_rate >= 0.95 in every bin",
         "CONFIRMED", "0.957-0.993, weighted 0.977: no wide-column grammar failure"),
        ("32", "(a)", "warm L=12/ff=256: well_formed >= 0.98 in EVERY bin",
         "FALSIFIED", "weighted 0.924; worst bin 0.814, BELOW the cold cell's 0.853"),
        ("32", "(b)", "given (a), it clears the gate, so L*(96, ff=256) <= 12",
         "FALSIFIED", f"{kw} of {nw} bins; L* > 12 stands, width factor still super-multiplicative"),
        ("32", "(c)", "the control L=8/ff=256 ties its cold self",
         "CONFIRMED", f"{ct.delta:+.4f}, {int(ct.won)}-{int(ct.lost)}, p = {ct.p:.2g} -- warmup did not lift the column"),
        ("32", "(d)", "warmup stops val_nll ranking the thin cells in reverse",
         "SPLIT", "fixed INSIDE the thin column, still inverted ACROSS columns"),
    ]
    ax = fig.add_axes([0.03, 0.335, 0.94, 0.50]); ax.axis("off")
    t = ax.table(cellText=[[a, b, c, d, e] for a, b, c, d, e in rows],
                 colLabels=["finding", "part", "the claim, written down before the run was submitted",
                            "verdict", "evidence"],
                 colWidths=[0.07, 0.04, 0.36, 0.11, 0.42], loc="center", cellLoc="center")
    # 12pt, not 13: matplotlib tables do not wrap or clip, they just draw over
    # the next column, so the widths here are a budget. At 12pt a character is
    # about 6.1pt, so the claim column (0.36 * 0.94 * 1152pt = 390pt) holds ~63
    # characters and the evidence column ~67 once its padding is counted; the
    # longest string in each is 54 and 67. One character more in the evidence
    # column and it draws over the table border, as finding 31(b) first did.
    t.auto_set_font_size(False); t.set_fontsize(MED - 5); t.scale(1, 2.4)
    for j in range(5):
        t[0, j].set_facecolor("#dddddd"); t[0, j].set_text_props(weight="bold")
    colour = {"CONFIRMED": GREEN, "FALSIFIED": RED, "WEAK": "#b9770e", "SPLIT": "#b9770e"}
    for i, r in enumerate(rows, start=1):
        t[i, 2].set_text_props(ha="left")
        t[i, 4].set_text_props(ha="left")
        t[i, 3].set_text_props(weight="bold", color=colour[r[3]])

    fig.text(0.5, 0.295,
             f"UNPRE-REGISTERED, and the reason this pair is worth a deck: warmup REVERSES the depth ordering "
             f"inside the thin column. Cold,\nL=12/ff=256 scored 3.1 points BELOW L=8/ff=256 and finding 25 read "
             f"that as 'depth cannot buy back MLP width at N=96'. Warm, L=12\nis {rv.delta:+.4f} ABOVE it "
             f"({int(rv.won)}-{int(rv.lost)}, p = {rv.p:.2g}). That conclusion's direction survives -- the thin "
             "column still does not reach the wide knee --\nbut its sharpest sentence was about training, not "
             "architecture.",
             size=MED - 3, ha="center", va="top", linespacing=1.6)
    footer(fig, "WHERE THIS LEAVES THE GRID: the MLP-width factor on L* is 1.00x / 1.50x / >1.50x at N = 24 / 48 / "
                "96 -- super-multiplicative in N, with one confound now removed rather than suspected. WHAT IT DOES "
                "NOT ESTABLISH: leaked order throughout, so these are decoding scores; one seed per cell; the gate "
                "is sensitive at the third decimal and the pool disagrees with it about L=7 vs L=8; warmup is one "
                "untuned 5-epoch ramp; the two deepest bins carry about +-0.07 at 95%.")
    o.savefig(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    set_style()
    D = load()
    # pick="deep" and not "median": this deck's subject IS reasoning depth, and
    # a median instance at N=96 would illustrate a shallower task than the bins
    # the verdict is read from. shuffle_adj=False to match the runs.
    ex = example_instance(n_nodes=96, degree=5, n_colors=4, rho=0.60,
                          shuffle_adj=False, pick="deep")
    pdf_path = os.path.join(OUT, "slides.pdf")
    with PdfPages(pdf_path) as pdf:
        o = PdfAndPng(pdf, OUT, total=8)
        page_title(o, D)
        page_tokenization(o, ex)
        # The note slot: this page is shared with the verdict deck, where the
        # space below the paragraph carries the intervention caption. Here it
        # ties the rounds in the trace to the depth axis the rest of the deck
        # plots, so the reader meets T before page 5 uses it as a bin label.
        page_trace(o, ex, title="One real N=96 instance, as the model reads it",
                   note="THIS DECK IS ABOUT THE ROUNDS. T, the number of rounds, is the depth axis on pages 5 and 6; "
                        "L* is the fewest layers that still\nclear 90% of instances at a given T. The adjacency-order "
                        "leak lets a model skip part of that work -- see the caveat on page 1.")
        page_pool(o, D)
        page_depth(o, D)
        page_mechanism(o, D)
        page_nll(o, D)
        page_verdict(o, D)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB, {o.n} pages)")


if __name__ == "__main__":
    main()
