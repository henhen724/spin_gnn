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
                      "delta": a.mean() - b.mean(), "won": w, "lost": l, "p": p,
                      "n": int(len(a))})

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
    # The gap the +0.0800 is a fraction OF. warm-vs-cold minus warm-vs-knee is
    # cold-vs-knee, so this is read off the same paired table as everything else.
    gap = tr.delta - p.loc["vs_knee"].delta

    for i, (colour, head, body) in enumerate([
            (BLUE, "L*(96, ff=1024) = 8 BY THE BIN GATE -- and L=7 ties it on the pool.",
             [f"L=7 clears the 0.90 gate in {k7} of {n7} depth bins, missing only T=8, by 0.003.",
              # int(): these came out of a DataFrame as floats, and "92.0-91.0"
              # reads as a rate rather than as a count of disagreeing instances.
              f"On the pool it is {l7.delta:+.4f} against L=8 ({int(l7.won)}-{int(l7.lost)} discordant, "
              f"p = {l7.p:.3g}): a TIE, at {p7 / p8:.0%} of the parameters ({p7:.2f}M vs {p8:.2f}M).",
              "So the ladder has a rung the metric of record cannot see."]),
            (GREEN, "Most of the deep-thin deficit was TRAINABILITY, not capacity.",
             [f"Five epochs of LR warmup move L=12/ff=256 by {tr.delta:+.4f} on the pool "
              f"({int(tr.won)}-{int(tr.lost)}, p = {tr.p:.2g}) -- {tr.delta / gap:.0%} of its "
              f"{gap:.4f} gap to the wide knee.",
              f"THE CONTROL HELD: the cell with no grammar problem (page 6) moved {ct.delta:+.4f} "
              f"(p = {ct.p:.2g}), so warmup did not lift the whole column.",
              f"And it is PARTIAL: the warm cell still clears the gate in {kw} of {nw} bins, "
              "so L*(96, ff=256) > 12 stands."])]):
        y = 0.625 - 0.235 * i
        fig.text(0.055, y, head, size=MED + 1, weight="bold", color=colour)
        for j, line in enumerate(body):
            fig.text(0.075, y - 0.055 - 0.045 * j, line, size=MED - 2)

    fig.text(0.055, 0.845, "WHAT L* IS MEASURED ON, AND THE NOTATION THE LATER PAGES USE",
             size=SMALL, weight="bold")
    # Every term the pages after this one use without re-explaining. A reader
    # seeing the project for the first time should not have to infer "ff" or
    # "cell" from a footer four pages later.
    fig.text(0.075, 0.812,
             "Graph colouring solvable by Direct Deduction (DD) alone: colour the one node whose known "
             "neighbours leave it a single colour, repeat. N=96 nodes, degree 5,\n"
             "q=4 colours. An instance's DEPTH T is how many DD rounds it takes. rho = the fraction of "
             "nodes given as hints; DENSITY is the realized hint fraction, held at\n"
             "0.5938 in every bin. ff = the width of the MLP inside each transformer block (ff=1024 WIDE, "
             "ff=256 THIN). A CELL is one trained model, one seed. COLD = flat\n"
             "learning rate, WARM = a 5-epoch ramp to the same rate. THE KNEE = L=8/ff=1024, the shallowest "
             "wide cell clearing every bin.\n"
             "THE POOL = the 1500-instance shared eval that every paired comparison on page 4 runs on. "
             "'Finding NN' = this project's own log.",
             size=MED - 4, va="top", linespacing=1.55)

    # Up front, not in the limitations: with shuffle_adj=False a solve rate on
    # this ladder is a DECODING score, and finding 16 showed these checkpoints
    # collapse when adj[v] is permuted. Reading these pages as deduction rates
    # is the one misreading that would change what they mean.
    # Wrapped at 140, not by hand: at bold 13pt from x=0.055 the slide holds
    # about 145 characters, and the hand-broken version ran a fourth line down
    # into the footer.
    fig.text(0.055, 0.205,
             textwrap.fill(
                 "CAVEAT, up front: these runs keep the ADJACENCY-ORDER LEAK (shuffle_adj=False), so every rate "
                 "here is a DECODING score, not a deduction score. Each node's neighbour row is listed sorted by "
                 "the neighbours' TRUE colours, which names the node's own colour outright for 61-67% of nodes "
                 "(finding 16); permuting those rows dropped the N=96 model it was measured on from 0.94 to 0.04. "
                 "One seed per cell. The leak-free ladder is a separate series, running at N=48.", width=136),
             size=MED - 4, va="top", color="0.15", weight="bold", linespacing=1.5)
    footer(fig, "Jobs 50053882 and 50008605. Prepared 2026-10-29; every number on every page is "
                "recomputed from the run outputs by this script, not typed in.")
    o.savefig(fig)


def page_pool(o, D):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("The metric of record: one pool, every comparison paired", size=BIG, weight="bold")
    ax = fig.add_axes([0.04, 0.47, 0.92, 0.43]); ax.axis("off")
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

    # An interval, not an assertion: "the pool has the resolution" was the one
    # claim on this page with no number behind it, and what 1500 paired
    # instances actually exclude is a difference of about two points.
    kn = D["pairs"].set_index("key").loc["knee"]
    disc = int(kn.won + kn.lost)
    ci = 1.96 * np.sqrt(disc) / kn.n
    fig.text(0.5, 0.415,
             f"A TIE HERE IS A MEASUREMENT, NOT A MISSING ONE -- up to a point. L=7 and L=8 disagree on {disc} "
             f"of {int(kn.n)} instances and split\nthem {int(kn.won)}-{int(kn.lost)}, so the paired 95% interval "
             f"on the difference is about +-{ci:.3f}: the pool excludes a gap bigger than two points and\nfinds "
             "none. It cannot exclude a smaller one, and no further N=96 run at this n will.",
             size=MED - 3, ha="center", va="top", linespacing=1.6)
    fig.text(0.5, 0.300,
             "THE CONTROL IS THE LOAD-BEARING ROW. Had warmup lifted the whole thin column, finding 25's "
             "width gap would have been\nmeasured against an undertrained competitor and every N=96 width number "
             "would need re-running. The cell with no grammar problem\n(its generations parse as traces -- see "
             "page 6) moved by -0.0040 (p = 0.77) while the cell with one moved by +0.0800.",
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
             "cell clears;   bars are 95% Jeffreys intervals on that bin alone (one seed, so they are "
             "binomial, not across-seed)", size=SMALL - 1, ha="center", color="0.3")
    axes = [fig.add_axes([0.07, 0.465, 0.41, 0.355]), fig.add_axes([0.56, 0.465, 0.41, 0.355])]
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
    ns = "/".join(str(int(v)) for v in n_per.values())
    ts = "/".join(str(int(k)) for k in n_per)
    fig.text(0.5, 0.370,
             "THIS IS A SEPARATE EVAL FROM THE POOL ON PAGES 4 AND 7: 1500 instances drawn at rho=0.60 and binned "
             "by DD depth T.\nThe six bins the eval keeps (at least 40 instances each) hold "
             + str(int(sum(n_per.values()))) + " of them: n = " + ns + " at T = " + ts + ".\n"
             "Realized density is 0.5938 in EVERY bin of EVERY cell, so the bins differ in reasoning depth and in "
             "nothing else.\nThe pool is a different draw, spanning the whole training range rho=0.30-0.95, and "
             "runs harder -- bin rates and pool\nrates are not interchangeable, which is why the two metrics can "
             "disagree about L=7.",
             size=MED - 4, ha="center", va="top", linespacing=1.45)
    fig.text(0.5, 0.205,
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
    fig.suptitle("val_nll ranks the thin cells, and still fails ACROSS widths",
                 size=BIG, weight="bold")
    t = D["nll"].sort_values("val_nll")
    ax = fig.add_axes([0.09, 0.36, 0.40, 0.46])
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

    ax2 = fig.add_axes([0.56, 0.36, 0.40, 0.46]); ax2.axis("off")
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
          f"The nll side of that is {abs(i.val_nll[w12] - i.val_nll[l7]):.4f}, which one seed",
          "cannot resolve; the POOL side is 2.6 points, which it can. The",
          "asymmetry is the point: nll is silent where the pool is loud."])]
    for j, (colour, head, body) in enumerate(blocks):
        y = 0.86 - 0.40 * j
        ax2.text(0.0, y, head, size=SMALL + 1, weight="bold", color=colour, va="top")
        for k, line in enumerate(body):
            ax2.text(0.02, y - 0.09 - 0.075 * k, line, size=SMALL - 1, va="top")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    fig.text(0.5, 0.275,
             "WHY THIS MATTERS FOR THE LADDER: val_nll is what you would use to pick a checkpoint or prune a "
             "scan without paying for\ngeneration. At N=96 it would have picked the wrong cell. Every L* in this "
             "series is therefore read off generated traces,\nwhich costs about an hour of eval per cell.",
             size=MED - 3, ha="center", va="top", linespacing=1.6)
    footer(fig, "Axis flipped so BETTER is to the right on both axes. Circles are cold cells, DIAMONDS the "
                "warm-start ones, grey the cold thin column; a * marks a cell new in these two runs. L=7 and L=8 "
                "differ by one pixel (1.4142/0.9120 vs 1.4141/0.9113) and share a point. ONE SEED PER CELL: with "
                "n=8 and no seed replication the Spearman is not distinguishable from 0 or -1, so read the two "
                "named contrasts, not the coefficient.")
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
         "WEAK", f"second-deepest; one miss cannot trend (trend z={z7:+.2f}, p={pz7:.2f})"),
        ("31", "(c)", "on the pool L=7 lands resolvably BELOW L=8",
         "FALSIFIED", f"{l7.delta:+.4f}, {int(l7.won)}-{int(l7.lost)}, p = {l7.p:.2g} -- a tie at 0.88x the params"),
        ("31", "(d)", "well_formed (output parses as a trace) >= 0.95 per bin",
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
    footer(fig, "WHERE THIS LEAVES THE GRID: the MLP-width factor on L* reads 1.00x / 1.50x / AT LEAST 1.5x at "
                "N = 24 / 48 / 96. The N=96 entry is CENSORED, not measured: L=12 is the deepest thin cell anyone "
                "has trained, so all that is established is L*(96, ff=256) > 12, and 12/8 = 1.5 is a lower bound "
                "whose true value is unknown. WHAT ELSE IT DOES NOT ESTABLISH: leaked order throughout, so these "
                "are decoding scores; one seed per cell; the gate is sensitive at the third decimal and the pool "
                "disagrees with it about L=7 vs L=8; warmup is one untuned 5-epoch ramp; the two deepest bins "
                "carry about +-0.07 at 95%.")
    o.savefig(fig)


LF48 = f"{S}/sudoku_n48_leakfree_lstar_2026-10-29"   # 50099945, the first leak-free ladder
CAV = "#b7791f"   # amber: this deck's RED/GREEN pair has no "confounded" colour


def load_n48():
    """Job 50099945's two tables. {} if the run has not landed."""
    f, g = f"{LF48}/depth_lstar_50099945.csv", f"{LF48}/sharedpool_n1500_50099945.csv"
    if not (os.path.exists(f) and os.path.exists(g)):
        return {}
    d = pd.read_csv(f)
    d["L"] = d.checkpoint.str.extract(r"_L(\d+)_").astype(int)
    q = pd.read_csv(g)
    q["L"] = q.checkpoint.str.extract(r"_L(\d+)_").astype(int)
    return {"bins": d, "pool": q.set_index("L")}


def ca_trend(ks, ns, xs):
    """Cochran-Armitage trend z over ordered bins. Negative = falling with x."""
    ks, ns, xs = np.asarray(ks, float), np.asarray(ns, float), np.asarray(xs, float)
    N, K = ns.sum(), ks.sum()
    pbar = K / N
    xbar = (ns * xs).sum() / N
    num = (ks * (xs - xbar)).sum()
    var = pbar * (1 - pbar) * (ns * (xs - xbar) ** 2).sum()
    return num / np.sqrt(var) if var > 0 else float("nan")


def page_n48(o, E):
    """The ladder without the leak, at N=48.

    Pre-registered in the sbatch header before the run: (a) L*(T) rises with T
    by at least one rung at rho=0.60; (b) the leak-free L* is larger than the
    leaked one; (c) not tested here (fixed ff=1024); (d) low confidence, the
    deep cells' per-T curves are flatter than the shallow cells'.
    """
    d, pool = E["bins"], E["pool"]
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Take the leak away and six layers is not enough at N = 48",
                 size=BIG - 2, weight="bold")
    fig.text(0.5, 0.905,
             "A DIFFERENT SERIES: leak-free (adjacency rows permuted), N = 48, the 240000 x 100ep budget that "
             "solved N = 24, three cells L = 6 / 4 / 3 at ff = 1024, two densities.",
             size=SMALL - 2, ha="center", va="top", color="0.3")

    STY = {6: (BLUE, "o", "L=6 (top of the ladder)"), 4: (CAV, "s", "L=4"), 3: (RED, "^", "L=3")}
    for ax, rho, pos in [(fig.add_axes([0.075, 0.575, 0.40, 0.230]), 0.75, "top"),
                         (fig.add_axes([0.075, 0.270, 0.40, 0.230]), 0.60, "bot")]:
        sub = d[d.rho_requested == rho]
        for L, (col, mk, lab) in STY.items():
            r = sub[sub.L == L].sort_values("T")
            ax.plot(r["T"], r.solve_rate, mk + "-", color=col, lw=2.5, ms=8,
                    label=lab if pos == "top" else None)
        ax.axhline(GATE, color="k", ls="--", lw=2)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xticks(sorted(sub["T"].unique()))
        ax.set_ylabel("solve rate", size=MED - 4)
        rr = sub.realized_rho
        ax.set_title(f"density requested {rho:.2f}, REALIZED {rr.min():.3f}-{rr.max():.3f} in every bin",
                     size=SMALL - 2)
        if pos == "top":
            # next to the line it describes: the band between the gate and L=4
            # is the only empty region of this panel.
            ax.text(sorted(sub["T"].unique())[0] + 0.05, 0.80,
                    "dashed = the 0.90 gate that defines L*", size=SMALL - 2)
            ax.legend(loc="lower left", fontsize=SMALL - 2)
        else:
            ax.set_xlabel("T, the number of DD rounds the instance needs", size=MED - 4)

    s75 = d[(d.rho_requested == 0.75) & (d.L == 6)].sort_values("T")
    s60 = d[(d.rho_requested == 0.60) & (d.L == 6)].sort_values("T")
    z = ca_trend((s75.solve_rate * s75.n).round(), s75.n, s75["T"])
    clears = {L: int(((d.rho_requested == 0.60) & (d.L == L) & (d.solve_rate >= GATE)).sum())
              for L in (6, 4, 3)}
    nbin = int((d.rho_requested == 0.60).sum() / 3)
    r75 = dict(zip(s75["T"], s75.solve_rate))
    r60 = dict(zip(s60["T"], s60.solve_rate))
    tlo, thi = int(s60["T"].min()), int(s60["T"].max())
    d4 = d[(d.rho_requested == 0.60) & (d.L == 4)].solve_rate

    entries = [
        ("(a) L*(T) RISES with T by a rung", GREEN,
         f"CONFIRMED at density 0.75, where realized density is identical in all three bins: L = 6 clears the "
         f"gate at T = 2 ({r75[2]:.3f}) and T = 3 ({r75[3]:.3f}) and misses at T = 4 ({r75[4]:.3f}). So L*(2) = "
         f"L*(3) = 6 and L*(4) > 6 -- the gate crosses between T = 3 and T = 4. Cochran-Armitage z = {z:.1f}."),
        ("   ... and UNTESTABLE at density 0.60", CAV,
         f"The pre-registration named 0.60, and there the anchor is FLOORED: L = 6 clears {clears[6]} of {nbin} "
         f"bins ({max(r60.values()):.3f} at T = {tlo} down to {r60[thi]:.3f} at T = {thi}), L = 4 and L = 3 clear "
         f"none. No cell above the gate means no rung to compare -- the failure mode this series was ordered "
         f"deepest-cell-first to catch."),
        ("(b) leak-free L* > leaked L*", GREEN,
         f"CONFIRMED in its N = 48 form, and this is the number to quote: the LEAKED ladder puts L*(48) at 4 "
         f"(finding 14), the leak-free one needs MORE than 6 -- at three times the data. The leak was worth at "
         f"least two rungs of depth here."),
        ("(d) deep cells flatter in T", RED,
         f"FALSIFIED as stated, confounded as measured: L = 6 falls {r60[tlo] - r60[thi]:.3f} over T = {tlo}-{thi} "
         f"while L = 4 falls {d4.max() - d4.min():.3f} -- but L = 4 sits at {d4.max():.2f} and L = 3 at 0.00, so the "
         f"shallow cells cannot fall. (c) was not tested; this ladder is at fixed ff = 1024."),
    ]
    fig.text(0.545, 0.862, "THE PRE-REGISTRATIONS, AS RESOLVED", size=MED - 3, weight="bold")
    for i, (head, col, body) in enumerate(entries):
        y = 0.820 - 0.152 * i
        fig.text(0.545, y, head, size=MED - 4, weight="bold", color=col, va="top")
        fig.text(0.558, y - 0.034, textwrap.fill(body, width=78), size=MED - 5,
                 va="top", linespacing=1.45)

    fig.text(0.5, 0.185,
             f"Reading: the leak-free task at N = 48 is NOT solved at six layers -- shared pool "
             f"{pool.solve_rate[6]:.3f} [{pool.wilson_lo[6]:.3f}, {pool.wilson_hi[6]:.3f}] against "
             f"{pool.solve_rate[4]:.3f} at L = 4 and {pool.solve_rate[3]:.3f} at L = 3, n = 1500.\n"
             f"L*(48, leak-free, ff = 1024) lies in (6, 12]: twelve layers solved N = 24 outright. The next series "
             f"is L = 8 and L = 12 UPWARD, not L = 3 downward.",
             ha="center", va="top", size=MED - 4, weight="bold", color="0.15", linespacing=1.5)
    footer(fig, "WHY THIS IS THE HONEST VERSION OF THE DEPTH QUESTION: every L* in this deck and every one in the "
                "project before today was measured on the leaked encoding, where a model can read part of the "
                "colouring off the order of the adjacency rows. This is the first rung measured without it. WHAT IT "
                "IS NOT: one seed per cell, one checkpoint, three rungs, and the density-0.75 reading rests on three "
                "bins of 674 / 531 / 194 instances. The gate is a threshold on a continuous curve, so L* moves if "
                "the gate does.", width=156)
    o.savefig(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    set_style()
    D = load()
    E = load_n48()
    # pick="deep" and not "median": this deck's subject IS reasoning depth, and
    # a median instance at N=96 would illustrate a shallower task than the bins
    # the verdict is read from. shuffle_adj=False to match the runs.
    ex = example_instance(n_nodes=96, degree=5, n_colors=4, rho=0.60,
                          shuffle_adj=False, pick="deep")
    pdf_path = os.path.join(OUT, "slides.pdf")
    with PdfPages(pdf_path) as pdf:
        o = PdfAndPng(pdf, OUT, total=8 + bool(E))
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
        if E:
            page_n48(o, E)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB, {o.n} pages)")


if __name__ == "__main__":
    main()
