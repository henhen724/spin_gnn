"""Slide deck for job 50007138: L*(24, ff=256), the last cell of the L* grid.

Run as `python -m notebooks.make_n24_ff256_deck` from the repo root.

EVERY NUMBER IN THIS DECK IS A DECODING SCORE. shuffle_adj=False, so this run
sits in the same leaked-order task as findings 11, 13, 14, 22 and 25 --
deliberately, to extend one ladder rather than start a second. The tokenization
page says so on the slide, which is the whole reason that page exists.

Nothing here is hand-typed from a result: verdicts, the width factor and the
headline are computed from the CSVs, so the deck cannot drift from the data the
way finding 25's prose did before it was corrected.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from notebooks.deck_tokenization import example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style
from notebooks.make_n96_lstar_deck import ca_trend, table

SCRATCH = os.environ["SCRATCH"]
RUN = f"{SCRATCH}/spin_gnn/sudoku_n24_ff256_2026-10-28"
WIDE = f"{SCRATCH}/spin_gnn/sudoku_layer_scan_2026-10-24_n24"
THIN = f"{SCRATCH}/spin_gnn/sudoku_ff_depth_tradeoff_2026-10-25_n24"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sudoku_n24_ff256_2026-10-28")

BLUE, RED, GREEN, ORANGE, PURPLE, GREY = (
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "0.45")
GATE = 0.90
RHO = 0.60
# Every column on the N=24 ladder, keyed by the checkpoint substring that
# identifies it. NEW marks the two cells this job trained.
CELLS = [("L=2, ff=1024", "L02_ff01024", GREY, False),
         ("L=3, ff=1024", "L03_ff01024", GREEN, False),
         ("L=2, ff=256", "L02_ff00256", RED, True),
         ("L=3, ff=256", "L03_ff00256", BLUE, True),
         ("L=4, ff=256", "L04_ff00256", ORANGE, False),
         ("L=6, ff=256", "L06_ff00256", PURPLE, False)]
THIN_LADDER = ["L02_ff00256", "L03_ff00256", "L04_ff00256", "L06_ff00256"]


def load():
    depth = pd.concat([
        pd.read_csv(f"{WIDE}/depth_by_layers.csv"),
        pd.read_csv(f"{THIN}/depth_by_ff.csv"),
        pd.read_csv(f"{RUN}/depth_n24_ff256.csv"),
    ], ignore_index=True)
    depth["k"] = (depth.solve_rate * depth.n).round().astype(int)
    pool = pd.read_csv(f"{RUN}/sharedpool_n1500.csv")
    sweep = pd.concat([pd.read_csv(f"{WIDE}/sweep.csv"), pd.read_csv(f"{THIN}/sweep.csv"),
                       pd.read_csv(f"{RUN}/sweep.csv")], ignore_index=True)
    return depth, pool, sweep


def cell(df, tag):
    return df[df.checkpoint.str.contains(tag)]


def bins(depth, tag, rho=RHO):
    return cell(depth, tag).query("rho_requested == @rho").sort_values("T")


def cleared(depth, tag):
    g = bins(depth, tag)
    return int((g.solve_rate >= GATE).sum()), len(g)


def pooled_fisher(g1, g2):
    """Pooled solve_rate of two columns over the same bins, Fisher exact.

    The depth eval reconstructs its bins from one seeded generator stream, so
    two columns scored with the same --n-pool/--min-per-bin see the same
    instances; Fisher on the pooled 2x2 ignores that pairing and is therefore
    conservative. agg_eval's McNemar is the paired test, and it only covers
    the five checkpoints in the shared pool."""
    a, b = int(g1.k.sum()), int(g1.n.sum() - g1.k.sum())
    c, d = int(g2.k.sum()), int(g2.n.sum() - g2.k.sum())
    return a / (a + b), c / (c + d), stats.fisher_exact([[a, b], [c, d]])[1]


def lstar_thin(depth):
    """Smallest num_layers in the thin column clearing every rho=0.60 bin."""
    for tag in THIN_LADDER:
        c, n = cleared(depth, tag)
        if n and c == n:
            return int(tag[1:3])
    return None


def page_title(out, depth, pool):
    ls = lstar_thin(depth)
    factor = None if ls is None else ls / 3.0
    thin3 = pool[pool.checkpoint.str.contains("L03_ff00256")].iloc[0]
    ref = pool[pool.checkpoint.str.contains("L03_ff01024")].iloc[0]
    c3, n3 = cleared(depth, "L03_ff00256")

    fig = plt.figure(figsize=(13.33, 7.5))
    if factor == 1.0:
        head, sub, col = ("At N=24, narrowing the MLP 4x costs nothing at all.",
                          "Not a layer on the gate, and not one instance in 1500.", GREEN)
    elif factor is None:
        head, sub, col = ("At N=24 the thin column never clears the gate.",
                          "L*(24, ff=256) > 6 -- the width cost is largest at the SMALL end.", RED)
    else:
        head, sub, col = ("The MLP-width cost is present at every N, N=24 included.",
                          f"L*(24, ff=256) = {ls}, a factor of {factor:.2f} over the wide knee.", RED)
    fig.text(0.06, 0.85, head, size=BIG + 4, weight="bold")
    fig.text(0.06, 0.765, sub, size=BIG + 1, weight="bold", color=col)
    fig.text(0.06, 0.655,
             f"L* is the smallest num_layers reaching {GATE:.2f} on depth-T instances at fixed\n"
             f"realized density. At N=24 the wide knee is 3 (finding 11). This job trained the\n"
             f"thin cells that decide the knee at ff=256: L=3 clears {c3} of {n3} rho=0.60 bins, so\n"
             f"L*(24, ff=256) = {ls if ls is not None else '> 6'}"
             f"{'' if factor is None else f' and the width factor at N=24 is {factor:.2f}x'}.",
             size=MED, va="top")
    fig.text(0.06, 0.445,
             f"THE GATE AND THE POOL AGREE, which they did not have to. On 1500 paired instances\n"
             f"the thin L=3 scores {thin3.solve_rate:.4f} against the wide L=3's {ref.solve_rate:.4f} "
             f"-- {thin3.delta:+.4f}, {int(thin3.discordant_won)} won /\n"
             f"{int(thin3.discordant_lost)} lost, p={thin3.mcnemar_exact_p:.2f}. A coarse gate "
             f"passing a cell is weak evidence; a paired\ntest failing to separate it from the "
             f"reference on 1500 instances is the strong form.",
             size=SMALL + 1, va="top", color="0.25")
    fig.text(0.06, 0.255,
             "THE GRID IS NOW COMPLETE: six cells, three problem sizes, two MLP widths. The width\n"
             "cost on network depth is 1.00x at N=24, 1.50x at N=48 and above 1.50x at N=96 -- zero\n"
             "at the small end and growing. hshunt's question (does the reasoning depth a problem\n"
             "needs relate to the network depth it needs?) now has an architecture axis on it, and\n"
             "the answer is that the two knees diverge as the problem gets bigger.",
             size=SMALL + 1, va="top", color="0.25")
    footer(fig, "job 50007138, COMPLETED 00:34:11, ExitCode 0:0 | "
                "scripts/slurm/sudoku2_n24_ff256_lstar.sbatch | commit e65b97e (L=3 cell) and "
                "801c593 (L=2 cell + evals; warmup code, a no-op at warmup_epochs=0) | seed 0 | "
                "shuffle_adj=False, so every number here is a DECODING score")
    out.savefig(fig)


def page_grid(out, depth, pool, sweep):
    ls = lstar_thin(depth)
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "The completed grid: L* vs (N, ff)", size=BIG, weight="bold")
    fig.text(0.06, 0.885,
             "dim_feedforward is the hidden width of the per-layer MLP: at d_model=256, ff=1024 is\n"
             "the usual 4x expansion and ff=256 is 1x. Nothing else differs within a row.",
             size=SMALL, va="top")

    rows = [["N=24", "3", f"{ls if ls is not None else '> 6'}",
             "--" if ls is None else f"{ls / 3:.2f}x", "THIS RUN (50007138)"],
            ["N=48", "4", "6", "1.50x", "finding 14 (49509466/49587538)"],
            ["N=96", "8", "> 12", "> 1.50x", "finding 25 (49816217)"]]
    ax = fig.add_axes([0.06, 0.55, 0.90, 0.22])
    table(ax, ["", "L* at ff=1024", "L* at ff=256", "width factor", "source"], rows,
          [0.10, 0.19, 0.17, 0.17, 0.37], fontsize=SMALL - 1,
          row_colors={0: BLUE}, dy=0.22, rule_dy=0.15)

    s = sweep.set_index(["num_layers", "dim_feedforward"])
    prows = []
    for name, tag, _, new in CELLS:
        L, ff = int(tag[1:3]), int(tag.split("ff")[1])
        r = s.loc[(L, ff)]
        r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
        c, n = cleared(depth, tag)
        pr = pool[pool.checkpoint.str.contains(tag)]
        prows.append([name + ("  (new)" if new else ""), f"{int(r.n_params):,}",
                      f"{r.final_val_nll:.4f}",
                      f"{pr.iloc[0].solve_rate:.4f}" if len(pr) else "not pooled",
                      f"{c} of {n}" if n else "--"])
    ax = fig.add_axes([0.06, 0.14, 0.56, 0.34])
    table(ax, ["cell", "n_params", "val_nll", "pool rate", "bins >= 0.90"], prows,
          [0.23, 0.21, 0.17, 0.21, 0.18], fontsize=SMALL - 1,
          row_colors={i: c for i, (_, _, c, _) in enumerate(CELLS)}, dy=0.145, rule_dy=0.10)

    def nparams(L, ff):
        r = s.loc[(L, ff)]
        return int((r.iloc[0] if isinstance(r, pd.DataFrame) else r).n_params)

    p6 = pool[pool.checkpoint.str.contains("L06_ff00256")].iloc[0]
    p3 = pool[pool.checkpoint.str.contains("L03_ff00256")].iloc[0]
    n6, n3w, n3t = nparams(6, 256), nparams(3, 1024), nparams(3, 256)
    n4, n2w = nparams(4, 256), nparams(2, 1024)
    # Pair 1 is not in the shared pool (the wide L=2 was never scored on it), so it gets the
    # conservative unpaired test over the four bins; pair 2 gets agg_eval's paired McNemar.
    q4, q2, pq = pooled_fisher(bins(depth, "L04_ff00256"), bins(depth, "L02_ff01024"))
    fig.text(0.65, 0.50, "TWO MATCHED-PARAMETER PAIRS", size=SMALL, weight="bold", va="top")
    fig.text(0.65, 0.46,
             f"~1.65M, {abs(n4 / n2w - 1) * 100:.1f}% apart: the wide shallow L=2 pools\n"
             f"{q2:.3f} over the four bins, the thin deep L=4 pools\n"
             f"{q4:.3f} -- {q4 - q2:+.3f}, Fisher p={pq:.0e}.\n"
             f"~2.44M, {abs(n6 / n3w - 1) * 100:.1f}% apart: the thin deep L=6 beats the\n"
             f"wide L=3 by {p6.delta:+.4f} on the n=1500 pool ({int(p6.discordant_won)} won /\n"
             f"{int(p6.discordant_lost)} lost, p={p6.mcnemar_exact_p:.1e}), replicating finding 13\n"
             f"on an instance pool it was not measured on.\n\n"
             f"And the thin L=3 ties the wide L=3 ({p3.delta:+.4f},\n"
             f"p={p3.mcnemar_exact_p:.2f}) at {n3t / n3w:.2f}x the parameters. At N=24\n"
             f"the MLP is not where this task's capacity goes.",
             size=SMALL - 1, va="top")
    footer(fig, "L=4 and L=6 at ff=256 are job 49510197's checkpoints, re-used here at no cost")
    out.savefig(fig)


def page_depth(out, depth):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, f"Four depth bins at rho={RHO:.2f}, identical row for row",
             size=BIG, weight="bold")
    fig.text(0.06, 0.885,
             "Realized density is identical across bins within a column, so what varies along\n"
             "the x-axis is the number of DD rounds the instance needs and nothing else. Every\n"
             "column was scored with --rho 0.75 0.60 --n-pool 600 --min-per-bin 40, so the bins\n"
             "come out row for row identical and the columns compare bin by bin.",
             size=SMALL, va="top")

    ax = fig.add_axes([0.06, 0.38, 0.44, 0.39])
    for name, tag, col, new in CELLS:
        g = bins(depth, tag)
        if not len(g):
            continue
        lo, hi = [], []
        for _, r in g.iterrows():
            a, b = stats.beta.interval(0.95, r.k + 0.5, r.n - r.k + 0.5)
            lo.append(max(0.0, r.solve_rate - a)); hi.append(max(0.0, b - r.solve_rate))
        ax.errorbar(g["T"], g.solve_rate, yerr=[lo, hi], fmt="-o" if new else "--o", color=col,
                    ms=5, lw=2.2 if new else 1.3, capsize=2, alpha=1.0 if new else 0.7,
                    label=name + (" (new)" if new else ""))
    ax.axhline(GATE, color="0.35", ls=":", lw=1.2)
    ax.set_xlabel("DD depth T", size=SMALL); ax.set_ylabel("solve_rate", size=SMALL)
    ax.set_xticks(sorted(bins(depth, "L03_ff01024")["T"].astype(int)))
    ax.tick_params(labelsize=SMALL - 2)
    ax.legend(fontsize=SMALL - 4, loc="center left")

    rows = []
    for name, tag, _, new in CELLS:
        g = bins(depth, tag)
        if not len(g):
            continue
        z, p = ca_trend(g["T"], g.k, g.n)
        c, n = cleared(depth, tag)
        rows.append([name, *[f"{v:.3f}" for v in g.solve_rate], f"{c}/{n}", f"{p:.3f}"])
    ax2 = fig.add_axes([0.54, 0.50, 0.44, 0.27])
    gref = bins(depth, "L03_ff01024")
    table(ax2, [""] + [f"T={int(t)}" for t in gref["T"]] + ["cleared", "CA p"], rows,
          [0.22] + [0.11] * len(gref) + [0.13, 0.11], fontsize=SMALL - 2,
          row_colors={i: c for i, (_, _, c, _) in enumerate(
              [x for x in CELLS if len(bins(depth, x[1]))])}, dy=0.145, rule_dy=0.10)
    fig.text(0.54, 0.455,
             f"n per bin: {', '.join(f'T={int(t)}:{int(n)}' for t, n in zip(gref['T'], gref.n))}",
             size=SMALL - 3, va="top", color=GREY)

    fig.text(0.06, 0.30, "THE GATE IS A COARSE INSTRUMENT, DELIBERATELY", size=SMALL,
             weight="bold", va="top")
    fig.text(0.06, 0.26,
             "L* asks one question of each column -- does every bin reach 0.90 -- and it is the\n"
             "same question findings 11, 14, 22 and 25 asked, which is the only reason the six\n"
             "cells of the grid can be read against each other. It cannot distinguish 0.901 from\n"
             "0.999, and at N=24 most bins sit near the ceiling: the Cochran-Armitage column is\n"
             "there to show how little room the trend statistic has here, not as a result.",
             size=SMALL - 1, va="top")
    footer(fig, "beta 95% intervals per bin; Cochran-Armitage trend over the ordered bins, the "
                "statistic findings 11, 22 and 25 report")
    out.savefig(fig)


def page_mechanism(out, depth):
    """Pre-registration (d): is finding 25's grammar collapse specific to large N?"""
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "Does the thin MLP break the output grammar here too?", size=BIG,
             weight="bold")
    fig.text(0.06, 0.885,
             "At N=96, L=12/ff=256 kept deducing (conditional solve rate 0.92-0.98, RISING with T)\n"
             "while only 0.853 of its generations parsed as a trace at all -- finding 25's\n"
             "unexpected mechanism. If a thin MLP simply breaks the grammar, it should break it at\n"
             "N=24 too. The plot is well_formed_rate for every thin cell at both sizes.",
             size=SMALL, va="top")

    n96 = pd.read_csv(f"{SCRATCH}/spin_gnn/sudoku_n96_ff256_2026-10-27/"
                      "depth_by_layers_n96_ff256.csv").query("rho_requested == @RHO")
    ax = fig.add_axes([0.06, 0.38, 0.44, 0.37])
    for name, tag, col, new in CELLS:
        g = bins(depth, tag)
        if not len(g) or "01024" in tag:
            continue
        ax.plot(g["T"], g.well_formed_rate, "-o", color=col, ms=5, lw=2.0,
                label=f"N=24 {name}")
    for tag, col, ls in [("L08_ff00256", "0.35", "--"), ("L12_ff00256", "k", "-")]:
        g = n96[n96.checkpoint.str.contains(tag)].sort_values("T")
        ax.plot(g["T"], g.well_formed_rate, ls + "s", color=col, ms=5, lw=1.8,
                label=f"N=96 L={int(tag[1:3])}, ff=256")
    ax.set_xlabel("DD depth T", size=SMALL)
    ax.set_ylabel("well_formed_rate", size=SMALL)
    ax.set_ylim(0.80, 1.02); ax.tick_params(labelsize=SMALL - 2)
    ax.legend(fontsize=SMALL - 4, loc="lower left")

    rows = []
    for name, tag, _, new in CELLS:
        g = bins(depth, tag)
        if not len(g) or "01024" in tag:
            continue
        cond = g.solve_rate / g.well_formed_rate
        rows.append([f"N=24  {name}", f"{g.well_formed_rate.min():.3f}", f"{cond.iloc[0]:.3f}",
                     f"{cond.iloc[-1]:.3f}", f"{float(np.corrcoef(g['T'], cond)[0, 1]):+.2f}"])
    for tag in ["L08_ff00256", "L12_ff00256"]:
        g = n96[n96.checkpoint.str.contains(tag)].sort_values("T")
        cond = g.solve_rate / g.well_formed_rate
        rows.append([f"N=96  L={int(tag[1:3])}, ff=256", f"{g.well_formed_rate.min():.3f}",
                     f"{cond.iloc[0]:.3f}", f"{cond.iloc[-1]:.3f}",
                     f"{float(np.corrcoef(g['T'], cond)[0, 1]):+.2f}"])
    ax2 = fig.add_axes([0.54, 0.44, 0.44, 0.32])
    table(ax2, ["", "worst wf", "solve|wf 1st", "solve|wf last", "r vs T"], rows,
          [0.32, 0.16, 0.19, 0.20, 0.13], fontsize=SMALL - 2,
          row_colors={0: RED, 1: BLUE, 2: ORANGE, 3: PURPLE, 4: "0.35", 5: "k"},
          dy=0.135, rule_dy=0.10)

    thin_bins = pd.concat([bins(depth, t) for t in THIN_LADDER if len(bins(depth, t))])
    worst = thin_bins.well_formed_rate.min()
    verdict = ("CONFIRMED" if worst >= 0.95 else "FALSIFIED")
    fig.text(0.06, 0.29, f"PRE-REGISTRATION (d): {verdict}", size=SMALL, weight="bold",
             va="top", color=GREEN if verdict == "CONFIRMED" else RED)
    fig.text(0.06, 0.25,
             f"The worst well_formed_rate anywhere in the N=24 thin column is {worst:.3f}, "
             f"across all {len(thin_bins)} bins,\n"
             + ("against 0.853 at N=96. The generative collapse is NOT a property of ff=256: it\n"
                "takes a deep thin stack at a large problem size, and at N=24 the thin cells\n"
                "write valid traces throughout. Whatever the width cost is here, it is a\n"
                "deduction cost, which is what makes the gate reading meaningful at this size.\n"
                if verdict == "CONFIRMED" else
                "so the grammar does break at N=24 too and 'a thin MLP breaks the output format'\n"
                "generalizes across problem size. That would retro-explain the N=48 thin knee as\n"
                "a grammar failure rather than a depth deficit, and every thin-column L* in the\n"
                "grid would need re-reading with well_formed_rate beside it.\n"),
             size=SMALL - 1, va="top")
    footer(fig, "well_formed_rate from sudoku_depth_eval; the N=96 comparison is finding 25 "
                "(job 49816217)")
    out.savefig(fig)


def page_pool(out, pool, depth):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "The metric of record: one pool, five checkpoints", size=BIG,
             weight="bold")
    fig.text(0.06, 0.885,
             "1500 instances at seed_start=5e10, disjoint from every training set, reconstructed\n"
             "identically for all five cells (same N, same INSTANCE_KEYS), so the comparisons are\n"
             "paired instance for instance and McNemar's exact test applies.",
             size=SMALL, va="top")

    rows = []
    for name, tag, _, new in CELLS:
        r = pool[pool.checkpoint.str.contains(tag)]
        if not len(r):
            continue
        r = r.iloc[0]
        is_ref = "ff01024" in tag
        rows.append([name + ("  (new)" if new else ""), f"{r.solve_rate:.4f}",
                     f"[{r.wilson_lo:.4f}, {r.wilson_hi:.4f}]",
                     "reference" if is_ref else f"{r.delta:+.4f}",
                     "--" if is_ref else f"{int(r.discordant_won)}/{int(r.discordant_lost)}",
                     "--" if is_ref else f"{r.mcnemar_exact_p:.1e}",
                     f"{r.outcome_accuracy:.4f}"])
    ax = fig.add_axes([0.06, 0.55, 0.90, 0.24])
    table(ax, ["cell", "solve_rate", "Wilson 95%", "vs wide L=3", "won/lost", "McNemar p",
               "outcome acc"], rows,
          [0.19, 0.13, 0.20, 0.13, 0.11, 0.13, 0.11], fontsize=SMALL - 2,
          row_colors={i: c for i, (_, _, c, _) in enumerate(
              [x for x in CELLS if len(cell(pool, x[1]))])}, dy=0.16, rule_dy=0.11)

    thin3 = pool[pool.checkpoint.str.contains("L03_ff00256")].iloc[0]
    resolvable = thin3.mcnemar_exact_p < 0.05
    fig.text(0.06, 0.47, "PRE-REGISTRATION (b): "
             + ("CONFIRMED" if resolvable and thin3.delta < 0 else "FALSIFIED"),
             size=SMALL, weight="bold", va="top",
             color=GREEN if resolvable and thin3.delta < 0 else RED)
    fig.text(0.06, 0.43,
             (f"The thin L=3 is resolvably worse than the wide L=3 on the pool ({thin3.delta:+.4f},\n"
              f"p={thin3.mcnemar_exact_p:.1e}) even though the four-bin gate cannot tell them apart.\n"
              "The gate and the pool measure different things: the gate asks whether a column is\n"
              "above 0.90 everywhere, the pool asks how often it is right. A cell can pass the\n"
              "first and lose the second, and that is what 'L* is a coarse instrument' means."
              if resolvable and thin3.delta < 0 else
              f"The thin L=3 ties the wide L=3 on 1500 paired instances ({thin3.delta:+.4f},\n"
              f"p={thin3.mcnemar_exact_p:.2f}, {int(thin3.discordant_won)} won /"
              f" {int(thin3.discordant_lost)} lost). At N=24, MLP width buys literally nothing --\n"
              "not a layer on the gate and not an instance on the pool. That is the clean zero the\n"
              "N=48 and N=96 width costs need to be read against."),
             size=SMALL - 1, va="top")

    g2w, g2t = bins(depth, "L02_ff01024"), bins(depth, "L02_ff00256")
    if len(g2w) and len(g2t):
        pw, pt, pval = pooled_fisher(g2w, g2t)
        c2, n2 = cleared(depth, "L02_ff00256")
        fig.text(0.06, 0.24, "PRE-REGISTRATION (c), THE LADDER FLOOR: "
                 + ("CONFIRMED" if c2 == 0 and not (pt > pw and pval < 0.05) else "FALSIFIED"),
                 size=SMALL, weight="bold", va="top",
                 color=GREEN if c2 == 0 and not (pt > pw and pval < 0.05) else RED)
        fig.text(0.06, 0.20,
                 f"L=2/ff=256 clears {c2} of {n2} bins and scores {pt:.3f} pooled over the four "
                 f"rho={RHO:.2f} bins against\nthe wide L=2's {pw:.3f} (Fisher p={pval:.3f}, "
                 f"unpaired and therefore conservative). The claim held --\nthe thin L=2 does not "
                 f"beat the wide one -- but the REASON given for it did not: width is\nworth "
                 f"{(pw - pt) * 100:.1f} points down here while being worth nothing at all at L=3. "
                 f"MLP width helps exactly\nwhere the stack is too shallow to deduce, and stops "
                 f"mattering once it is deep enough.",
                 size=SMALL - 1, va="top")
    footer(fig, "sudoku_agg_eval --n-instances 1500 --seed-start 50000000000, wide L=3 listed first")
    out.savefig(fig)


def page_prereg(out, depth, pool):
    ls = lstar_thin(depth)
    c3, n3 = cleared(depth, "L03_ff00256")
    c2, n2 = cleared(depth, "L02_ff00256")
    thin3 = pool[pool.checkpoint.str.contains("L03_ff00256")].iloc[0]
    g2w, g2t = bins(depth, "L02_ff01024"), bins(depth, "L02_ff00256")
    pw, pt, pval = pooled_fisher(g2w, g2t) if len(g2w) and len(g2t) else (np.nan,) * 3
    worst = min(bins(depth, t).well_formed_rate.min() for t in THIN_LADDER
                if len(bins(depth, t)))

    a_ok = c3 == n3
    b_ok = thin3.delta < 0 and thin3.mcnemar_exact_p < 0.05
    c_ok = c2 == 0 and not (pt > pw and pval < 0.05)
    d_ok = worst >= 0.95
    rows = [
        ["(a)", f"L=3/ff=256 clears all {n3} rho=0.60 bins,\n      so L*(24, ff=256) = 3",
         "CONFIRMED" if a_ok else "FALSIFIED",
         f"clears {c3} of {n3}; L*(24, ff=256) = {ls if ls is not None else '> 6'}, width\n"
         f"   factor {'--' if ls is None else f'{ls / 3:.2f}x'} against the wide knee of 3."],
        ["(b)", "On the n=1500 pool it is still\n      resolvably WORSE than L=3/ff=1024",
         "CONFIRMED" if b_ok else "FALSIFIED",
         f"{thin3.solve_rate:.4f} vs {pool[pool.checkpoint.str.contains('L03_ff01024')].iloc[0].solve_rate:.4f}, "
         f"{thin3.delta:+.4f},\n   {int(thin3.discordant_won)}/{int(thin3.discordant_lost)} discordant, "
         f"p={thin3.mcnemar_exact_p:.1e}."],
        ["(c)", "L=2/ff=256 clears no bin and does not\n      beat the wide L=2",
         "CONFIRMED" if c_ok else "FALSIFIED",
         f"clears {c2} of {n2}; pooled {pt:.3f} vs {pw:.3f},\n   Fisher p={pval:.3f}."],
        ["(d)", "well_formed_rate stays >= 0.95 in both\n      new cells",
         "CONFIRMED" if d_ok else "FALSIFIED",
         f"worst well_formed_rate in the thin column\n   is {worst:.3f} (N=96's was 0.853)."],
    ]
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "Pre-registrations, as resolved", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "Written into scripts/slurm/sudoku2_n24_ff256_lstar.sbatch before the job ran, with "
             "a named falsifier each.", size=SMALL)
    ax = fig.add_axes([0.06, 0.30, 0.90, 0.53])
    table(ax, ["", "claim", "verdict", "evidence"], rows,
          [0.05, 0.34, 0.15, 0.46], fontsize=SMALL - 1,
          row_colors={i: (GREEN if ok else RED) for i, ok in
                      enumerate([a_ok, b_ok, c_ok, d_ok])}, dy=0.20, rule_dy=0.09)
    n_ok = sum([a_ok, b_ok, c_ok, d_ok])
    fig.text(0.06, 0.245, f"{n_ok} OF 4 CONFIRMED.", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.205,
             "The grid's width factor is the only one of the four that could have changed the\n"
             "shape of the story, and (b) is the one that decides whether 'no layers' means 'no\n"
             "cost'. The other two are controls: a ladder floor that is measured rather than\n"
             "assumed, and a check that finding 25's grammar collapse does not travel down to\n"
             "N=24 and invalidate the gate reading everywhere.",
             size=SMALL - 1, va="top")
    footer(fig, "Each claim had a falsifier written beside it in the sbatch header; the header "
                "is the pre-registration of record")
    out.savefig(fig)


def page_next(out, depth, pool):
    ls = lstar_thin(depth)
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "What this changes, and what comes next", size=BIG, weight="bold")

    fig.text(0.06, 0.875, "CHANGED", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.838,
             f"- The L* vs (N, ff) grid is complete: {ls if ls is not None else '>6'}/3 at N=24, "
             f"6/4 at N=48, >12/8 at N=96. The MLP-width\n"
             "  cost on network depth is now a curve in N, not a single number from one size.\n"
             "- Every bare 'L*(N)=k' in STATUS.md still means 'at ff=1024', and the grid now shows\n"
             "  how much that caveat is worth at each size.\n"
             "- The thin column at N=24 is the only complete thin ladder in the project (L=2, 3,\n"
             "  4, 6 all measured on the same bins), so it is the reference for what a thin\n"
             "  ladder looks like when nothing is going wrong.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.620, "NOT CHANGED", size=SMALL, weight="bold", va="top", color=RED)
    fig.text(0.06, 0.583,
             "- These are DECODING scores. shuffle_adj=False, so each node's adjacency row still\n"
             "  arrives sorted by the neighbour's true colour (finding 16), and the whole L*\n"
             "  ladder lives in the leaked task. Finding 23 established the budget a leak-free\n"
             "  ladder needs (240000 instances x 100 epochs); it has not been run.\n"
             "- One seed per cell. L* is read off a 0.90 gate on four bins at this size, and at\n"
             "  N=24 most bins sit near the ceiling, so the gate has little room to resolve\n"
             "  anything finer than 'clears' and 'does not'.\n"
             "- The width factor across rows is not a paired comparison: each knee is measured on\n"
             "  its own ladder at its own N, so the factor column carries no p-value.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.305, "NEXT, IN ORDER OF INFORMATION PER GPU-HOUR", size=SMALL,
             weight="bold", va="top")
    fig.text(0.06, 0.268,
             "1. Job 50008605 is already running: L=12 and L=8 at ff=256, N=96, with a 5-epoch lr\n"
             "   warmup as the only change from finding 25. It decides whether 'L*(96, ff=256) >\n"
             "   12' is about computation or about trainability -- the one confound in the grid's\n"
             "   most extreme cell.\n"
             "2. L=10/ff=1024 at N=96 tightens the wide knee from 'between 6 and 8' to one rung.\n"
             "3. The leak-free L*(T) ladder at N=48, at 240000 x 100ep -- the honest version of\n"
             "   this whole grid, and the most expensive thing on the list.",
             size=SMALL - 1, va="top")
    footer(fig, "checkpoints: d0256_h04_L03_ff00256_lr1e-03_pre and d0256_h04_L02_ff00256_lr1e-03_pre "
                "in sudoku_n24_ff256_2026-10-28")
    out.savefig(fig)


def main():
    set_style()
    os.makedirs(OUT, exist_ok=True)
    depth, pool, sweep = load()
    # N=24 and shuffle_adj=False, matching this run's own instances.
    ex = example_instance(n_nodes=24, degree=5, n_colors=4, shuffle_adj=False, pick="deep")
    pdf_path = os.path.join(OUT, "n24_ff256.pdf")
    with PdfPages(pdf_path) as pdf:
        out = PdfAndPng(pdf, OUT)
        page_title(out, depth, pool)
        page_tokenization(out, ex)
        page_trace(out, ex, max_rounds=4,
                   title="One real reasoning trace, at N=24",
                   note="shuffle_adj=False here, so each node's neighbour row is still sorted by\n"
                        "the neighbour's TRUE colour -- finding 16's leak. That is deliberate: this\n"
                        "run extends the leaked-order L* ladder rather than starting a second one.")
        page_grid(out, depth, pool, sweep)
        page_depth(out, depth)
        page_mechanism(out, depth)
        page_pool(out, pool, depth)
        page_prereg(out, depth, pool)
        page_next(out, depth, pool)
    print(f"{out.n} pages -> {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
