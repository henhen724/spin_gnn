"""Slide deck for job 49816217: narrowing the MLP at N=96, and L* vs (N, ff).

Run as `python -m notebooks.make_ff256_deck` from the repo root.

EVERY NUMBER IN THIS DECK IS A DECODING SCORE. shuffle_adj=False, so this run
sits in the same leaked-order task as findings 14 and 22 -- deliberately, to
extend one ladder rather than start a second. The tokenization page says so on
the slide, which is the whole reason that page exists.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from notebooks.deck_tokenization import MONO, example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style
from notebooks.make_n96_lstar_deck import ca_trend, split_test, table

SCRATCH = os.environ["SCRATCH"]
RUN = f"{SCRATCH}/spin_gnn/sudoku_n96_ff256_2026-10-27"
WIDE = f"{SCRATCH}/spin_gnn/sudoku_n96_layer_scan_2026-10-26"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sudoku_n96_ff256_2026-10-28")

BLUE, RED, GREEN, ORANGE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "0.45"
GATE = 0.90
# The three columns compared, keyed by the checkpoint substring that identifies them.
CELLS = [("L=8, ff=1024", "L08_ff01024", GREEN),
         ("L=8, ff=256", "L08_ff00256", ORANGE),
         ("L=12, ff=256", "L12_ff00256", RED)]


def load():
    thin = pd.read_csv(f"{RUN}/depth_by_layers_n96_ff256.csv")
    wide = pd.read_csv(f"{WIDE}/depth_by_layers_n96.csv")
    depth = pd.concat([wide[wide.checkpoint.str.contains("L08_ff01024")], thin],
                      ignore_index=True)
    depth["k"] = (depth.solve_rate * depth.n).round().astype(int)
    pool = pd.read_csv(f"{RUN}/sharedpool_n1500.csv")
    sweep = pd.concat([pd.read_csv(f"{WIDE}/sweep.csv"), pd.read_csv(f"{RUN}/sweep.csv")],
                      ignore_index=True)
    return depth, pool, sweep


def cell(df, tag):
    return df[df.checkpoint.str.contains(tag)]


def bins60(depth, tag):
    return cell(depth, tag).query("rho_requested == 0.60").sort_values("T")


def cleared(depth, tag):
    g = bins60(depth, tag)
    return int((g.solve_rate >= GATE).sum()), len(g)


def page_title(out, pool, sweep):
    ref = pool[pool.checkpoint.str.contains("L08_ff01024")].iloc[0]
    thin8 = pool[pool.checkpoint.str.contains("L08_ff00256")].iloc[0]
    thin12 = pool[pool.checkpoint.str.contains("L12_ff00256")].iloc[0]
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.84, "At N=96, depth cannot buy back MLP width.", size=BIG + 5,
             weight="bold")
    fig.text(0.06, 0.745, "Adding four layers to the thin model made it WORSE.",
             size=BIG + 3, weight="bold", color=RED)
    fig.text(0.06, 0.635,
             f"Narrowing dim_feedforward 1024 -> 256 at N=96 costs "
             f"{abs(thin8.delta) * 100:.1f} points ({thin8.solve_rate:.4f} vs\n"
             f"{ref.solve_rate:.4f}, p={thin8.mcnemar_exact_p:.1e}). Going from there to L=12 "
             f"costs another {(thin8.solve_rate - thin12.solve_rate) * 100:.1f},\n"
             f"landing at {thin12.solve_rate:.4f} (vs the wide knee, "
             f"{thin12.delta:+.4f}, p={thin12.mcnemar_exact_p:.1e}).",
             size=MED + 1, va="top")
    fig.text(0.06, 0.455,
             "THE PREDICTION THIS KILLS. At N=48 the thin knee sat 1.5x above the wide one\n"
             "(L*=4 wide, 6 thin, finding 14). I predicted the same factor at N=96, so\n"
             "L*(96, ff=256) ~ 12. Instead L=12/ff=256 clears NONE of the six rho=0.60 depth\n"
             "bins (L=8/ff=256 clears three), so L*(96, ff=256) > 12: the width cost is\n"
             "SUPER-multiplicative in N, which no cell in this project had shown.",
             size=SMALL + 1, va="top", color="0.25")
    fig.text(0.06, 0.245,
             "AND IT IS NOT A PARAMETER STORY. L=12/ff=256 has 5.01M parameters against the\n"
             "wide knee's 6.58M -- 0.76x -- but L=8/ff=256 has only 3.43M and beats it by 3.1\n"
             "points. Within the thin column, +1.58M parameters spent on depth bought a LOSS.\n"
             "Finding 13's 'depth is the cheaper currency' does not survive to N=96.",
             size=SMALL + 1, va="top", color="0.25")
    footer(fig, "job 49816217, COMPLETED 15:50:15, ExitCode 0:0 | commit 3873723 | "
                "scripts/slurm/sudoku2_n96_ff256_lstar.sbatch | seed 0 | shuffle_adj=False, "
                "so every number here is a DECODING score")
    out.savefig(fig)


def page_grid(out, depth, pool, sweep):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "The grid this job was filling in: L* vs (N, ff)", size=BIG,
             weight="bold")
    fig.text(0.06, 0.885,
             "L* is the smallest num_layers reaching 0.90 on depth-T instances at fixed\n"
             "realized density. Before this run, ff=256 was measured at one size only.",
             size=SMALL, va="top")

    rows = [["N=24", "3", "not measured", "--"],
            ["N=48", "4", "6", "1.5x  (finding 14)"],
            ["N=96", "8", "> 12", "> 1.5x  (THIS RUN)"]]
    ax = fig.add_axes([0.06, 0.60, 0.50, 0.21])
    table(ax, ["", "L* at ff=1024", "L* at ff=256", "width factor"], rows,
          [0.20, 0.28, 0.26, 0.30], fontsize=SMALL - 1,
          row_colors={2: RED}, dy=0.21, rule_dy=0.14)

    fig.text(0.60, 0.815, "WHAT 'ff' IS", size=SMALL, weight="bold", va="top")
    fig.text(0.60, 0.775,
             "dim_feedforward: the hidden width of the\n"
             "per-layer MLP. At d_model=256, ff=1024 is\n"
             "the usual 4x expansion and ff=256 is 1x --\n"
             "no expansion at all. Attention, d_model,\n"
             "heads, lr, batch and epochs are untouched.",
             size=SMALL - 1, va="top")

    s = sweep.set_index(["num_layers", "dim_feedforward"])
    prows = []
    for name, tag, _ in CELLS:
        L = int(tag.split("_")[0][1:])
        ff = int(tag.split("ff")[1])
        r = s.loc[(L, ff)]
        pr = pool[pool.checkpoint.str.contains(tag)].iloc[0]
        c, n = cleared(depth, tag)
        prows.append([name, f"{int(r.n_params):,}", f"{r.n_params / 6578026:.2f}x",
                      f"{r.final_val_nll:.4f}", f"{pr.solve_rate:.4f}", f"{c} of {n}"])
    ax = fig.add_axes([0.06, 0.34, 0.90, 0.26])
    table(ax, ["cell", "n_params", "vs wide knee", "val_nll", "pool solve_rate",
               "rho=0.60 bins >= 0.90"], prows,
          [0.17, 0.16, 0.17, 0.13, 0.19, 0.18], fontsize=SMALL - 1,
          row_colors={0: GREEN, 1: ORANGE, 2: RED}, dy=0.24, rule_dy=0.16)

    fig.text(0.06, 0.30,
             "Read the last two columns together. The pool rate falls monotonically down the\n"
             "table, and so does the bin count -- but the parameter count does NOT: the worst\n"
             "cell has 1.46x the parameters of the middle one. Whatever the thin MLP costs, it\n"
             "is not capacity in the counting sense, and more layers do not substitute for it.",
             size=SMALL - 1, va="top")
    fig.text(0.06, 0.17,
             "And val_nll ranks the three cells in almost the REVERSE order: the worst cell on the\n"
             "pool (0.8100) has the BEST teacher-forced nll (1.4108). Next-token loss and free\n"
             "generation come apart here, which is the first thing the mechanism page explains.",
             size=SMALL - 1, va="top", color=RED)
    footer(fig, "L* at N=24 and N=48 from findings 18 and 14; wide knee at N=96 from finding 22 "
                "(job 49639312); all at shuffle_adj=False")
    out.savefig(fig)


def page_depth(out, depth):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "Six depth bins, same instances, three columns", size=BIG,
             weight="bold")
    fig.text(0.06, 0.885,
             "rho=0.60, realized density 0.594 in every bin. The depth eval used the same\n"
             "--n-pool 600 --min-per-bin 40 and generator stream as finding 22, so the bins\n"
             "come out identical row by row and the columns compare bin by bin.",
             size=SMALL, va="top")

    ax = fig.add_axes([0.06, 0.40, 0.44, 0.38])
    for name, tag, col in CELLS:
        g = bins60(depth, tag)
        lo, hi = [], []
        for _, r in g.iterrows():
            a, b = stats.beta.interval(0.95, r.k + 0.5, r.n - r.k + 0.5)
            lo.append(max(0.0, r.solve_rate - a)); hi.append(max(0.0, b - r.solve_rate))
        ax.errorbar(g["T"], g.solve_rate, yerr=[lo, hi], fmt="-o", color=col, ms=5, lw=1.8,
                    capsize=2, label=name)
    ax.axhline(GATE, color="0.35", ls="--", lw=1.2)
    ax.text(8.6, GATE + 0.006, "gate 0.90", size=SMALL - 3, color="0.35")
    ax.set_xlabel("DD depth T", size=SMALL); ax.set_ylabel("solve_rate", size=SMALL)
    ax.set_xticks(range(4, 10)); ax.tick_params(labelsize=SMALL - 2)
    ax.legend(fontsize=SMALL - 3, loc="lower left")

    rows = []
    for name, tag, _ in CELLS:
        g = bins60(depth, tag)
        z, p = ca_trend(g["T"], g.k, g.n)
        rows.append([name, *[f"{v:.3f}" for v in g.solve_rate], f"{p:.3f}"])
    ax2 = fig.add_axes([0.54, 0.56, 0.44, 0.22])
    table(ax2, ["", "T=4", "T=5", "T=6", "T=7", "T=8", "T=9", "CA p"], rows,
          [0.22, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10], fontsize=SMALL - 2,
          row_colors={0: GREEN, 1: ORANGE, 2: RED}, dy=0.20, rule_dy=0.13)
    g = bins60(depth, "L08_ff01024")
    fig.text(0.54, 0.51, f"n per bin: {', '.join(f'T={t}:{n}' for t, n in zip(g['T'], g.n))}",
             size=SMALL - 3, va="top", color=GREY)

    fig.text(0.54, 0.455, "THE TWO THIN CELLS FAIL IN OPPOSITE SHAPES",
             size=SMALL, weight="bold", va="top")
    fig.text(0.54, 0.415,
             "L=8/ff=256 (orange) declines with T the way a\n"
             "capacity-limited reasoner should, clearing the\n"
             "gate at T=4 and T=5 and falling to 0.812 by T=7.\n\n"
             "L=12/ff=256 (red) is FLAT -- 0.829 to 0.868 across\n"
             "the whole range, and its Cochran-Armitage trend is\n"
             "positive, not negative. A model that is equally bad\n"
             "at T=4 and T=9 is not failing at deduction. The next\n"
             "page shows what it is failing at instead.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.33, "THE T=9 BIN IS NOT A RECOVERY", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.29,
             "Both thin cells jump at T=9 (0.953 and 0.860).\n"
             "That bin has n=43, the smallest in the sweep,\n"
             "and the same non-monotone bounce appears in the\n"
             "wide column of finding 22. Deep instances that\n"
             "survive the min-per-bin filter are selected for\n"
             "being generable, so the last bin is not a\n"
             "like-for-like extension of the others. Nothing\n"
             "here rests on it.",
             size=SMALL - 1, va="top")
    footer(fig, "Cochran-Armitage trend over the six ordered bins, the same statistic "
                "findings 11 and 22 reported")
    out.savefig(fig)


def page_mechanism(out, depth):
    """The claim that the deep-thin cell fails at GRAMMAR, with the numbers."""
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "The deep-thin cell stopped writing valid traces", size=BIG,
             weight="bold")
    fig.text(0.06, 0.885,
             "well_formed_rate is the fraction of generations that parse as a trace at all.\n"
             "Dividing solve_rate by it gives the solve rate CONDITIONAL on being well formed:\n"
             "how good the deduction is, once the grammar is out of the way.",
             size=SMALL, va="top")

    ax = fig.add_axes([0.06, 0.44, 0.44, 0.33])
    for name, tag, col in CELLS:
        g = bins60(depth, tag)
        ax.plot(g["T"], g.solve_rate / g.well_formed_rate, "-o", color=col, ms=5, lw=1.8,
                label=name)
        ax.plot(g["T"], g.well_formed_rate, "--s", color=col, ms=4, lw=1.2, alpha=0.55)
    ax.set_xlabel("DD depth T", size=SMALL)
    ax.set_ylabel("solid: solve | well formed\ndashed: well_formed_rate", size=SMALL - 2)
    ax.set_xticks(range(4, 10)); ax.tick_params(labelsize=SMALL - 2)
    ax.legend(fontsize=SMALL - 4, loc="lower left")

    rows = []
    for name, tag, _ in CELLS:
        g = bins60(depth, tag)
        cond = g.solve_rate / g.well_formed_rate
        rows.append([name, f"{g.well_formed_rate.min():.3f}", f"{cond.iloc[0]:.3f}",
                     f"{cond.iloc[-1]:.3f}",
                     f"{float(np.corrcoef(g['T'], cond)[0, 1]):+.2f}"])
    ax2 = fig.add_axes([0.54, 0.600, 0.44, 0.175])
    table(ax2, ["", "worst well\nformed", "solve|wf\nat T=4", "solve|wf\nat T=9", "r vs T"],
          rows, [0.26, 0.19, 0.19, 0.19, 0.13], fontsize=SMALL - 2,
          row_colors={0: GREEN, 1: ORANGE, 2: RED}, dy=0.27, rule_dy=0.26)

    fig.text(0.54, 0.55, "WHAT THE TWO PANELS SAY", size=SMALL, weight="bold", va="top")
    fig.text(0.54, 0.51,
             "L=12/ff=256 (red): well-formedness collapses, but\n"
             "once it parses it solves 0.92-0.98 and IMPROVES\n"
             "with T -- deduction near the wide knee's, output\n"
             "malformed.\n\n"
             "L=8/ff=256 (orange): the reverse. Well formed\n"
             "throughout, conditional rate declining 0.976 ->\n"
             "0.836 -- the genuine depth-limited reasoner, and\n"
             "the cell that defines the width cost.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.215, "WHY THIS MATTERS FOR READING L*", size=SMALL, weight="bold",
             va="top", color=RED)
    fig.text(0.06, 0.175,
             "L*(96, ff=256) > 12 is correct as stated -- L=12 does not clear the gate -- but it is NOT\n"
             "evidence that 12 thin layers cannot do depth-9 deduction. The pool agrees the failure is\n"
             "generative (truncated_rate 0.0007 -> 0.0167, outcome_accuracy 0.986 -> 0.869). At ff=256 a\n"
             "12-layer stack becomes hard to TRAIN TO EMIT the format, and that confounds the knee.",
             size=SMALL - 2, va="top")
    footer(fig, "well_formed_rate from sudoku_depth_eval; pool measures from sudoku_agg_eval")
    out.savefig(fig)


def page_pool(out, pool):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "The shared pool, and a reference that reproduced", size=BIG,
             weight="bold")

    rows = []
    for name, tag, _ in CELLS:
        r = pool[pool.checkpoint.str.contains(tag)].iloc[0]
        rows.append([name, f"{r.solve_rate:.4f}",
                     f"[{r.wilson_lo:.4f}, {r.wilson_hi:.4f}]",
                     "reference" if "1024" in tag else f"{r.delta:+.4f}",
                     "--" if "1024" in tag else f"{int(r.discordant_won)}/{int(r.discordant_lost)}",
                     "--" if "1024" in tag else f"{r.mcnemar_exact_p:.1e}",
                     f"{r.truncated_rate:.4f}", f"{r.outcome_accuracy:.4f}"])
    ax = fig.add_axes([0.06, 0.62, 0.90, 0.22])
    table(ax, ["cell", "solve_rate", "Wilson 95%", "delta", "won/lost", "McNemar p",
               "trunc", "outcome acc"], rows,
          [0.15, 0.12, 0.19, 0.11, 0.11, 0.12, 0.09, 0.11], fontsize=SMALL - 2,
          row_colors={0: GREEN, 1: ORANGE, 2: RED}, dy=0.26, rule_dy=0.18)

    fig.text(0.06, 0.545, "A FREE REPLICATION CHECK", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.505,
             "The reference checkpoint L=8/ff=1024 was scored by TWO independent agg_eval calls\n"
             "a day apart -- once in job 49639312's pool and once here -- and both report\n"
             "0.911333 on 1500 instances. Same seed_start (5e10), same INSTANCE_KEYS, so the\n"
             "pool is reconstructed identically and generation is deterministic. That is a\n"
             "check on the harness, not a result, but it is the kind of thing that is cheap to\n"
             "verify and expensive to discover later.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.32, "WHAT IS PAIRED HERE AND WHAT IS NOT", size=SMALL, weight="bold",
             va="top")
    fig.text(0.06, 0.28,
             "All three cells share the pool: they differ only in architecture, and\n"
             "shuffle_adj is False in all of them, so the prompts are byte-identical and McNemar\n"
             "applies. The L*(N, ff) comparisons ACROSS rows of the previous page's grid are not\n"
             "paired -- different N means a different instance distribution and a different\n"
             "vocabulary -- so the 'width factor' column is a comparison of two knees, each\n"
             "measured on its own ladder, and carries no p-value.",
             size=SMALL - 1, va="top")
    footer(fig, "sudoku_agg_eval --n-instances 1500 --seed-start 50000000000, reference first")
    out.savefig(fig)


def page_prereg(out, depth, pool):
    thin8 = pool[pool.checkpoint.str.contains("L08_ff00256")].iloc[0]
    thin12 = pool[pool.checkpoint.str.contains("L12_ff00256")].iloc[0]
    c8, n8 = cleared(depth, "L08_ff00256")
    c12, _ = cleared(depth, "L12_ff00256")
    g12 = bins60(depth, "L12_ff00256")
    z12, p12 = ca_trend(g12["T"], g12.k, g12.n)

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "Pre-registrations, as resolved", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "Written into scripts/slurm/sudoku2_n96_ff256_lstar.sbatch before the job ran.",
             size=SMALL)

    rows = [
        ["(a)", "L=8/ff=256 does NOT clear all six\n      rho=0.60 bins", "CONFIRMED",
         f"clears {c8} of {n8} (T=4, 5 and 9). The width\n   cost of finding 14 is present at N=96."],
        ["(b)", "L=12/ff=256 DOES clear all six,\n      i.e. L*(96, ff=256) <= 12", "FALSIFIED",
         f"clears {c12} of {n8}. The named falsifier --\n   'a super-multiplicative width cost,\n"
         f"   which needs its own replication' -- holds."],
        ["(c)", "L=12/ff=256 is at least as good as\n      L=8/ff=1024 at 0.76x parameters",
         "FALSIFIED", f"{thin12.solve_rate:.4f} vs 0.9113, {thin12.delta:+.4f},\n"
                      f"   p={thin12.mcnemar_exact_p:.1e}. Resolvably worse."],
        ["(d)", "Inside whichever thin column clears,\n      L*(T) rises with T", "UNRESOLVABLE",
         f"neither thin column clears, so the test\n   has no subject. The L=12 trend is in fact\n"
         f"   POSITIVE (CA z={z12:+.2f}, p={p12:.3f})."],
    ]
    ax = fig.add_axes([0.06, 0.30, 0.90, 0.53])
    table(ax, ["", "claim", "verdict", "evidence"], rows,
          [0.05, 0.34, 0.15, 0.46], fontsize=SMALL - 1,
          row_colors={0: GREEN, 1: RED, 2: RED, 3: GREY}, dy=0.20, rule_dy=0.09)

    fig.text(0.06, 0.245, "THREE OF FOUR WRONG, ALL IN THE SAME DIRECTION.",
             size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.205,
             "(b), (c) and the hypothesis in the header all assumed the N=48 width factor would\n"
             "transfer. It did not, and my pre-registered falsifier for (b) names exactly what\n"
             "happened, which is the only reason this reads as a result rather than a surprise.\n"
             "(d) is a design fault, not a prediction error: I made it conditional on a cell\n"
             "clearing, so a two-cell job that fails twice cannot answer it at all.",
             size=SMALL - 1, va="top")
    footer(fig, "L=16 was deliberately excluded from the grid: it would have pushed training "
                "past the eval stage's start")
    out.savefig(fig)


def page_next(out):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "What this changes, and what comes next", size=BIG, weight="bold")

    fig.text(0.06, 0.875, "CHANGED", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.838,
             "- Every L* in STATUS.md keeps its 'at ff=1024' caveat, and the caveat is now known\n"
             "  to be load-bearing rather than cautious: at N=96 the thin knee is above 12 while\n"
             "  the wide one is 8.\n"
             "- Finding 13's 'depth is the cheaper currency' is demoted to small N. At N=96 the\n"
             "  extra layers did not merely fail to pay -- they cost 3.1 points.\n"
             "- The L* vs (N, ff) grid now has five of six entries, and the missing one (N=24 at\n"
             "  ff=256) is the cheapest cell in the project to fill.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.585, "NOT CHANGED", size=SMALL, weight="bold", va="top", color=RED)
    fig.text(0.06, 0.548,
             "- These are DECODING scores. shuffle_adj=False, so the adjacency rows still arrive\n"
             "  sorted by the neighbour's true colour (finding 16). The whole L* ladder lives in\n"
             "  the leaked task, and job 49705064 has only just established the budget a leak-free\n"
             "  ladder would need.\n"
             "- The L=12 result is confounded by trainability, not just capacity (previous page).\n"
             "- One seed per cell, and L* is read off a 0.90 gate on six bins -- a coarse\n"
             "  instrument that cannot distinguish L*=13 from L*=20.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.295, "NEXT, IN ORDER OF INFORMATION PER GPU-HOUR", size=SMALL,
             weight="bold", va="top")
    fig.text(0.06, 0.258,
             "1. Re-run L=12/ff=256 with a cosine schedule and warmup. If well_formed_rate\n"
             "   recovers to 0.98 and the bins clear, the knee was trainability; if it stays\n"
             "   flat, L*(96, ff=256) > 12 is a statement about computation after all.\n"
             "2. L=10/ff=1024 at N=96, to tighten the wide knee from 'between 6 and 8' to one rung.\n"
             "3. N=24 at ff=256 -- hours, not a day -- to get the width factor at the small end\n"
             "   and see whether it grows with N monotonically.",
             size=SMALL - 1, va="top")
    footer(fig, "checkpoints: d0256_h04_L08_ff00256_lr1e-03_pre and d0256_h04_L12_ff00256_lr1e-03_pre "
                "in sudoku_n96_ff256_2026-10-27")
    out.savefig(fig)


def main():
    set_style()
    os.makedirs(OUT, exist_ok=True)
    depth, pool, sweep = load()
    # N=96 and shuffle_adj=False, matching this run's own instances: the trace is
    # 4x longer than the N=24 decks' and the adjacency leak is still in place.
    ex = example_instance(n_nodes=96, degree=5, n_colors=4, shuffle_adj=False, pick="deep")
    pdf_path = os.path.join(OUT, "ff256.pdf")
    with PdfPages(pdf_path) as pdf:
        out = PdfAndPng(pdf, OUT)
        page_title(out, pool, sweep)
        page_tokenization(out, ex)
        page_trace(out, ex, max_rounds=4,
                   title="One real reasoning trace, at N=96",
                   note="shuffle_adj=False here, so each node's neighbour row is still sorted by\n"
                        "the neighbour's TRUE colour -- finding 16's leak. That is deliberate: this\n"
                        "run extends the leaked-order L* ladder rather than starting a second one.")
        page_grid(out, depth, pool, sweep)
        page_depth(out, depth)
        page_mechanism(out, depth)
        page_pool(out, pool)
        page_prereg(out, depth, pool)
        page_next(out)
    print(f"{out.n} pages -> {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
