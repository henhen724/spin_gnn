"""Slide deck for job 49639312: the N=96 layer ladder, and L*(T) at last rising.

Run as `python -m notebooks.make_n96_lstar_deck` from the repo root.

Carries the tokenization and reasoning-trace pages (hshunt's standing request),
built at N=96 so they describe this run's own instances rather than N=24's.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from notebooks.deck_tokenization import MONO, example_instance, page_tokenization, page_trace
from notebooks.deck_utils import BIG, MED, SMALL, PdfAndPng, footer, set_style

SCRATCH = os.environ["SCRATCH"]
RUN = f"{SCRATCH}/spin_gnn/sudoku_n96_layer_scan_2026-10-26"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sudoku_n96_lstar_2026-10-27")

BLUE, RED, GREEN, ORANGE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "0.45"
GATE = 0.90
LADDER = [4, 6, 8]
COLORS = {4: RED, 6: ORANGE, 8: GREEN, 12: BLUE}


def load():
    d = pd.read_csv(f"{RUN}/depth_by_layers_n96.csv")
    d["L"] = d.checkpoint.str.extract(r"_L(\d+)_").astype(int)
    d["k"] = (d.solve_rate * d.n).round().astype(int)
    pool = pd.read_csv(f"{RUN}/sharedpool_n1500.csv")
    pool["L"] = pool.checkpoint.str.extract(r"_L(\d+)_").astype(int)
    sweep = pd.read_csv(f"{RUN}/sweep.csv")
    return d, pool.set_index("L").sort_index(), sweep.set_index("num_layers").sort_index()


def ca_trend(T, k, n):
    """Cochran-Armitage trend in solve_rate across ordered depth bins.

    The same statistic finding 11 used to report its one weak signal (p=0.088),
    so the two numbers are directly comparable."""
    T, k, n = np.asarray(T, float), np.asarray(k, float), np.asarray(n, float)
    N, K = n.sum(), k.sum()
    p = K / N
    Tbar = (n * T).sum() / N
    z = (k * (T - Tbar)).sum() / np.sqrt(p * (1 - p) * (n * (T - Tbar) ** 2).sum())
    return z, 2 * stats.norm.sf(abs(z))


def split_test(g, cut=6):
    """Pooled shallow (T<=cut) vs deep (T>cut) solve_rate, Fisher exact."""
    sh, dp = g[g["T"] <= cut], g[g["T"] > cut]
    a, b = int(sh.k.sum()), int(sh.n.sum() - sh.k.sum())
    c, e = int(dp.k.sum()), int(dp.n.sum() - dp.k.sum())
    return a / (a + b), c / (c + e), stats.fisher_exact([[a, b], [c, e]])[1]


def table(ax, header, rows, colw, fontsize=SMALL, row_colors=None, dy=0.115, rule_dy=0.09):
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
    ax.set_xlim(0, sum(colw))
    ax.set_ylim(0, 1)


def page_title(out, wide, pool):
    cleared = {L: int((wide[wide.L == L].solve_rate >= GATE).sum()) for L in LADDER}
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.80, "L*(96) = 8, and it is DEFINED.", size=BIG + 7, weight="bold")
    fig.text(0.06, 0.70, "For the first time, L* rises with reasoning depth.",
             size=BIG + 7, weight="bold")
    fig.text(0.06, 0.565,
             f"At rho=0.60, with realized density identical at 0.594 in every bin, the ladder\n"
             f"clears the 0.90 gate in {cleared[4]} of 6 depth bins at L=4, {cleared[6]} of 6 at L=6, "
             f"and {cleared[8]} of 6 at L=8.\n"
             f"So the threshold is 8 -- and the three bins L=6 fails are the three DEEPEST\n"
             f"ones (T=7, 8, 9), which is the shape finding 11 never saw.",
             size=MED + 1, va="top")
    fig.text(0.06, 0.345,
             "Finding 11 measured L*(T) across 20 bins and found it flat: +0.00 layers per round, L*=3 at\n"
             "every T. Those bins were all at N=24, where T spans 1-5 and every cell is either floored or\n"
             "saturated. At N=96, T spans 4-9 and L=6 sits ON the threshold -- the only place the effect\n"
             "can show. It shows there and nowhere else: the trend is significant only at L=6.",
             size=SMALL + 1, va="top", color="0.25")
    fig.text(0.06, 0.155,
             f"Leaked-order run (shuffle_adj=False), so every number here is a DECODING score, not a\n"
             f"deduction score (findings 16-17). The shared-pool best is still L=12 at "
             f"{pool.loc[12, 'solve_rate']:.4f}; L=8 ties it\n"
             f"({pool.loc[8, 'solve_rate']:.4f}, delta {pool.loc[8, 'delta']:+.4f}, "
             f"p={pool.loc[8, 'mcnemar_exact_p']:.3f}) at 1.5x fewer layers.",
             size=SMALL, va="top", color=GREY)
    footer(fig, "job 49639312, COMPLETED 17:45:48, ExitCode 0:0 | "
                "scripts/slurm/sudoku2_n96_layer_scan_80k.sbatch")
    out.savefig(fig)


def page_ladder(out, wide, shallow, sweep):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "The ladder, bin by bin", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "L* is the smallest num_layers that reaches 0.90 on instances of DD depth T at FIXED "
             "density.", size=SMALL)

    Ts = sorted(wide["T"].unique())
    rows = []
    for L in LADDER:
        g = wide[wide.L == L].set_index("T")
        cells = [f"{g.loc[t, 'solve_rate']:.3f}" for t in Ts]
        rows.append([f"L={L}", *cells,
                     f"{int((g.solve_rate >= GATE).sum())}/6",
                     f"{sweep.loc[L, 'final_val_nll']:.4f}"])
    ax = fig.add_axes([0.06, 0.56, 0.90, 0.23])
    table(ax, ["", *[f"T={t}" for t in Ts], "bins>=0.90", "val_nll"], rows,
          [0.09, *([0.085] * len(Ts)), 0.13, 0.12], fontsize=SMALL,
          row_colors={i: COLORS[L] for i, L in enumerate(LADDER)}, dy=0.22, rule_dy=0.15)
    # NB `wide.T` is DataFrame.transpose; the depth column always needs brackets.
    per_bin = wide.groupby("T").n.first().astype(int)
    fig.text(0.06, 0.515,
             f"n per bin: {', '.join(f'T={t}:{v}' for t, v in per_bin.items())}"
             f"   ({int(per_bin.sum())} instances per row, realized density 0.594 in every bin)",
             size=SMALL - 2, va="top", color=GREY)

    ax2 = fig.add_axes([0.07, 0.125, 0.36, 0.25])
    for L in LADDER:
        g = shallow[shallow.L == L].sort_values("T")
        ax2.plot(g["T"], g.solve_rate, "-o", color=COLORS[L], ms=4, lw=1.6, label=f"L={L}")
    ax2.axhline(GATE, color="0.6", ls=":", lw=1.0)
    ax2.set_title("rho=0.75, the shallow control", size=SMALL)
    ax2.set_xlabel("DD depth T", size=SMALL - 1); ax2.set_ylabel("solve_rate", size=SMALL - 1)
    ax2.set_ylim(0.6, 1.02); ax2.tick_params(labelsize=SMALL - 3); ax2.legend(fontsize=SMALL - 4)

    fig.text(0.50, 0.38,
             "WHY THE SHALLOW CONTROL MATTERS",
             size=SMALL, weight="bold", va="top")
    fig.text(0.50, 0.335,
             "At rho=0.75 every cell including L=4 clears the gate at\n"
             "every T, and L=8 scores 1.000 in all four bins. The ladder\n"
             "is therefore not measuring a model that is simply bad: the\n"
             "same three checkpoints are at ceiling on easier instances.\n"
             "Whatever L=4 and L=6 lose at rho=0.60 they lose to the\n"
             "combination of density and depth, not to competence.",
             size=SMALL - 1, va="top")
    footer(fig, "sudoku_depth_eval --rho 0.75 0.60; all three cells at n_train=80000, 150 epochs")
    out.savefig(fig)


def page_lstar(out, wide):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "Where L* actually moves", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "Only L=6 is near the gate, and only L=6 has a depth trend. That is the whole result.",
             size=SMALL)

    ax = fig.add_axes([0.07, 0.42, 0.40, 0.40])
    for L in LADDER:
        g = wide[wide.L == L].sort_values("T")
        lo, hi = [], []
        for _, r in g.iterrows():
            a, b = stats.beta.interval(0.95, r.k + 0.5, r.n - r.k + 0.5)
            lo.append(r.solve_rate - a); hi.append(b - r.solve_rate)
        ax.errorbar(g["T"], g.solve_rate, yerr=[lo, hi], fmt="-o", color=COLORS[L],
                    ms=4, lw=1.8, capsize=2, label=f"L={L}")
    ax.axhline(GATE, color="0.35", ls="--", lw=1.2)
    ax.text(8.6, GATE + 0.008, "0.90 gate", size=SMALL - 3, color="0.35")
    ax.set_xlabel("DD depth T", size=SMALL); ax.set_ylabel("solve_rate", size=SMALL)
    ax.set_title("rho=0.60, density fixed at 0.594", size=SMALL + 1)
    ax.set_ylim(0.6, 1.0); ax.set_xticks(sorted(wide["T"].unique()))
    ax.tick_params(labelsize=SMALL - 2); ax.legend(fontsize=SMALL - 3, loc="lower left")

    rows = []
    for L in LADDER:
        g = wide[wide.L == L]
        z, p = ca_trend(g["T"], g["k"], g["n"])
        s, d, pf = split_test(g)
        rows.append([f"L={L}", f"{g.k.sum()}/{g.n.sum()} = {g.k.sum() / g.n.sum():.4f}",
                     f"{z:+.2f}", f"{p:.4f}", f"{s:.4f}", f"{d:.4f}", f"{pf:.4f}"])
    ax2 = fig.add_axes([0.53, 0.52, 0.44, 0.30])
    table(ax2, ["", "pooled", "CA z", "CA p", "T<=6", "T>=7", "Fisher p"], rows,
          [0.10, 0.26, 0.11, 0.14, 0.13, 0.13, 0.13], fontsize=SMALL - 1,
          row_colors={i: COLORS[L] for i, L in enumerate(LADDER)}, dy=0.20, rule_dy=0.14)

    fig.text(0.53, 0.45, "THE PATTERN IS THE EVIDENCE", size=SMALL, weight="bold", va="top")
    fig.text(0.53, 0.405,
             "L=4 is on the floor (0.75) and L=8 is above the gate\n"
             "everywhere (0.937), so neither CAN show a depth trend --\n"
             "and neither does (p=0.42, p=0.81). L=6 sits at 0.897,\n"
             "within a hair of the gate, and it is the one cell with a\n"
             "trend (CA p=0.055) and a shallow-vs-deep gap (+0.054,\n"
             "p=0.072). A depth cost that appears only where the\n"
             "threshold is, is what a real threshold effect looks like.",
             size=SMALL - 1, va="top")

    # Everything in this left column must stay clear of x=0.53, where the table
    # and the block beside it start -- hence the narrow wrap rather than the
    # full-width paragraphs used elsewhere.
    fig.text(0.07, 0.355, "L*(T) = 6 FOR T<=6, 8 FOR T>=7: A RISE OF ONE RUNG.",
             size=SMALL, weight="bold", va="top")
    fig.text(0.07, 0.315,
             "Against finding 11's +0.00 layers per round over 20\n"
             "bins at N=24. Not established: p=0.055 and p=0.072\n"
             "are marginal, from one seed, and the two tests are\n"
             "not independent. The claim is narrower -- the shape\n"
             "finding 11 looked for appears as soon as the measu-\n"
             "rement has a cell at the threshold and a T range\n"
             "wide enough to split.",
             size=SMALL - 1, va="top")
    fig.text(0.07, 0.125,
             "That is also a warning for the planned leak-free L*(T) ladder at N=24: T spans 2-5 there,\n"
             "which is the range that produced the flat answer. It may need N=48 to be informative.",
             size=SMALL - 1, va="top", color=RED)
    footer(fig, "error bars are 95% Jeffreys intervals; Cochran-Armitage is the same trend test "
                "finding 11 reported at p=0.088")
    out.savefig(fig)


def page_across_n(out, pool, wide):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "L* across problem size, and the cost of the threshold",
             size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "All at d_model=256, ff=1024 -- finding 14 showed L* moves when the MLP narrows, so the "
             "width is part of the claim.", size=SMALL)

    rows = [["N=24", "3", "finding 11/14", "0.9960", "L=12"],
            ["N=48", "4", "finding 14 (49509466/49523504)", "0.9573", "L=8, ff=256"],
            ["N=96", "8", "THIS RUN (49639312)", f"{pool.loc[12, 'solve_rate']:.4f}", "L=12"]]
    ax = fig.add_axes([0.06, 0.60, 0.88, 0.20])
    table(ax, ["problem", "L* at ff=1024", "measured in", "best solve_rate", "best cell"],
          rows, [0.12, 0.20, 0.34, 0.20, 0.14], fontsize=SMALL,
          row_colors={2: GREEN}, dy=0.26, rule_dy=0.18)

    fig.text(0.06, 0.545,
             "L* = 3, 4, 8 for N = 24, 48, 96. Mean DD depth grows with N, so this is the across-problem "
             "version of\nthe same claim the T bins make within N=96 -- and the two now agree for the "
             "first time.",
             size=SMALL - 1, va="top")

    ax2 = fig.add_axes([0.07, 0.17, 0.37, 0.28])
    ls = sorted(pool.index)
    ax2.bar([str(L) for L in ls], [pool.loc[L, "solve_rate"] for L in ls],
            color=[COLORS[L] for L in ls], width=0.6)
    for L in ls:
        ax2.text(str(L), pool.loc[L, "solve_rate"] + 0.012, f"{pool.loc[L, 'solve_rate']:.3f}",
                 ha="center", size=SMALL - 3)
    ax2.axhline(GATE, color="0.35", ls="--", lw=1.0)
    ax2.set_ylim(0.6, 1.0); ax2.set_xlabel("num_layers", size=SMALL - 1)
    ax2.set_ylabel("shared-pool solve_rate", size=SMALL - 1); ax2.tick_params(labelsize=SMALL - 3)

    fig.text(0.50, 0.44, "THRESHOLD IS NOT OPTIMUM -- pre-registration (d)",
             size=SMALL, weight="bold", va="top")
    fig.text(0.50, 0.395,
             f"On the shared n=1500 pool, paired against cell B (L=12):\n"
             f"  L=4   {pool.loc[4, 'solve_rate']:.4f}   {pool.loc[4, 'delta']:+.4f}  "
             f"p={pool.loc[4, 'mcnemar_exact_p']:.1e}\n"
             f"  L=6   {pool.loc[6, 'solve_rate']:.4f}   {pool.loc[6, 'delta']:+.4f}  "
             f"p={pool.loc[6, 'mcnemar_exact_p']:.1e}\n"
             f"  L=8   {pool.loc[8, 'solve_rate']:.4f}   {pool.loc[8, 'delta']:+.4f}  "
             f"p={pool.loc[8, 'mcnemar_exact_p']:.3f}\n"
             f"  L=12  {pool.loc[12, 'solve_rate']:.4f}   reference\n\n"
             f"I pre-registered L=8 >= L=12. It is a TIE, not a win: the\n"
             f"point estimate favours L=12 by 0.016 and the test cannot\n"
             f"resolve it. So the threshold (8) is below the optimum (12)\n"
             f"but nothing above 8 is resolvably better -- which is the\n"
             f"separation the pre-registration was for, reached by the\n"
             f"weaker of the two routes.",
             size=SMALL - 1, va="top", **MONO)
    footer(fig, "sudoku_agg_eval, shared pool at seed_start=5e10, n=1500, paired McNemar exact; "
                "N=96 pool is its own instance set")
    out.savefig(fig)


def page_prereg(out, wide, pool):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "Pre-registrations, as resolved", size=BIG, weight="bold")
    fig.text(0.06, 0.875,
             "Written into scripts/slurm/sudoku2_n96_layer_scan_80k.sbatch before the job ran.",
             size=SMALL)

    z6, p6 = ca_trend(wide[wide.L == 6]["T"], wide[wide.L == 6]["k"], wide[wide.L == 6]["n"])
    s6, d6, pf6 = split_test(wide[wide.L == 6])
    rows = [
        ["(a)", "L*(96) >= 6", "CONFIRMED",
         "L=4 clears 0 of 6 bins; L* = 8"],
        ["(b)", "L*(96) is DEFINED inside {4, 6, 8}", "CONFIRMED",
         "L=8 clears 6 of 6 bins at rho=0.60"],
        ["(c)", "L*(T) is flat in T, as in finding 11", "FALSIFIED\n   (marginally)",
         f"L=6 clears T=4,5,6 and fails T=7,8,9;\n"
         f"   CA p={p6:.3f}, shallow-vs-deep {d6:.3f} vs {s6:.3f},\n"
         f"   Fisher p={pf6:.3f}. Only at L=6."],
        ["(d)", "L=8 >= L=12, separating the threshold\n      from the optimum",
         "TIE", f"{pool.loc[8, 'solve_rate']:.4f} vs {pool.loc[12, 'solve_rate']:.4f}, "
                f"delta {pool.loc[8, 'delta']:+.4f},\n   p={pool.loc[8, 'mcnemar_exact_p']:.3f} "
                f"-- not resolvable either way"],
    ]
    ax = fig.add_axes([0.06, 0.38, 0.90, 0.43])
    table(ax, ["", "claim", "verdict", "evidence"], rows,
          [0.05, 0.40, 0.15, 0.40], fontsize=SMALL - 1,
          row_colors={0: GREEN, 1: GREEN, 2: RED, 3: "0.35"}, dy=0.21, rule_dy=0.11)

    fig.text(0.06, 0.295,
             "(c) is the one worth being careful about: it is the prediction I most wanted to be wrong.",
             size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.25,
             "I pre-registered flatness because that is what finding 11 measured, and a result that\n"
             "agrees with the prediction you wanted to fail deserves more scepticism than one that does\n"
             "not. Both tests are marginal (0.055, 0.072), they are not independent, and this is one\n"
             "seed. The reason I report it as falsified rather than unresolved is the pattern across\n"
             "cells: the trend appears at the one layer count near the gate and at neither of the two\n"
             "that are pinned, which is a prediction flatness does not make.",
             size=SMALL - 1, va="top")
    footer(fig, "my record on pre-registrations in this project: 2 confirmed, 1 falsified, "
                "1 tie, on this job")
    out.savefig(fig)


def page_caveats(out, pool):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.06, 0.93, "What this changes, and what it does not", size=BIG, weight="bold")

    fig.text(0.06, 0.855, "WHAT IT CHANGES", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.815,
             "- L*(96) was UNDEFINED after finding 15 (the control stalled at 0.8267 and cleared no\n"
             "  bin). It is now 8, so the N ladder reads L* = 3, 4, 8 at N = 24, 48, 96.\n"
             "- Finding 11's flat L*(T) is demoted from 'depth does not track depth' to 'not resolvable\n"
             "  at N=24': at N=96, with a cell at the gate and T spanning 4-9, it rises by one rung.\n"
             "- The N=96 question is no longer 'can it learn at all' but 'how deep must it be'.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.60, "WHAT IT DOES NOT", size=SMALL, weight="bold", va="top", color=RED)
    fig.text(0.06, 0.56,
             "- This is a LEAKED-ORDER run (shuffle_adj=False). Every solve_rate here is a decoding\n"
             "  score. Finding 16 showed this exact family of checkpoints collapses to 0.03-0.15 when\n"
             "  adj[v] is permuted, so none of these numbers is a deduction score, and L* measured on\n"
             "  a shortcut task need not equal L* on the real one.\n"
             "- p=0.055 and p=0.072 are marginal, from one seed, on two non-independent tests.\n"
             "- L* depends on the MLP width (finding 14): every L* above is 'at ff=1024'. A thin-deep\n"
             "  N=96 cell could move it, and has not been run.",
             size=SMALL - 1, va="top")

    fig.text(0.06, 0.305, "WHAT I WOULD DO NEXT", size=SMALL, weight="bold", va="top")
    fig.text(0.06, 0.265,
             "1. The leak-free L*(T) ladder, but at N=48 rather than only N=24 -- this run is the\n"
             "   argument for it. N=24's T range is 2-5, the range that produced the flat answer.\n"
             "2. L=10 at N=96 would tighten L* from 'between 6 and 8' to a single rung, and is one cell.\n"
             "3. A thin-deep N=96 cell (ff=256), which finding 13 says should match L=8/ff=1024 at\n"
             "   roughly a third of the parameters.",
             size=SMALL - 1, va="top")
    footer(fig, "next: the budget from 49705064 decides whether the leak-free ladder can run at all")
    out.savefig(fig)


def main():
    set_style()
    os.makedirs(OUT, exist_ok=True)
    depth, pool, sweep = load()
    wide = depth[depth.rho_requested == 0.60].copy()
    shallow = depth[depth.rho_requested == 0.75].copy()
    # N=96 and shuffle_adj=False, so the tokenization and trace pages describe
    # THIS run's instances -- including the leak, which is visible in the graph
    # section and is the reason these are decoding scores.
    ex = example_instance(n_nodes=96, shuffle_adj=False, pick="median")
    pdf_path = os.path.join(OUT, "n96_lstar.pdf")
    with PdfPages(pdf_path) as pdf:
        out = PdfAndPng(pdf, OUT)
        page_title(out, wide, pool)
        page_tokenization(out, ex)
        page_trace(out, ex)
        page_ladder(out, wide, shallow, sweep)
        page_lstar(out, wide)
        page_across_n(out, pool, wide)
        page_prereg(out, wide, pool)
        page_caveats(out, pool)
    print(f"{out.n} pages -> {pdf_path} ({os.path.getsize(pdf_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
